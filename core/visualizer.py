"""
visualizer_fast.py — drop-in faster PNG writer (no Matplotlib)

This module provides functions to generate audio visualizer sequences from MP3 files.
It includes both a traditional batch processing version and a streaming version for
better memory efficiency with large files.

Functions:
- write_visualizer_sequence: Traditional batch processing (loads entire audio file)
- write_visualizer_sequence_streaming: Streaming processing (processes audio in chunks)
- generate_visualizer_clip: Compatibility function for MoviePy clips
- generate_visualizer_clip_realtime: Real-time processing version for MoviePy clips

The streaming version is more memory-efficient and suitable for real-time or large file
processing, though it normalizes each chunk independently which may result in slightly
different visualizations compared to the batch version.

Visualizer Positioning:
The visualizer baseline is placed at the vertical center of the video (x=0, y=0.5*video_height).
"""
import os, math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from moviepy.editor import AudioFileClip
from PIL import Image
from tqdm import tqdm

# --- Resource Management ---
CPU_USAGE_PERCENT = 0.75
MAX_WORKERS = max(1, int(os.cpu_count() * CPU_USAGE_PERCENT))

@dataclass(frozen=True)
class Band:
    name: str
    low_freq: float
    high_freq: float
    color_rgba: Tuple[int, int, int, int]  # pre-baked with opacity

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_bands(start_freq: float = 80.0,
                   end_freq: float = 3000.0,
                   step: float = 20.0,
                   opacity: float = 0.6) -> Tuple[Band, ...]:
    bands = []
    f_low = start_freq
    a = max(0, min(1, opacity))
    A = int(round(255 * a))
    while f_low < end_freq:
        f_high = f_low + step
        t = (f_low - start_freq) / (end_freq - start_freq)
        if t < 0.5:
            r, g, b = int(255 * (1 - t * 2)), int(255 * (t * 2)), 32
        else:
            r, g, b = 32, int(255 * (1 - (t - 0.5) * 2)), int(255 * ((t - 0.5) * 2))
        bands.append(Band(f"{int(f_low)}–{int(f_high)}", f_low, f_high, (r, g, b, A)))
        f_low = f_high
    return tuple(bands)

def _frame_indices(total: int, n_chunks: int) -> Iterable[range]:
    if n_chunks <= 1: return [range(0, total)]
    size = math.ceil(total / n_chunks)
    return [range(i, min(i + size, total)) for i in range(0, total, size)]

def _stft_magnitudes(audio: np.ndarray,
                     sr: int,
                     fps: int,
                     bands: Tuple[Band, ...],
                     silence_threshold: float = 0.01) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    win = int(sr / fps)
    if win <= 0: win = 1470  # ~44100/30
    n_frames = int(math.ceil(len(audio) / win))
    pad_len = n_frames * win - len(audio)
    if pad_len > 0:
        audio = np.pad(audio, (0, pad_len), mode="constant")
    frames = audio.reshape(n_frames, win)
    spec = np.fft.rfft(frames, axis=1)
    mags = np.abs(spec)
    freqs = np.fft.rfftfreq(win, d=1.0/sr)
    idx_list = [np.where((freqs >= b.low_freq) & (freqs <= b.high_freq))[0] for b in bands]
    idx_list = [idx if len(idx) > 0 else np.array([0]) for idx in idx_list]
    band_mags = np.stack([mags[:, idx].mean(axis=1) for idx in idx_list], axis=1)

    # Normalisasi per frame
    max_per_frame = band_mags.max(axis=1, keepdims=True)
    max_per_frame[max_per_frame == 0] = 1.0
    band_mags = band_mags / max_per_frame

    # Thresholding supaya bar diam saat hening
    band_mags[band_mags < silence_threshold] = 0.0

    return band_mags.astype(np.float32)

def _render_chunk(args):
    (chunk, out_dir, W, H, H_vis, gap_px, bar_w, colors, mags) = args
    n_bands = colors.shape[0]
    count = 0
    chunk_list = list(chunk) if isinstance(chunk, range) else chunk
    # posisikan visualizer baseline di tengah layar (x=0, y=0.5*video_height)
    y_pos = int(H * 0)
    for idx, i in enumerate(chunk_list):
        if idx >= mags.shape[0]:
            break
        vals = mags[idx]
        full_frame = np.zeros((H, W, 4), dtype=np.uint8)
        vis_frame = np.zeros((H_vis, W, 4), dtype=np.uint8)
        x = gap_px
        for b in range(n_bands):
            h = int(vals[b] * H_vis)
            if h <= 0:
                x += bar_w + gap_px
                continue
            y0 = H_vis - h
            y1 = H_vis
            x0 = x
            x1 = min(W, x + bar_w)
            vis_frame[y0:y1, x0:x1, :] = colors[b]
            x += bar_w + gap_px
        start_y = y_pos
        end_y = min(y_pos + H_vis, H)
        start_x = 0
        end_x = W
        if end_y > start_y:
            full_frame[start_y:end_y, start_x:end_x, :] = vis_frame[0:(end_y-start_y), :, :]
        img = Image.fromarray(full_frame, mode="RGBA")
        img.save(os.path.join(out_dir, f"vis_frame_{i:08d}.png"), format="PNG")
        count += 1
    return (chunk_list[0] if chunk_list else 0, count)

def write_visualizer_sequence(
    mp3_path: str,
    fps: int = 30,
    resolution=(1280, 720),
    opacity: float = 0.6,
    scale_height: float = 0.3,
    out_pattern: str = None,
    silence_threshold: float = 0.01
):
    if not out_pattern:
        raise ValueError("out_pattern is required")
    out_dir = os.path.dirname(out_pattern)
    os.makedirs(out_dir, exist_ok=True)
    sr = 44100
    clip = AudioFileClip(mp3_path)
    audio = clip.to_soundarray(fps=sr)
    duration = clip.duration
    clip.close()
    total_frames = int(math.ceil(duration * fps))
    if total_frames <= 0:
        return 0
    bands = generate_bands(80, 3000, 20, opacity)
    mags = _stft_magnitudes(audio, sr, fps, bands, silence_threshold)
    n_frames, n_bands = mags.shape
    W, H = int(resolution[0]), int(resolution[1])
    H_vis = max(1, int(H * scale_height))
    gap_px = max(1, int(W * 0.002))
    total_gap = gap_px * (n_bands + 1)
    bar_w = max(1, (W - total_gap) // n_bands)
    colors = np.array([b.color_rgba for b in bands], dtype=np.uint8)
    chunks = _frame_indices(total_frames, MAX_WORKERS)
    tasks = [(chunk, out_dir, W, H, H_vis, gap_px, bar_w, colors, mags) for chunk in chunks]
    written = 0
    with tqdm(total=total_frames, desc="🟡 Rendering Visualizer → PNG (fast)", unit="frame", colour="yellow") as pbar:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_render_chunk, t) for t in tasks]
            for f in as_completed(futs):
                _, count = f.result()
                written += count
                pbar.update(count)
    return written

def write_visualizer_sequence_streaming(
    mp3_path: str,
    fps: int = 30,
    resolution=(1280, 720),
    opacity: float = 0.6,
    scale_height: float = 0.3,
    out_pattern: str = None,
    chunk_duration: float = 5.0,
    silence_threshold: float = 0.01
):
    if not out_pattern:
        raise ValueError("out_pattern is required")
    out_dir = os.path.dirname(out_pattern)
    os.makedirs(out_dir, exist_ok=True)
    sr = 44100
    clip = AudioFileClip(mp3_path)
    duration = clip.duration
    total_frames = int(math.ceil(duration * fps))
    if total_frames <= 0:
        clip.close()
        return 0
    bands = generate_bands(80, 3000, 20, opacity)
    n_bands = len(bands)
    W, H = int(resolution[0]), int(resolution[1])
    H_vis = max(1, int(H * scale_height))
    gap_px = max(1, int(W * 0.002))
    total_gap = gap_px * (n_bands + 1)
    bar_w = max(1, (W - total_gap) // n_bands)
    colors = np.array([b.color_rgba for b in bands], dtype=np.uint8)
    chunk_size = int(sr * chunk_duration)
    written = 0
    with tqdm(total=total_frames, desc="🟡 Rendering Visualizer (streaming)", unit="frame", colour="yellow") as pbar:
        start_sample = 0
        frame_offset = 0
        while start_sample < int(duration * sr):
            end_sample = min(start_sample + chunk_size, int(duration * sr))
            try:
                if end_sample >= int(duration * sr):
                    audio_chunk = clip.subclip(start_sample/sr).to_soundarray(fps=sr)
                else:
                    audio_chunk = clip.subclip(start_sample/sr, end_sample/sr).to_soundarray(fps=sr)

                mags_chunk = _stft_magnitudes(audio_chunk, sr, fps, bands, silence_threshold)

                n_chunk_frames = min(mags_chunk.shape[0], total_frames - frame_offset)
                start_frame_idx = frame_offset
                end_frame_idx = frame_offset + n_chunk_frames
                chunk_range = range(start_frame_idx, end_frame_idx)
                if mags_chunk.shape[0] > n_chunk_frames:
                    mags_chunk = mags_chunk[:n_chunk_frames]
                args = (chunk_range, out_dir, W, H, H_vis, gap_px, bar_w, colors, mags_chunk)
                _, count = _render_chunk(args)
                written += count
                pbar.update(count)
                start_sample = end_sample
                frame_offset += n_chunk_frames
            except Exception as e:
                print(f"Warning: Error processing chunk {start_sample/sr}-{end_sample/sr}: {e}")
                start_sample = end_sample
                continue
        clip.close()
    return written

def generate_visualizer_clip(mp3_path: str, fps: int = 30, resolution=(1280, 720), 
                            opacity: float = 0.6, scale_height: float = 0.3,
                            silence_threshold: float = 0.01):
    import tempfile
    from moviepy.editor import ImageSequenceClip
    temp_dir = tempfile.mkdtemp(prefix="vis_frames_")
    out_pattern = os.path.join(temp_dir, "vis_frame_%08d.png")
    try:
        written = write_visualizer_sequence(
            mp3_path,
            fps=fps,
            resolution=resolution,
            opacity=opacity,
            scale_height=scale_height,
            out_pattern=out_pattern,
            silence_threshold=silence_threshold
        )
        if written > 0:
            clip = ImageSequenceClip(temp_dir, fps=fps)
            return clip
        else:
            from moviepy.editor import ColorClip
            return ColorClip(size=resolution, color=(0, 0, 0), duration=1)
    except Exception as e:
        from moviepy.editor import ColorClip
        print(f"Warning: Could not generate visualizer clip: {e}")
        return ColorClip(size=resolution, color=(0, 0, 0), duration=1)

def generate_visualizer_clip_realtime(mp3_path: str, fps: int = 30, resolution=(1280, 720), 
                                     opacity: float = 0.6, scale_height: float = 0.3,
                                     silence_threshold: float = 0.01):
    import tempfile
    from moviepy.editor import ImageSequenceClip
    temp_dir = tempfile.mkdtemp(prefix="vis_frames_rt_")
    out_pattern = os.path.join(temp_dir, "vis_frame_%08d.png")
    try:
        written = write_visualizer_sequence_streaming(
            mp3_path,
            fps=fps,
            resolution=resolution,
            opacity=opacity,
            scale_height=scale_height,
            out_pattern=out_pattern,
            chunk_duration=3.0,
            silence_threshold=silence_threshold
        )
        if written > 0:
            clip = ImageSequenceClip(temp_dir, fps=fps)
            return clip
        else:
            from moviepy.editor import ColorClip
            return ColorClip(size=resolution, color=(0, 0, 0), duration=1)
    except Exception as e:
        from moviepy.editor import ColorClip
        print(f"Warning: Could not generate real-time visualizer clip: {e}")
        return ColorClip(size=resolution, color=(0, 0, 0), duration=1)
