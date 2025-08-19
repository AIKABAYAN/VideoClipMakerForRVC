# visualizer.py
import math
import numpy as np
import os
from typing import Tuple, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from moviepy.editor import AudioFileClip
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patheffects as path_effects
from PIL import Image
from tqdm import tqdm

# --- Resource Management ---
CPU_USAGE_PERCENT = 0.50
MAX_WORKERS = max(1, int(os.cpu_count() * CPU_USAGE_PERCENT))


class Band:
    def __init__(self, name: str, low_freq: float, high_freq: float, color: str):
        self.name = name
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.color = color


class AudioAnalyzer:
    def __init__(self, audio_path, bands: Tuple['Band', ...]):
        self.audio_path = audio_path
        self.bands = bands
        self.audio_clip = AudioFileClip(audio_path)
        self.audio_fps = 44100
        self.audio_array = self.audio_clip.to_soundarray(fps=self.audio_fps)
        self.duration = self.audio_clip.duration

    def magnitudes_at_time(self, t: float):
        start_idx = int(t * self.audio_fps)
        end_idx = start_idx + int(self.audio_fps / 30)
        if end_idx > len(self.audio_array):
            end_idx = len(self.audio_array)

        window = self.audio_array[start_idx:end_idx]
        if window.ndim == 2:
            window = window.mean(axis=1)

        fft_result = np.fft.rfft(window)
        magnitudes = np.abs(fft_result)
        freqs = np.fft.rfftfreq(len(window), d=1.0 / self.audio_fps)

        results = []
        for band in self.bands:
            idx = np.where((freqs >= band.low_freq) & (freqs <= band.high_freq))[0]
            mag = magnitudes[idx].mean() if len(idx) > 0 else 0.0
            results.append(mag)

        max_mag = max(results) if results else 1.0
        if max_mag > 0:
            results = [v / max_mag for v in results]
        return results

    def close(self):
        try:
            self.audio_clip.close()
        except Exception:
            pass


def generate_bands(start_freq: float = 80.0,
                   end_freq: float = 3000.0,
                   step: float = 20.0) -> Tuple[Band, ...]:
    bands = []
    f_low = start_freq
    while f_low < end_freq:
        f_high = f_low + step
        t = (f_low - start_freq) / (end_freq - start_freq)
        if t < 0.5:
            r, g, b = int(255 * (1 - t * 2)), int(255 * (t * 2)), 32
        else:
            r, g, b = 32, int(255 * (1 - (t - 0.5) * 2)), int(255 * ((t - 0.5) * 2))
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        bands.append(Band(f"{int(f_low)}–{int(f_high)}", f_low, f_high, hex_color))
        f_low = f_high
    return tuple(bands)


def _render_and_save_chunk(args):
    """
    Worker: renders a range of frames and writes PNG files directly to disk.
    Returns (first_index, count) for progress tallying.
    """
    frame_indices, mp3_path, bands, resolution, opacity, scale_height, fps, out_dir = args

    analyzer = AudioAnalyzer(mp3_path, bands)

    fig_w, fig_h = resolution
    fig = Figure(figsize=(fig_w / 100, (fig_h * scale_height) / 100),
                 dpi=100, facecolor='none')
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(bands) - 0.5)
    ax.set_xticks([])
    ax.axis('off')

    bars = ax.bar(range(len(bands)),
                  [0.0] * len(bands),
                  color=[b.color for b in bands],
                  alpha=opacity)
    for bar in bars:
        bar.set_path_effects([
            path_effects.withStroke(linewidth=12,
                                    foreground=bar.get_facecolor(),
                                    alpha=opacity * 0.15),
            path_effects.withStroke(linewidth=8,
                                    foreground=bar.get_facecolor(),
                                    alpha=opacity * 0.25),
            path_effects.withStroke(linewidth=6,
                                    foreground=bar.get_facecolor(),
                                    alpha=opacity * 0.40),
            path_effects.Normal()
        ])

    canvas = FigureCanvasAgg(fig)

    # Render and save each frame directly
    saved = 0
    for i in frame_indices:
        t = i / fps
        vals = analyzer.magnitudes_at_time(t)
        for bar, v in zip(bars, vals):
            bar.set_height(float(v))

        canvas.draw()

        # ARGB -> RGBA
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, [1, 2, 3, 0]]

        # Save PNG with alpha
        img = Image.fromarray(buf, mode="RGBA")
        out_path = os.path.join(out_dir, f"vis_frame_{i:08d}.png")
        img.save(out_path, format="PNG")
        saved += 1

    analyzer.close()
    return (frame_indices[0], saved)


def _split_into_chunks(total: int, n_chunks: int) -> Iterable[range]:
    if n_chunks <= 1:
        return [range(0, total)]
    chunk_size = math.ceil(total / n_chunks)
    return [range(i, min(i + chunk_size, total))
            for i in range(0, total, chunk_size)]


def write_visualizer_sequence(
    mp3_path: str,
    fps: int = 30,
    resolution=(1280, 720),
    opacity: float = 0.5,
    scale_height: float = 0.2,
    out_pattern: str = None,
):
    """
    Render the visualizer as a PNG sequence directly to disk (no in-memory clip).

    Args:
        mp3_path: path to audio
        fps: frames per second
        resolution: (W, H) of the final video
        opacity: bar opacity
        scale_height: height of the visualizer relative to H (0..1)
        out_pattern: e.g. '/tmp/vis_frames/vis_frame_%08d.png'
    """
    if not out_pattern:
        raise ValueError("out_pattern is required (e.g., '/tmp/vis_frames/vis_frame_%08d.png').")

    out_dir = os.path.dirname(out_pattern)
    os.makedirs(out_dir, exist_ok=True)

    bands = generate_bands(80, 3000, 20)
    # Use lightweight AudioFileClip just to get duration
    tmp_clip = AudioFileClip(mp3_path)
    duration = tmp_clip.duration
    tmp_clip.close()

    total_frames = int(math.ceil(duration * fps))
    if total_frames <= 0:
        return 0

    frame_chunks = _split_into_chunks(total_frames, MAX_WORKERS)
    tasks = [(chunk, mp3_path, bands, resolution, opacity, scale_height, fps, out_dir)
             for chunk in frame_chunks]

    written_total = 0
    with tqdm(total=total_frames,
              desc="🟡 Rendering Visualizer → PNG",
              unit="frame",
              colour="yellow") as pbar:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_render_and_save_chunk, task): i
                       for i, task in enumerate(tasks)}
            for future in as_completed(futures):
                _, saved = future.result()
                written_total += saved
                pbar.update(saved)

    return written_total
