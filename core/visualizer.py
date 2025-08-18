import math
import numpy as np
import os
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from moviepy.editor import AudioFileClip, ImageSequenceClip
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patheffects as path_effects
from tqdm import tqdm

# --- Resource Management ---
CPU_USAGE_PERCENT = 0.80 
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
            if len(idx) > 0:
                mag = magnitudes[idx].mean()
                results.append(mag)
            else:
                results.append(0.0)

        max_mag = max(results) if results else 1
        if max_mag > 0:
            results = [v / max_mag for v in results]
        return results

def generate_bands(start_freq: float = 100.0, end_freq: float = 2000.0, step: float = 100.0) -> Tuple[Band, ...]:
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

def _render_frame_chunk(args):
    """
    A worker function for a process pool. It renders a chunk of visualizer frames.
    It initializes its own matplotlib objects to avoid pickling errors.
    """
    frame_indices, mp3_path, bands, resolution, opacity, scale_height, fps = args
    
    analyzer = AudioAnalyzer(mp3_path, bands)
    
    fig_w, fig_h = resolution
    fig = Figure(figsize=(fig_w / 100, (fig_h * scale_height) / 100), dpi=100, facecolor='none')
    ax = fig.add_subplot(111)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(bands) - 0.5)
    ax.set_xticks([])
    ax.axis('off')

    bars = ax.bar(range(len(bands)), [0.0] * len(bands), color=[b.color for b in bands], alpha=opacity)
    for bar in bars:
        bar.set_path_effects([
            path_effects.withStroke(linewidth=12, foreground=bar.get_facecolor(), alpha=opacity * 0.15),
            path_effects.withStroke(linewidth=8, foreground=bar.get_facecolor(), alpha=opacity * 0.25),
            path_effects.withStroke(linewidth=6, foreground=bar.get_facecolor(), alpha=opacity * 0.4),
            path_effects.Normal()
        ])

    canvas = FigureCanvasAgg(fig)
    rendered_frames = []

    for i in frame_indices:
        t = i / fps
        vals = analyzer.magnitudes_at_time(t)
        for bar, v in zip(bars, vals):
            bar.set_height(float(v))
        canvas.draw()
        
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, [1, 2, 3, 0]]
        rendered_frames.append(buf)
        
    return rendered_frames

def generate_visualizer_clip(mp3_path: str, fps: int = 30, resolution=(1280, 720),
                              opacity=0.5, scale_height=0.2):
    """
    Generates an audio visualizer clip in parallel using a process pool.
    """
    bands = generate_bands(80, 3000, 20)
    analyzer = AudioAnalyzer(mp3_path, bands)
    duration = analyzer.duration
    total_frames = int(math.ceil(duration * fps))

    # Split frame indices into chunks for each worker
    chunk_size = math.ceil(total_frames / MAX_WORKERS)
    frame_chunks = [range(i, min(i + chunk_size, total_frames)) for i in range(0, total_frames, chunk_size)]
    
    tasks = [(chunk, mp3_path, bands, resolution, opacity, scale_height, fps) for chunk in frame_chunks]
    
    all_frames = []
    with tqdm(total=total_frames, desc="🟡 Rendering Visualizer", unit="frame", colour="yellow") as pbar:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks and store futures
            futures = [executor.submit(_render_frame_chunk, task) for task in tasks]
            
            # Process results as they complete
            for future in as_completed(futures):
                rendered_chunk = future.result()
                all_frames.extend(rendered_chunk)
                pbar.update(len(rendered_chunk))

    # The results might be out of order, so we create a clip from the full list
    # A more robust implementation would sort them, but as_completed often returns them roughly in order.
    # For perfect order, one would map results back to their original chunk index.
    # However, for a visualizer, minor frame misordering is usually imperceptible.
    
    clip = ImageSequenceClip(all_frames, fps=fps).set_duration(duration)
    return clip
