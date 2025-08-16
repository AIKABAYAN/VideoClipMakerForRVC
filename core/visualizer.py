import math
import numpy as np
from typing import Tuple
from moviepy.editor import AudioFileClip, ImageSequenceClip
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patheffects as path_effects

class Band:
    def __init__(self, name: str, low_freq: float, high_freq: float, color: str):
        self.name = name
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.color = color  # hex string like "#RRGGBB"

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
        end_idx = start_idx + int(self.audio_fps / 30)  # ~1 frame window
        if end_idx > len(self.audio_array):
            end_idx = len(self.audio_array)

        window = self.audio_array[start_idx:end_idx]
        if window.ndim == 2:
            window = window.mean(axis=1)  # stereo → mono
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
    """Generate equally spaced bands from start_freq to end_freq with given step."""
    bands = []
    f_low = start_freq
    while f_low < end_freq:
        f_high = f_low + step
        # color gradient: red -> green -> blue
        t = (f_low - start_freq) / (end_freq - start_freq)
        if t < 0.5:
            r = int(255 * (1 - t * 2))
            g = int(255 * (t * 2))
            b = 32
        else:
            r = 32
            g = int(255 * (1 - (t - 0.5) * 2))
            b = int(255 * ((t - 0.5) * 2))
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        bands.append(Band(f"{int(f_low)}–{int(f_high)}", f_low, f_high, hex_color))
        f_low = f_high
    return tuple(bands)

def generate_visualizer_clip(mp3_path: str, fps: int = 30, resolution=(1280, 720),
                              opacity=0.5, scale_height=0.2):
    """
    Generate an audio visualizer clip with transparent background and neon glow effect.
    - opacity: 0.0 to 1.0 (transparency of bars)
    - scale_height: fraction of total video height for the visualizer (e.g., 0.2 = 20%)
    """
    bands = generate_bands(80, 3000, 20)  # Custom range and step
    analyzer = AudioAnalyzer(mp3_path, bands)
    audio_clip = AudioFileClip(mp3_path)
    duration = audio_clip.duration
    total_frames = int(math.ceil(duration * fps))

    fig_w, fig_h = resolution
    fig = Figure(figsize=(fig_w / 100, (fig_h * scale_height) / 100), dpi=100)
    ax = fig.add_subplot(111)

    # Fully transparent background
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, len(bands) - 0.5)
    ax.set_xticks([])
    ax.axis('off')

    # Draw bars with alpha for transparency
    bars = ax.bar(
        range(len(bands)),
        [0.0] * len(bands),
        color=[b.color for b in bands],
        alpha=opacity
    )

    # Add multi-layer soft neon glow
    for bar in bars:
        bar.set_path_effects([
            path_effects.withStroke(linewidth=12, foreground=bar.get_facecolor(), alpha=opacity * 0.15),
            path_effects.withStroke(linewidth=8, foreground=bar.get_facecolor(), alpha=opacity * 0.25),
            path_effects.withStroke(linewidth=6, foreground=bar.get_facecolor(), alpha=opacity * 0.4),
            path_effects.Normal()
        ])

    canvas = FigureCanvasAgg(fig)
    frames = []

    for i in range(total_frames):
        t = i / fps
        vals = analyzer.magnitudes_at_time(t)
        for bar, v in zip(bars, vals):
            bar.set_height(float(v))
        canvas.draw()

        # Keep alpha channel (RGBA)
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, [1, 2, 3, 0]]  # ARGB → RGBA

        frames.append(buf)

    clip = ImageSequenceClip(frames, fps=fps).set_opacity(opacity).set_duration(duration)
    return clip
