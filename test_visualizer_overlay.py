from moviepy.editor import ColorClip, CompositeVideoClip, AudioFileClip
from core.visualizer import generate_visualizer_clip

# ==== CONFIG ====

mp3_path = r"C:/Users/SIKABAYAN/Desktop/mini hati hati.MP3"   # change to your MP3
resolution = (1280, 720)
opacity = 0.5
scale_height = 0.2  # 20% of video height
vis_bottom_offset_ratio = 0.3  # bottom 30% start
output_path = r"C:/Users/SIKABAYAN/Desktop/result/test.mp4"
# ===============

# Create a plain background just for testing
background = ColorClip(size=resolution, color=(30, 30, 30), duration=AudioFileClip(mp3_path).duration)

# Generate visualizer clip
vis_clip = generate_visualizer_clip(mp3_path, fps=30, resolution=resolution,
                                    opacity=opacity, scale_height=scale_height)

# Calculate Y position for bottom 30% start
vis_y_pos = resolution[1] * (1 - vis_bottom_offset_ratio)

# Composite: visualizer in front of background
final_clip = CompositeVideoClip(
    [
        background,
        vis_clip.set_position(("center", vis_y_pos))
    ],
    size=resolution
).set_audio(AudioFileClip(mp3_path))

# Write video
final_clip.write_videofile(output_path, fps=30, codec="libx264", preset="ultrafast")

print(f"Test visualizer video saved to {output_path}")
