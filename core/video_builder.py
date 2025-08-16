from PIL import Image, ImageFilter

# Pillow 10+ compatibility shim
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS
    
import os
import tempfile
import platform
import shutil
import subprocess
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
from os.path import basename, splitext
import numpy as np
from moviepy.editor import (
    ImageClip, ColorClip, CompositeVideoClip,
    AudioFileClip, TextClip, VideoFileClip
)
from moviepy.config import change_settings
from core.utils import list_images
from core.logger import get_logger
from tqdm import tqdm
from threading import Thread

from .animations import apply_image_animation
from .transitions import apply_random_transition
from core.visualizer import generate_visualizer_clip

logger = get_logger("sikabayan")

# Console colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def log_frame(stage, message, color=""):
    border = "═" * (len(stage) + len(message) + 5)
    tqdm.write(f"╔{border}╗")
    tqdm.write(f"║ {color}{stage}{RESET} | {message} ║")
    tqdm.write(f"╚{border}╝")

# -------------------------------
# FFmpeg & ImageMagick
# -------------------------------
def check_ffmpeg_encoders():
    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, check=True)
        encoders = result.stdout
        if "libx264" in encoders:
            return "libx264"
        if "h264_nvenc" in encoders:
            try:
                nvidia_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if nvidia_check.returncode == 0:
                    test_cmd = ["ffmpeg", "-f", "lavfi", "-i", "testsrc",
                                "-c:v", "h264_nvenc", "-preset", "fast", "-t", "1", "-f", "null", "-"]
                    subprocess.run(test_cmd, capture_output=True, check=True)
                    return "h264_nvenc"
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("NVENC detected but no working NVIDIA GPU found")
        if "mpeg4" in encoders:
            return "mpeg4"
        raise RuntimeError("No suitable video encoder found in FFmpeg")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg encoder check failed: {e.stderr}")
        raise RuntimeError("Could not check FFmpeg encoders")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found in PATH")
# -------------------------------
# Disclaimer generator
# -------------------------------
def save_disclaimer(mp3_path, artist_name, output_path):
    base_name = os.path.splitext(os.path.basename(mp3_path))[0]
    output_txt = os.path.splitext(output_path)[0] + ".txt"

    disclaimer = f"""🎵 {base_name} - {artist_name}  

⚠️ **Legal Disclaimer**:  
- This is a **fan-made AI cover** for **hobby/experimental purposes only**.  
- I do **not** claim ownership of the original music/voice. 
  All rights belong to the copyright holders.  
- **Not monetized** → This channel earns **no money** from these videos.  

🔧 **RVC Model Used**: Sharkoded  
🎶 **Original Song**: {base_name}  

📌 **About This Project**:  
- Created using **Retrieval-Based Voice Conversion (RVC)**, an AI voice-cloning tool.  
- This is a **non-professional hobbyist experiment**—not intended for commercial use.  

    Voice Synthesis: Retrieval-Based Voice Conversion (RVC) 

    Audio Production: Replay, LMMS Studio

    Video Production: open source https://github.com/AIKABAYAN/VideoClipMakerForRVC.git 

💬 **Suggest me what song want to RVC** Comment requests below!   

⚠️ COPYRIGHT & TAKEDOWN POLICY

This video is a transformative work created under the principles of Fair Use. 
It is a non-profit, hobbyist project and is not monetized. 
I do not claim ownership of the original musical composition or lyrics. 
All rights, credits, and ownership belong to the original artists, their labels, and publishers.

If you are a copyright holder and would like this content removed, 
please do not issue a copyright strike. 
Contact me directly at the email address below, 
and I will promptly and respectfully remove the video upon verification of your claim.

Contact for Takedown Requests: sharkoded@gmail.com

Thank you for watching and supporting this creative experiment! 
If you enjoyed it, please consider liking and subscribing for more.

#AICover #RVC #AImusic #VoiceCloning #AIvoice #MusicTech 
#{artist_name.replace(" ", "")} 
# #{base_name.replace(" ", "")} 
# #VoiceSynthesis #DeepfakeVoice #FYP
#RVC #AIcover #VoiceSynthesis #AIsinging #AIvoice #FYP #Viral #AImusic 
#DeepfakeVoice #Hobbyist #NoCopyrightIntended  
#sharkoded

👍 **Like & Subscribe** for more AI voice experiments!  
"""

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(disclaimer)

def configure_imagemagick():
    if platform.system() == "Windows":
        possible_paths = [
            r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.0.11-Q16-HDRI\magick.exe"
        ]
        for imagemagick_path in possible_paths:
            try:
                if os.path.exists(imagemagick_path):
                    change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
                    logger.info(f"ImageMagick binary configured at: {imagemagick_path}")
                    return True
            except Exception as e:
                logger.error(f"Error configuring ImageMagick: {e}")
        logger.warning("ImageMagick binary not found at standard paths.")
        return False
    return True

# -------------------------------
# Background processing
# -------------------------------
def create_blurred_background(img_path, resolution, blur_level, temp_dir):
    try:
        img = Image.open(img_path).convert("RGB")
        img_bg = img.resize(resolution, Image.LANCZOS)
        radius = max(1, min(10, blur_level))
        img_bg = img_bg.filter(ImageFilter.GaussianBlur(radius=radius))
        temp_path = os.path.join(temp_dir, f"blur_{os.path.basename(img_path)}.jpg")
        img_bg.save(temp_path, quality=95)
        return temp_path
    except Exception as e:
        logger.error(f"Failed to create blurred background: {e}")
        raise

def create_darkened_background(img_path, resolution, darkness_level, temp_dir):
    try:
        img = Image.open(img_path).convert("RGB")
        img_bg = img.resize(resolution, Image.LANCZOS)
        arr = np.array(img_bg)
        darkness = max(0, min(1, darkness_level/10))
        arr = (arr * darkness).astype(np.uint8)
        img_bg = Image.fromarray(arr)
        temp_path = os.path.join(temp_dir, f"dark_{os.path.basename(img_path)}.jpg")
        img_bg.save(temp_path, quality=95)
        return temp_path
    except Exception as e:
        logger.error(f"Failed to create darkened background: {e}")
        raise

def select_bitrate_for_resolution(resolution):
    w, h = resolution
    max_dim = max(w, h)
    if max_dim >= 3840:
        return "50000k"
    if max_dim >= 2560:
        return "20000k"
    return "8000k"

# -------------------------------
# Chunk rendering
# -------------------------------
def render_chunk(chunk_imgs, duration_per_image, bg_mode, blur_level, resolution, temp_dir, idx, animations=None):
    start_time = time.time()
    log_frame("🟢 Render", f"PID {os.getpid()} | Chunk {idx} | {len(chunk_imgs)} images", GREEN)

    base_clips = []
    for img in chunk_imgs:
        if img == "__BLACK__":
            main_clip = ColorClip(size=resolution, color=(0, 0, 0)).set_duration(duration_per_image)
        else:
            main_clip = ImageClip(img).set_duration(duration_per_image)

        animated_clip = apply_image_animation(main_clip, duration_per_image, resolution, animations)

        if bg_mode == "Blur" and img != "__BLACK__":
            blur_path = create_blurred_background(img, resolution, blur_level, temp_dir)
            bg_clip = ImageClip(blur_path).set_duration(duration_per_image)
            frame_clip = CompositeVideoClip([bg_clip, animated_clip.set_position("center")], size=resolution)
        elif bg_mode == "Darken" and img != "__BLACK__":
            dark_path = create_darkened_background(img, resolution, blur_level, temp_dir)
            bg_clip = ImageClip(dark_path).set_duration(duration_per_image)
            frame_clip = CompositeVideoClip([bg_clip, animated_clip.set_position("center")], size=resolution)
        else:
            frame_clip = CompositeVideoClip(
                [animated_clip.on_color(size=resolution, color=(0, 0, 0), pos=("center", "center"))],
                size=resolution
            )
        frame_clip = frame_clip.set_duration(duration_per_image)
        base_clips.append(frame_clip)

    if not base_clips:
        raise RuntimeError("No images for this chunk")

    timeline = []
    t_cursor = 0.0
    first = base_clips[0].set_start(t_cursor)
    timeline.append(first)
    t_cursor += duration_per_image

    for i in range(1, len(base_clips)):
        prev = timeline[-1]
        base_next = base_clips[i]
        prev_timed, next_timed, td = apply_random_transition(prev, base_next, duration_per_image, resolution)
        timeline[-1] = prev_timed
        timeline.append(next_timed)
        t_cursor += duration_per_image - td

    final_chunk = CompositeVideoClip(timeline, size=resolution)
    final_chunk = final_chunk.set_duration(t_cursor)

    part_path = os.path.join(temp_dir, f"part_{idx}.mp4")
    final_chunk.write_videofile(
        part_path, fps=30, codec="libx264", audio=False,
        preset="ultrafast", bitrate=select_bitrate_for_resolution(resolution),
        threads=1, verbose=False, logger=None
    )
    final_chunk.close()
    for c in base_clips:
        try:
            c.close()
        except Exception:
            pass

    elapsed = time.time() - start_time
    log_frame("✅ Render Done", f"Chunk {idx} → {os.path.basename(part_path)} ({elapsed:.1f}s)", GREEN)
    return part_path

# -------------------------------
# Merge worker
# -------------------------------
def merge_worker(queue, concat_list_path, merged_path, total_chunks):
    merged_count = 0
    with open(concat_list_path, "w", encoding="utf-8") as f:
        while True:
            part_path = queue.get()
            if part_path is None:
                break
            merged_count += 1
            f.write(f"file '{part_path}'\n")
            f.flush()
            log_frame("🔵 Merge", f"Added {os.path.basename(part_path)} ({merged_count}/{total_chunks})", BLUE)

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", merged_path],
        check=True
    )
    log_frame("✅ Merge Done", f"Merged {total_chunks} chunks → {os.path.basename(merged_path)}", BLUE)

# -------------------------------
# Custom frame logger (no MoviePy logger)
# -------------------------------
class FrameLogger:
    def __init__(self, total_frames=None, callback=None):
        self.total_frames = total_frames or 1
        self.callback = callback

    def log_message(self, msg: str):
        print(msg)

    def frame_callback(self, current_frame: int):
        if self.callback and self.total_frames:
            percent = int(current_frame / max(1, self.total_frames) * 100)
            self.callback(percent)

# -------------------------------
# Build video
# -------------------------------
def build_video_multithread(mp3_path, bg_folder, cover_path, song_name, artist_name,
                           output_path, resolution=(1280, 720), num_threads=None,
                           bg_mode="Blur", blur_level=3, progress_callback=None,
                           animations=None, include_visualizer=False,
                           visualizer_height=0.4):

    try:
        encoder = check_ffmpeg_encoders()
        logger.info(f"Using video encoder: {encoder}")
    except Exception as e:
        logger.error(f"Encoder initialization failed: {str(e)}")
        if progress_callback:
            progress_callback(0)
        return

    if not configure_imagemagick():
        logger.warning("Proceeding without ImageMagick - text rendering may not work")

    # Background images
    if not bg_folder or not os.path.isdir(bg_folder):
        images = ["__BLACK__"]
    else:
        try:
            images = list_images(bg_folder)
            if not images:
                images = ["__BLACK__"]
        except Exception:
            images = ["__BLACK__"]

    total_duration = AudioFileClip(mp3_path).duration
    duration_per_image = total_duration / len(images)

    song_text = f"{splitext(basename(mp3_path))[0]} - {artist_name}"
    temp_dir = tempfile.mkdtemp(prefix="sikabayan_")
    merged_video_path = os.path.join(temp_dir, "merged.mp4")
    concat_list_path = os.path.join(temp_dir, "concat.txt")
    queue = Queue()

    try:
        total_cores = os.cpu_count()
        chunk_size = max(1, math.ceil(len(images) / total_cores))
        chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]

        # Merge process
        merger_proc = Process(target=merge_worker, args=(queue, concat_list_path, merged_video_path, len(chunks)))
        merger_proc.start()

        # Multi-process rendering
        with ProcessPoolExecutor(max_workers=total_cores) as executor:
            futures = {executor.submit(
                render_chunk, chunk, duration_per_image, bg_mode,
                blur_level, resolution, temp_dir, idx, animations
            ): idx for idx, chunk in enumerate(chunks)}

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Chunks", unit="chunk")):
                part_path = future.result()
                queue.put(part_path)
                if progress_callback:
                    progress_callback((i + 1) * 40 // len(futures))

        queue.put(None)
        merger_proc.join()

        # -------------------------------
        # Overlay audio and visualizer
        # -------------------------------
        log_frame("🟡 Overlay", "Starting final video creation...", YELLOW)
        audio_clip = AudioFileClip(mp3_path)
        base_video = VideoFileClip(merged_video_path).set_audio(audio_clip)

        if include_visualizer:
            vis_clip = generate_visualizer_clip(
                mp3_path,
                fps=30,
                resolution=resolution,
                opacity=0.5,
                scale_height=visualizer_height
            )
            video_h = resolution[1]
            vis_height_px = video_h * visualizer_height
            leftover_space = video_h - vis_height_px
            offset_px = int(max(0, leftover_space * 0.15))
            vis_y_pos = video_h - vis_height_px - offset_px
            base_video = CompositeVideoClip(
                [base_video, vis_clip.set_position(("center", vis_y_pos))],
                size=resolution
            )

        # Text overlay
        try:
            text_clip = TextClip(song_text, fontsize=40, color="white", font="Arial-Bold")
            text_width, text_height = text_clip.w, text_clip.h
            scrolling_text = text_clip.set_position(
                lambda t: (resolution[0] - (t * 100) % (text_width + resolution[0]),
                           resolution[1] - text_height - 20)
            ).set_duration(total_duration)
            final_video = CompositeVideoClip([base_video, scrolling_text], size=resolution)
        except Exception as e:
            logger.warning(f"Text overlay failed: {e}")
            final_video = base_video

        # -------------------------------
        # Per-frame progress
        # -------------------------------
        frame_logger = FrameLogger(total_frames=int(final_video.duration*final_video.fps),
                                   callback=progress_callback)

        def monitor_video_progress(clip, logger: FrameLogger):
            total_frames = int(clip.duration * clip.fps)
            for i in range(total_frames):
                time.sleep(1 / clip.fps)
                logger.frame_callback(i)

        monitor_thread = Thread(target=monitor_video_progress, args=(final_video, frame_logger))
        monitor_thread.start()

        # Write final video
        final_video.write_videofile(
            output_path,
            fps=30,
            codec=encoder,
            preset="ultrafast",
            bitrate=select_bitrate_for_resolution(resolution),
            threads=total_cores,
            verbose=False,
            logger=None  # Disable MoviePy logger
        )
        monitor_thread.join()

        final_video.close()
        audio_clip.close()
        log_frame("✅ Done", f"Final video saved as {output_path}", GREEN)
        save_disclaimer(mp3_path, artist_name, output_path)
        logger.info(f"Disclaimer saved at {os.path.splitext(output_path)[0]}.txt")
        if progress_callback:
            progress_callback(100)

    except Exception as e:
        logger.exception(f"Error during build_video_multithread: {e}")
        if progress_callback:
            progress_callback(0)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
