from PIL import Image, ImageFilter

# Pillow 10+ compatibility shim
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

import os
import random
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
    AudioFileClip, TextClip, VideoFileClip, concatenate_videoclips
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
BLUE  = "\033[94m"
YELLOW= "\033[93m"
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
    if max_dim >= 3840:  # 4K+
        return "50000k"
    if max_dim >= 2560:  # QHD+
        return "20000k"
    if max_dim >= 1920:  # 1080p-Vertical (Shorts) or 1080p Horizontal
        return "12000k"
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
# Custom frame logger (manual progress)
# -------------------------------
class FrameLogger:
    def __init__(self, total_frames=None, callback=None, tag="finalize"):
        self.total_frames = max(1, int(total_frames or 1))
        self.callback = callback
        self.tag = tag

    def log_message(self, msg: str):
        print(msg)

    def frame_callback(self, current_frame: int):
        if self.callback:
            percent = int(current_frame / self.total_frames * 100)
            self.callback(percent, self.tag)

def _monitor_video_progress(clip, logger: FrameLogger):
    # Fire a coarse-grained progress based on wall-time sleep
    total_frames = int(clip.duration * clip.fps)
    total_frames = max(1, total_frames)
    for i in range(total_frames):
        time.sleep(1 / max(1, clip.fps))
        logger.frame_callback(i + 1)

# -------------------------------
# Shared pipeline (one orientation)
# -------------------------------
def _build_single_output(mp3_path, images, song_name, artist_name, output_path,
                         encoder, resolution, bg_mode, blur_level,
                         animations, include_visualizer, visualizer_height,
                         progress_callback=None, tag_prefix="youtube"):
    """
    Membangun satu output video (YouTube atau Shorts) dengan pipeline yang sama.
    tag_prefix: "youtube" atau "shorts" → untuk label progress.
    """
    # Durasi per gambar
    total_duration = AudioFileClip(mp3_path).duration
    duration_per_image = total_duration / max(1, len(images))

    song_text = f"{splitext(basename(mp3_path))[0]} - {artist_name}"
    temp_dir = tempfile.mkdtemp(prefix=f"sikabayan_{tag_prefix}_")
    merged_video_path = os.path.join(temp_dir, "merged.mp4")
    concat_list_path = os.path.join(temp_dir, "concat.txt")
    queue = Queue()

    try:
        total_cores = os.cpu_count() or 4
        chunk_size = max(1, math.ceil(len(images) / total_cores))
        chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]

        # Merge process
        merger_proc = Process(target=merge_worker, args=(queue, concat_list_path, merged_video_path, len(chunks)))
        merger_proc.start()

        # Multi-process rendering
        with ProcessPoolExecutor(max_workers=total_cores) as executor:
            futures = {
                executor.submit(
                    render_chunk, chunk, duration_per_image, bg_mode,
                    blur_level, resolution, temp_dir, idx, animations
                ): idx for idx, chunk in enumerate(chunks)
            }

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures),
                                            desc=f"Chunks-{tag_prefix}", unit="chunk")):
                part_path = future.result()
                queue.put(part_path)
                if progress_callback:
                    progress_callback((i + 1) * 40 // max(1, len(futures)), f"render_{tag_prefix}")

        queue.put(None)
        merger_proc.join()

        if progress_callback:
            progress_callback(60, f"merge_{tag_prefix}")

        # -------------------------------
        # Overlay audio and visualizer
        # -------------------------------
        log_frame("🟡 Overlay", f"Starting final video creation... ({tag_prefix})", YELLOW)
        if progress_callback:
            progress_callback(70, f"overlay_{tag_prefix}")

        audio_clip = AudioFileClip(mp3_path)
        base_video = VideoFileClip(merged_video_path).set_audio(audio_clip)

        # Visualizer (posisi menempel bawah, tetap proporsional utk horizontal/vertical)
        if include_visualizer:
            vis_clip = generate_visualizer_clip(
                mp3_path, fps=30, resolution=resolution,
                opacity=0.5, scale_height=visualizer_height
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

        # Text overlay (scrolling at bottom)
        try:
            text_clip = TextClip(song_text, fontsize=40, color="white", font="Arial-Bold")
            text_width, text_height = text_clip.w, text_clip.h
            scrolling_text = text_clip.set_position(
                lambda t: (resolution[0] - (t * 100) % (text_width + resolution[0]),
                           resolution[1] - text_height - 20)
            ).set_duration(total_duration)
            main_video = CompositeVideoClip([base_video, scrolling_text], size=resolution)
        except Exception as e:
            logger.warning(f"Text overlay failed ({tag_prefix}): {e}")
            main_video = base_video

        # Intro Screen — ambil gambar pertama jika ada
        base_name = os.path.splitext(os.path.basename(mp3_path))[0]
        if images and images[0] != "__BLACK__":
            bg_choice = images[0]
            intro_bg = ImageClip(bg_choice).set_duration(2).resize(height=resolution[1])
            intro_bg = intro_bg.on_color(size=resolution, color=(0, 0, 0), pos=("center", "center"))
        else:
            intro_bg = ColorClip(size=resolution, color=(0, 0, 0)).set_duration(2)

        title_clip = TextClip(base_name, fontsize=50, color="white", font="Arial-Bold") \
            .set_position(("center", resolution[1] * 0.4)) \
            .set_duration(2)

        artist_clip = TextClip(f"by {artist_name}", fontsize=35, color="white", font="Arial-Italic") \
            .set_position(("center", resolution[1] * 0.55)) \
            .set_duration(2)

        intro = CompositeVideoClip([intro_bg, title_clip, artist_clip], size=resolution)
        final_video = concatenate_videoclips([intro, main_video])

        # Save video
        log_frame("💾 Save", f"Writing final video → {output_path}", YELLOW)
        if progress_callback:
            progress_callback(90, f"finalize_{tag_prefix}")

        # Manual monitor progress (MoviePy tidak punya 'callbacks' arg)
        total_frames = int(final_video.duration * 30)
        frame_logger = FrameLogger(total_frames=total_frames,
                                   callback=lambda p, tag: progress_callback(p, tag) if progress_callback else None,
                                   tag=f"finalize_{tag_prefix}")
        monitor_thread = Thread(target=_monitor_video_progress, args=(final_video, frame_logger))
        monitor_thread.daemon = True
        monitor_thread.start()

        final_video.write_videofile(
            output_path,
            fps=30,
            codec=encoder,
            preset="medium",
            bitrate=select_bitrate_for_resolution(resolution),
            audio_codec="aac",
            threads=os.cpu_count() or 4,
            verbose=False,
            logger=None
        )
        monitor_thread.join()

        final_video.close()
        audio_clip.close()

        save_disclaimer(mp3_path, artist_name, output_path)
        log_frame("✅ Done", f"Video created successfully: {output_path}", GREEN)
        if progress_callback:
            progress_callback(100, f"finalize_{tag_prefix}")

    except Exception as e:
        logger.error(f"Error during single output build ({tag_prefix}): {e}", exc_info=True)
        if progress_callback:
            progress_callback(0, f"error_{tag_prefix}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# -------------------------------
# Build video (DICT VERSION) — YouTube & Shorts
# -------------------------------
def build_video_multithread(settings, progress_callback=None):
    """
    Build video menggunakan settings dict:
    Wajib:
      - mp3_file, song_name, artist_name, output_folder
    Opsional:
      - bg_folder, cover_image (tidak dipakai di pipeline), resolution (untuk YouTube)
      - export_youtube (bool, default True)
      - export_shorts  (bool, default False)
      - bg_mode: "Blur"|"Darken"|else (default "Blur")
      - blur_level: int (default 3)
      - animations: dict/None
      - include_visualizer: bool (default False)
      - visualizer_height: "40%" atau float [0..1] (default "40%")
      - shorts_resolution: tuple (default (1080, 1920))
    Progress stages:
      - setup, render_youtube, merge_youtube, overlay_youtube, finalize_youtube
      - render_shorts, merge_shorts, overlay_shorts, finalize_shorts
      - error_youtube / error_shorts
    """
    # --- unpack dict ---
    mp3_path       = settings["mp3_file"]
    bg_folder      = settings.get("bg_folder")
    # cover_path   = settings.get("cover_image")  # tidak dipakai saat ini
    song_name      = settings["song_name"]
    artist_name    = settings["artist_name"]
    output_folder  = settings["output_folder"]

    # Flags export
    export_youtube = settings.get("export_youtube", True)
    export_shorts  = settings.get("export_shorts", False)

    # Resolusi
    yt_resolution      = tuple(settings.get("resolution", (1280, 720)))
    shorts_resolution  = tuple(settings.get("shorts_resolution", (1080, 1920)))

    bg_mode            = settings.get("bg_mode", "Blur")
    blur_level         = settings.get("blur_level", 3)
    animations         = settings.get("animations", None)
    include_visualizer = settings.get("include_visualizer", False)
    vis_h_raw          = settings.get("visualizer_height", "40%")
    visualizer_height  = float(str(vis_h_raw).strip("%")) / 100.0

    if progress_callback:
        progress_callback(0, "setup")

    # Encoder & ImageMagick
    try:
        encoder = check_ffmpeg_encoders()
        logger.info(f"Using video encoder: {encoder}")
    except Exception as e:
        logger.error(f"Encoder initialization failed: {str(e)}")
        if progress_callback:
            progress_callback(0, "error_setup")
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

    # Pastikan folder output ada
    os.makedirs(output_folder, exist_ok=True)

    # Output paths
    yt_output_path     = os.path.join(output_folder, f"{song_name}.mp4")
    shorts_output_path = os.path.join(output_folder, f"{song_name}_Shorts.mp4")

    try:
        # YouTube
        if export_youtube:
            _build_single_output(
                mp3_path=mp3_path,
                images=images,
                song_name=song_name,
                artist_name=artist_name,
                output_path=yt_output_path,
                encoder=encoder,
                resolution=yt_resolution,
                bg_mode=bg_mode,
                blur_level=blur_level,
                animations=animations,
                include_visualizer=include_visualizer,
                visualizer_height=visualizer_height,
                progress_callback=progress_callback,
                tag_prefix="youtube"
            )

        # Shorts (9:16)
        if export_shorts:
            _build_single_output(
                mp3_path=mp3_path,
                images=images,
                song_name=song_name,
                artist_name=artist_name,
                output_path=shorts_output_path,
                encoder=encoder,
                resolution=shorts_resolution,
                bg_mode=bg_mode,
                blur_level=blur_level,
                animations=animations,
                include_visualizer=include_visualizer,
                visualizer_height=visualizer_height,
                progress_callback=progress_callback,
                tag_prefix="shorts"
            )

        # Selesai semua
        if progress_callback:
            progress_callback(100, "done")

    except Exception as e:
        logger.error(f"Error during build_video_multithread: {e}", exc_info=True)
        if progress_callback:
            progress_callback(0, "error")
