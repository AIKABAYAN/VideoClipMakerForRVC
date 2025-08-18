# video_builder.py
from PIL import Image, ImageFilter
# Pillow 10+ compatibility shim
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

import os
import io
import tempfile
import platform
import shutil
import subprocess
import math
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue
from os.path import basename, splitext
from typing import List, Tuple, Dict, Any

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

# These modules are project-specific; keep them as-is
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
# FFmpeg & ImageMagick / helpers
# -------------------------------
def _run(cmd: List[str], input_text: str = None):
    return subprocess.run(
        cmd,
        input=input_text.encode("utf-8") if input_text is not None else None,
        capture_output=True,
        text=False,
        check=True
    )

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except Exception:
        raise RuntimeError("FFmpeg not found in PATH")

def detect_best_encoder() -> Tuple[str, Dict[str, str]]:
    """
    Detect the fastest available hardware encoder.
    Returns (encoder_name, ffmpeg_extra_args_dict)
    """
    check_ffmpeg()
    try:
        enc_list = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True).stdout
    except Exception as e:
        raise RuntimeError(f"Failed to query FFmpeg encoders: {e}")

    # NVIDIA
    if "h264_nvenc" in enc_list:
        try:
            n = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if n.returncode == 0:
                return "h264_nvenc", {"preset": "p1", "rc": "vbr"}
        except Exception:
            pass

    # Intel QuickSync
    if "h264_qsv" in enc_list:
        return "h264_qsv", {"preset": "veryfast"}

    # AMD AMF
    if "h264_amf" in enc_list:
        return "h264_amf", {"quality": "speed"}

    # CPU fallback
    if "libx264" in enc_list:
        return "libx264", {"preset": "ultrafast", "tune": "zerolatency"}

    if "mpeg4" in enc_list:
        return "mpeg4", {}

    raise RuntimeError("No suitable H.264 encoder found (nvenc/qsv/amf/libx264/mpeg4).")

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
    try:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(disclaimer)
    except Exception as e:
        logger.warning(f"Could not write disclaimer file: {e}")

# -------------------------------
# Speed helpers
# -------------------------------
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

def _preprocess_image(img_path: str, resolution: Tuple[int, int], bg_mode: str, level: int):
    """
    Load once, resize to resolution, and create background array per mode.
    Returns dict with:
      - 'fg_path' (original path)
      - 'bg_arr' (numpy array) or None
    """
    try:
        img = Image.open(img_path).convert("RGB")
        bg = img.resize(resolution, Image.LANCZOS)
        if bg_mode == "Blur":
            radius = max(1, min(10, level))
            bg = bg.filter(ImageFilter.GaussianBlur(radius=radius))
        elif bg_mode == "Darken":
            factor = max(0, min(1, level / 10))
            arr = (np.array(bg, dtype=np.float32) * factor).astype(np.uint8)
            bg = Image.fromarray(arr)
        return {"fg_path": img_path, "bg_arr": np.array(bg)}
    except Exception as e:
        logger.warning(f"Preprocess failed for {img_path}: {e}")
        return {"fg_path": img_path, "bg_arr": None}

# -------------------------------
# NORMAL MODE (optimized original flow)
# -------------------------------
def render_chunk(chunk_imgs, duration_per_image, bg_mode, blur_level, resolution, temp_dir, idx, animations=None, encoder="libx264"):
    start_time = time.time()
    log_frame("🟢 Render", f"PID {os.getpid()} | Chunk {idx} | {len(chunk_imgs)} images", GREEN)

    base_clips = []
    for img in chunk_imgs:
        if img == "__BLACK__":
            main_clip = ColorClip(size=resolution, color=(0, 0, 0)).set_duration(duration_per_image)
            animated_clip = apply_image_animation(main_clip, duration_per_image, resolution, animations)
            frame_clip = CompositeVideoClip(
                [animated_clip.on_color(size=resolution, color=(0, 0, 0), pos=("center", "center"))],
                size=resolution
            )
        else:
            main_clip = ImageClip(img).set_duration(duration_per_image)
            animated_clip = apply_image_animation(main_clip, duration_per_image, resolution, animations)
            img_prep = _preprocess_image(img, resolution, bg_mode, blur_level)
            if img_prep["bg_arr"] is not None:
                bg_clip = ImageClip(img_prep["bg_arr"]).set_duration(duration_per_image)
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
        part_path,
        fps=30,
        codec=encoder,
        audio=False,
        preset="ultrafast" if encoder != "h264_nvenc" else "fast",
        bitrate=select_bitrate_for_resolution(resolution),
        threads=os.cpu_count() or 4,
        verbose=False,
        logger=None
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
    total_frames = int(clip.duration * clip.fps)
    total_frames = max(1, total_frames)
    for i in range(total_frames):
        time.sleep(1 / max(1, clip.fps))
        logger.frame_callback(i + 1)

# -------------------------------
# HYBRID TURBO (replacement for _build_single_output)
# -------------------------------
def _build_single_output(mp3_path, images, song_name, artist_name, output_path,
                         encoder, resolution, bg_mode, blur_level,
                         animations, include_visualizer, visualizer_height,
                         progress_callback=None, tag_prefix="youtube"):
    """
    HYBRID TURBO:
    - Preprocess images into base frames (BG + centered FG) in parallel.
    - Write static frames per image.
    - For each boundary, create transition frames via numpy linear blending (fast).
    - Render visualizer once (if requested), then overlay in a single FFmpeg encode (GPU if available).
    - Keep function signature the same so callers don't change.
    """
    import math
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor

    fps = 30
    transition_ratio = 0.25     # fraction of image duration used for transition
    min_transition = 0.25       # seconds
    max_transition = 1.0        # seconds
    png_compress_level = 0      # fastest
    frames_io_optimize = False  # set true if you prefer memory buffering (not used here)

    # ---------- timing ----------
    audio_clip = AudioFileClip(mp3_path)
    total_duration = audio_clip.duration
    audio_clip.close()

    if not images:
        images = ["__BLACK__"]

    n_images = max(1, len(images))
    duration_per_image = max(0.001, total_duration / n_images)
    trans_sec = max(min_transition, min(max_transition, duration_per_image * transition_ratio))
    trans_frames = max(1, int(round(trans_sec * fps)))
    frames_per_image = max(trans_frames + 1, int(round(duration_per_image * fps)))

    if progress_callback:
        progress_callback(2, f"setup_{tag_prefix}")

    # ---------- preprocess all images (parallel) ----------
    log_frame("🟢 Preprocess", f"{len(images)} images → RAM composites", GREEN)

    def _compose_base_frame_local(img_path: str) -> np.ndarray:
        W, H = resolution
        if img_path == "__BLACK__":
            return np.zeros((H, W, 3), dtype=np.uint8)
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Cannot open {img_path}: {e}")
            return np.zeros((H, W, 3), dtype=np.uint8)

        # background
        bg = im.resize((W, H), Image.LANCZOS)
        if bg_mode == "Blur":
            radius = max(1, min(10, blur_level))
            bg = bg.filter(ImageFilter.GaussianBlur(radius=radius))
        elif bg_mode == "Darken":
            factor = max(0, min(1, blur_level / 10))
            arr = (np.array(bg, dtype=np.float32) * factor).astype(np.uint8)
            bg = Image.fromarray(arr)

        # foreground scaled to fit by height then width if needed
        fw, fh = im.size
        scale = H / fh
        new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
        if new_w > W:
            scale = W / fw
            new_w, new_h = int(round(fw * scale)), int(round(fh * scale))
        fg = im.resize((new_w, new_h), Image.LANCZOS)

        canvas = bg.copy()
        x = (W - new_w) // 2
        y = (H - new_h) // 2
        canvas.paste(fg, (x, y))
        return np.array(canvas)

    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4)) as pool:
        base_frames = list(tqdm(
            pool.map(_compose_base_frame_local, images),
            total=len(images), desc=f"Compose-{tag_prefix}", unit="img"
        ))

    # ---------- frame emission (static + blended transitions) ----------
    temp_dir = tempfile.mkdtemp(prefix=f"sikabayan_hybrid_{tag_prefix}_")
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    H, W = resolution[1], resolution[0]
    frame_index = 0
    total_blocks = len(base_frames)

    log_frame("🟢 Frames", f"Generating frames @ {fps}fps with transitions", GREEN)
    for i in range(total_blocks):
        base = base_frames[i]

        static_count = frames_per_image if i == total_blocks - 1 else max(1, frames_per_image - trans_frames)
        for _ in range(static_count):
            out_path = os.path.join(frames_dir, f"frame_{frame_index:08d}.png")
            Image.fromarray(base).save(out_path, optimize=False, compress_level=png_compress_level)
            frame_index += 1

        if i < total_blocks - 1:
            nxt = base_frames[i + 1].astype(np.float32)
            cur = base.astype(np.float32)
            denom = float(trans_frames)
            for k in range(trans_frames):
                a = (k + 1) / denom
                blend = (1.0 - a) * cur + a * nxt
                out = blend.clip(0, 255).astype(np.uint8)
                out_path = os.path.join(frames_dir, f"frame_{frame_index:08d}.png")
                Image.fromarray(out).save(out_path, optimize=False, compress_level=png_compress_level)
                frame_index += 1

        if progress_callback:
            done = 2 + int(70 * (i + 1) / max(1, total_blocks))
            progress_callback(min(70, done), f"render_{tag_prefix}")

    # ---------- optional visualizer (pre-render once) ----------
    vis_path = None
    vis_y_pos = 0
    if include_visualizer:
        try:
            log_frame("🟡 Visualizer", "Rendering waveform clip…", YELLOW)
            vis_clip = generate_visualizer_clip(
                mp3_path, fps=fps, resolution=resolution,
                opacity=0.5, scale_height=visualizer_height
            )
            vis_path = os.path.join(temp_dir, "visualizer.mp4")
            try:
                enc_name_vis, enc_opts_vis = detect_best_encoder()
                if enc_name_vis == "h264_nvenc":
                    vis_clip.write_videofile(vis_path, fps=fps, codec="h264_nvenc",
                                             preset="p1", bitrate=select_bitrate_for_resolution(resolution),
                                             audio=False, verbose=False, logger=None)
                elif enc_name_vis == "h264_qsv":
                    vis_clip.write_videofile(vis_path, fps=fps, codec="h264_qsv",
                                             preset="veryfast", bitrate=select_bitrate_for_resolution(resolution),
                                             audio=False, verbose=False, logger=None)
                elif enc_name_vis == "h264_amf":
                    vis_clip.write_videofile(vis_path, fps=fps, codec="h264_amf",
                                             bitrate=select_bitrate_for_resolution(resolution),
                                             audio=False, verbose=False, logger=None)
                else:
                    vis_clip.write_videofile(vis_path, fps=fps, codec="libx264",
                                             preset="ultrafast", bitrate=select_bitrate_for_resolution(resolution),
                                             audio=False, verbose=False, logger=None)
            finally:
                try:
                    vis_clip.close()
                except Exception:
                    pass

            video_h = resolution[1]
            vis_height_px = int(video_h * visualizer_height)
            leftover_space = video_h - vis_height_px
            offset_px = int(max(0, leftover_space * 0.15))
            vis_y_pos = video_h - vis_height_px - offset_px
        except Exception as e:
            log_frame("⚠️ Visualizer", f"Skip (error: {e})", YELLOW)
            vis_path = None

    if progress_callback:
        progress_callback(85, f"encode_{tag_prefix}")

    # ---------- one-shot FFmpeg encode (GPU if available) ----------
    try:
        enc_name, enc_opts = detect_best_encoder()
    except Exception:
        enc_name, enc_opts = ("libx264", {"preset": "ultrafast", "tune": "zerolatency"})

    bitrate = select_bitrate_for_resolution(resolution)
    frames_pattern = os.path.join(frames_dir, "frame_%08d.png")

    cmd = ["ffmpeg", "-y",
           "-framerate", str(fps), "-i", frames_pattern,
           "-i", mp3_path]

    filter_complex = None
    map_video = "0:v"
    if include_visualizer and vis_path and os.path.exists(vis_path):
        cmd += ["-i", vis_path]
        filter_complex = f"[0:v][2:v]overlay=x=(W-w)/2:y={vis_y_pos}:shortest=1[vout]"
        map_video = "[vout]"

    if filter_complex:
        cmd += ["-filter_complex", filter_complex, "-map", map_video, "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]

    cmd += ["-pix_fmt", "yuv420p", "-shortest", "-c:a", "aac", "-b:a", "192k"]

    cmd += ["-c:v", enc_name]
    if enc_name == "h264_nvenc":
        cmd += ["-preset", enc_opts.get("preset", "p1"),
                "-rc:v", enc_opts.get("rc", "vbr"),
                "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", str(int(int(bitrate[:-1]) * 2)) + "k"]
    elif enc_name == "h264_qsv":
        cmd += ["-preset", enc_opts.get("preset", "veryfast"),
                "-b:v", bitrate]
    elif enc_name == "h264_amf":
        cmd += ["-quality", enc_opts.get("quality", "speed"),
                "-b:v", bitrate]
    else:
        if "preset" in enc_opts:
            cmd += ["-preset", enc_opts["preset"]]
        if "tune" in enc_opts:
            cmd += ["-tune", enc_opts["tune"]]
        cmd += ["-b:v", bitrate]

    cmd += [output_path]

    log_frame("🟡 Encode", f"FFmpeg → {enc_name} | {bitrate}", YELLOW)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg encode failed: {e.stderr.decode('utf-8', errors='ignore')}")
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    save_disclaimer(mp3_path, artist_name, output_path)
    log_frame("✅ Done (Hybrid Turbo)", f"Video created successfully: {output_path}", GREEN)
    if progress_callback:
        progress_callback(100, f"finalize_{tag_prefix}")

# -------------------------------
# PUBLIC ENTRY — DICT SETTINGS
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
      - fast_mode: bool (if True → TURBO hybrid)
    """
    mp3_path       = settings["mp3_file"]
    bg_folder      = settings.get("bg_folder")
    song_name      = settings["song_name"]
    artist_name    = settings["artist_name"]
    output_folder  = settings["output_folder"]

    export_youtube = settings.get("export_youtube", True)
    export_shorts  = settings.get("export_shorts", False)

    yt_resolution      = tuple(settings.get("resolution", (1280, 720)))
    shorts_resolution  = tuple(settings.get("shorts_resolution", (1080, 1920)))

    bg_mode            = settings.get("bg_mode", "Blur")
    blur_level         = settings.get("blur_level", 3)
    animations         = settings.get("animations", None)
    include_visualizer = settings.get("include_visualizer", False)
    vis_h_raw          = settings.get("visualizer_height", "40%")
    visualizer_height  = float(str(vis_h_raw).strip("%")) / 100.0

    fast_mode          = settings.get("fast_mode", False)

    if progress_callback:
        progress_callback(0, "setup")

    encoder_name = None
    if not fast_mode:
        try:
            encoder_name, _ = detect_best_encoder()
            logger.info(f"Using video encoder: {encoder_name}")
        except Exception as e:
            logger.error(f"Encoder initialization failed: {str(e)}")
            if progress_callback:
                progress_callback(0, "error_setup")
            return

    if not configure_imagemagick():
        logger.warning("Proceeding without ImageMagick - text rendering may not work")

    # Background images list
    if not bg_folder or not os.path.isdir(bg_folder):
        images = ["__BLACK__"]
    else:
        try:
            images = list_images(bg_folder)
            if not images:
                images = ["__BLACK__"]
        except Exception:
            images = ["__BLACK__"]

    os.makedirs(output_folder, exist_ok=True)

    yt_output_path     = os.path.join(output_folder, f"{song_name}.mp4")
    shorts_output_path = os.path.join(output_folder, f"{song_name}_Shorts.mp4")

    try:
        if fast_mode:
            # HYBRID TURBO — YouTube
            if export_youtube:
                _build_single_output(
                    mp3_path=mp3_path,
                    images=images,
                    song_name=song_name,
                    artist_name=artist_name,
                    output_path=yt_output_path,
                    encoder=None,
                    resolution=yt_resolution,
                    bg_mode=bg_mode,
                    blur_level=blur_level,
                    animations=animations,
                    include_visualizer=include_visualizer,
                    visualizer_height=visualizer_height,
                    progress_callback=progress_callback,
                    tag_prefix="youtube"
                )
            # HYBRID TURBO — Shorts
            if export_shorts:
                _build_single_output(
                    mp3_path=mp3_path,
                    images=images,
                    song_name=song_name,
                    artist_name=artist_name,
                    output_path=shorts_output_path,
                    encoder=None,
                    resolution=shorts_resolution,
                    bg_mode=bg_mode,
                    blur_level=blur_level,
                    animations=animations,
                    include_visualizer=include_visualizer,
                    visualizer_height=visualizer_height,
                    progress_callback=progress_callback,
                    tag_prefix="shorts"
                )
            if progress_callback:
                progress_callback(100, "done")
            return

        # NORMAL — YouTube
        if export_youtube:
            _build_single_output(
                mp3_path=mp3_path,
                images=images,
                song_name=song_name,
                artist_name=artist_name,
                output_path=yt_output_path,
                encoder=encoder_name,
                resolution=yt_resolution,
                bg_mode=bg_mode,
                blur_level=blur_level,
                animations=animations,
                include_visualizer=include_visualizer,
                visualizer_height=visualizer_height,
                progress_callback=progress_callback,
                tag_prefix="youtube"
            )

        # NORMAL — Shorts
        if export_shorts:
            _build_single_output(
                mp3_path=mp3_path,
                images=images,
                song_name=song_name,
                artist_name=artist_name,
                output_path=shorts_output_path,
                encoder=encoder_name,
                resolution=shorts_resolution,
                bg_mode=bg_mode,
                blur_level=blur_level,
                animations=animations,
                include_visualizer=include_visualizer,
                visualizer_height=visualizer_height,
                progress_callback=progress_callback,
                tag_prefix="shorts"
            )

        if progress_callback:
            progress_callback(100, "done")

    except Exception as e:
        logger.error(f"Error during build_video_multithread: {e}", exc_info=True)
        if progress_callback:
            progress_callback(0, "error")