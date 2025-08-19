import os
import re
import math
import time
import shutil
import tempfile
import subprocess
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from moviepy.editor import AudioFileClip
from core.visualizer import write_visualizer_sequence
from core.overlaytext import apply_scrolling_text_overlay
from core.disclaimer_builder import save_disclaimer
from core.video_utils import (
    log_frame, check_ffmpeg, detect_best_encoder,
    select_bitrate_for_resolution, get_font_path,
    MAX_WORKERS, GREEN, YELLOW
)
from core.logger import get_logger

logger = get_logger("sikabayan")


def create_intro_frame(song_name, artist_name, background_image, resolution):
    """Creates a single frame for the video intro with adaptive text."""
    W, H = resolution
    bg = background_image.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=15))
    draw = ImageDraw.Draw(bg)

    font_path = get_font_path()
    formatted_song_name = song_name.replace(" - ", "\n").strip()
    max_text_width = W * 0.9

    # Title font
    title_font_size = int(W / 15)
    title_font = ImageFont.load_default()
    try:
        title_font = ImageFont.truetype(font_path, title_font_size)
        longest_line = max(formatted_song_name.split('\n'),
                           key=lambda line: title_font.getbbox(line)[2])
        while title_font.getbbox(longest_line)[2] > max_text_width:
            title_font_size -= 2
            title_font = ImageFont.truetype(font_path, title_font_size)
            longest_line = max(formatted_song_name.split('\n'),
                               key=lambda line: title_font.getbbox(line)[2])
    except Exception:
        logger.warning("Font not found, using default.")

    # Artist font
    artist_font_size = int(W / 25)
    artist_font = ImageFont.load_default()
    try:
        artist_font = ImageFont.truetype(font_path, artist_font_size)
        while artist_font.getbbox(artist_name)[2] > max_text_width:
            artist_font_size -= 1
            artist_font = ImageFont.truetype(font_path, artist_font_size)
    except Exception:
        pass

    y_cursor = H * 0.1
    title_bbox = draw.textbbox((0, 0), formatted_song_name, font=title_font, align="center")
    title_w = title_bbox[2] - title_bbox[0]
    title_h = title_bbox[3] - title_bbox[1]
    draw.text(((W - title_w) / 2, y_cursor),
              formatted_song_name, font=title_font, fill="white", align="center")
    y_cursor += title_h + int(H * 0.02)

    artist_bbox = draw.textbbox((0, 0), artist_name, font=artist_font)
    artist_w = artist_bbox[2] - artist_bbox[0]
    draw.text(((W - artist_w) / 2, y_cursor),
              artist_name, font=artist_font, fill=(200, 200, 200))

    return np.array(bg)


def generate_frames_task(args):
    """Worker process to generate a chunk of frames."""
    (base_frames_chunk, start_frame_index, frames_per_image,
     trans_frames, frames_dir, is_last_chunk) = args

    frame_index = start_frame_index
    for i, current_frame_np in enumerate(base_frames_chunk):
        is_last_image_in_video = is_last_chunk and (i == len(base_frames_chunk) - 1)
        static_count = frames_per_image if is_last_image_in_video else (frames_per_image - trans_frames)

        for _ in range(static_count):
            Image.fromarray(current_frame_np).save(
                os.path.join(frames_dir, f"frame_{frame_index:08d}.jpg"),
                format='JPEG', quality=95)
            frame_index += 1

        if i + 1 < len(base_frames_chunk):
            next_frame_np = base_frames_chunk[i + 1].astype(np.float32)
            current_float_np = current_frame_np.astype(np.float32)
            for k in range(trans_frames):
                alpha = (k + 1) / float(trans_frames)
                blended = ((1.0 - alpha) * current_float_np + alpha * next_frame_np)
                Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(
                    os.path.join(frames_dir, f"frame_{frame_index:08d}.jpg"),
                    format='JPEG', quality=95)
                frame_index += 1

    return frame_index - start_frame_index


def _build_single_output_hybrid(settings, tag_prefix="", progress_callback=None):
    """Core rendering pipeline, now with intro and overlay capabilities."""
    perf_summary = {}
    mp3_path, images, output_path, resolution, bg_mode, blur_level, include_visualizer, visualizer_height, song_name, artist_name = (
        settings["mp3_file"], settings["images"], settings["output_path"], settings["resolution"],
        settings["bg_mode"], settings["blur_level"], settings["include_visualizer"], settings["visualizer_height"],
        settings["song_name"], settings["artist_name"]
    )
    intro_duration = settings.get("intro_duration", 2)
    scrolling_text = settings.get("scrolling_text", "")

    audio_clip = AudioFileClip(mp3_path)
    total_duration = audio_clip.duration
    audio_clip.close()

    fps = 30
    if not images:
        images = ["__BLACK__"]

    main_content_duration = total_duration - intro_duration
    if main_content_duration <= 0:
        raise ValueError("Intro duration cannot be longer than the total audio duration.")

    n_images = len(images)
    duration_per_image = main_content_duration / n_images
    trans_sec = max(0.25, min(1.0, duration_per_image * 0.25))
    trans_frames = max(1, int(round(trans_sec * fps)))
    frames_per_image = max(trans_frames + 1, int(round(duration_per_image * fps)))
    total_frames_estimate = int(total_duration * fps)
    intro_frames_count = int(intro_duration * fps)

    if progress_callback:
        progress_callback(1, f"{tag_prefix}_preprocess")
    start_time = time.time()

    def _compose_base_frame(img_path: str) -> np.ndarray:
        W, H = resolution
        if img_path == "__BLACK__":  # fallback frame
            return np.zeros((H, W, 3), dtype=np.uint8)
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            return np.zeros((H, W, 3), dtype=np.uint8)

        bg = im.resize((W, H), Image.LANCZOS)
        if bg_mode == "Blur":
            bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_level))
        elif bg_mode == "Darken":
            bg = Image.fromarray((np.array(bg) * (1.0 - blur_level / 10.0)).astype(np.uint8))

        im.thumbnail((W, H), Image.LANCZOS)
        canvas = bg.copy()
        canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
        return np.array(canvas)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        base_frames = list(pool.map(_compose_base_frame, images))
    perf_summary['Preprocess'] = time.time() - start_time
    log_frame("🟢 Preprocess", f"{len(images)} images composited in {perf_summary['Preprocess']:.1f}s", GREEN)
    if progress_callback:
        progress_callback(5, f"{tag_prefix}_preprocess")

    # temp dir
    start_time = time.time()
    temp_dir = tempfile.mkdtemp(prefix=f"sikabayan_hybrid_{tag_prefix}_")
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # intro frames
    first_image_pil = Image.fromarray(base_frames[0])
    intro_frame_np = create_intro_frame(song_name, artist_name, first_image_pil, resolution)
    for i in range(intro_frames_count):
        Image.fromarray(intro_frame_np).save(
            os.path.join(frames_dir, f"frame_{i:08d}.jpg"),
            format='JPEG', quality=95)

    num_workers = MAX_WORKERS
    chunk_size = math.ceil(len(base_frames) / num_workers)
    tasks = []
    frame_cursor = intro_frames_count

    for i in range(num_workers):
        chunk = base_frames[i * chunk_size:(i + 1) * chunk_size]
        if not chunk:
            continue
        is_last_chunk = (i == num_workers - 1)
        tasks.append((chunk, frame_cursor, frames_per_image, trans_frames, frames_dir, is_last_chunk))
        num_images_in_chunk = len(chunk)
        num_transitions_in_chunk = num_images_in_chunk if is_last_chunk else num_images_in_chunk - 1
        frame_cursor += num_images_in_chunk * frames_per_image - num_transitions_in_chunk * trans_frames

    total_frames_written = intro_frames_count
    with tqdm(total=total_frames_estimate, desc=f"🟢 Frames-{tag_prefix}", unit="frame", colour="green", initial=intro_frames_count) as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for written_count in executor.map(generate_frames_task, tasks):
                total_frames_written += written_count
                pbar.update(written_count)
                if progress_callback:
                    percent = 5 + int(45 * (total_frames_written / total_frames_estimate))
                    progress_callback(percent, f"{tag_prefix}_frames")

    perf_summary['Frames'] = time.time() - start_time
    log_frame("🟢 Frames", f"{total_frames_written} JPEG frames written in {perf_summary['Frames']:.1f}s", GREEN)
    if progress_callback:
        progress_callback(50, f"{tag_prefix}_frames")

    # visualizer
    vis_frames_pattern = None
    if include_visualizer:
        start_time_vis = time.time()
        try:
            vis_frames_dir = os.path.join(temp_dir, "vis_frames")
            os.makedirs(vis_frames_dir, exist_ok=True)
            vis_frames_pattern = os.path.join(vis_frames_dir, "vis_frame_%08d.png")

            write_visualizer_sequence(
                mp3_path,
                fps=fps,
                resolution=resolution,
                opacity=0.6,
                scale_height=float(str(visualizer_height).strip('%')) / 100.0,
                out_pattern=vis_frames_pattern,
            )

            if progress_callback:
                progress_callback(55, f"{tag_prefix}_visualizer")

            perf_summary['Visualizer'] = time.time() - start_time_vis
            log_frame("🟡 Visualizer", f"Wrote PNG sequence in {perf_summary['Visualizer']:.1f}s", YELLOW)
        except Exception as e:
            log_frame("⚠️ Visualizer", f"Skipping due to error: {e}", YELLOW)
            vis_frames_pattern = None
    if progress_callback:
        progress_callback(60, f"{tag_prefix}_visualizer")

    # encode
    start_time = time.time()
    enc_name, enc_opts = detect_best_encoder()
    bitrate = select_bitrate_for_resolution(resolution)
    frames_pattern = os.path.join(frames_dir, "frame_%08d.jpg")

    temp_output_path = os.path.join(temp_dir, "temp_video_no_text.mp4")

    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", frames_pattern]
    cmd.extend(["-i", mp3_path])

    filter_complex_parts = []
    current_video_stream = "[0:v]"

    if vis_frames_pattern:
        cmd.extend(["-framerate", str(fps), "-i", vis_frames_pattern])
        vis_input_index = 2
        vis_y_pos = resolution[1] - int(resolution[1] * (float(str(visualizer_height).strip('%')) / 100.0)) - int(resolution[1] * 0.05)
        filter_complex_parts.append(f"{current_video_stream}[{vis_input_index}:v]overlay=x=(W-w)/2:y={vis_y_pos}[v_with_vis]")
        current_video_stream = "[v_with_vis]"

    if filter_complex_parts:
        cmd.extend(["-filter_complex", ";".join(filter_complex_parts)])
        cmd.extend(["-map", current_video_stream])
    else:
        cmd.extend(["-map", "0:v"])

    cmd.extend(["-map", "1:a"])
    cmd.extend(["-pix_fmt", "yuv420p", "-shortest", "-c:a", "aac", "-b:a", "192k", "-c:v", enc_name])
    for key, value in enc_opts.items():
        cmd.extend([f"-{key}", str(value)])
    cmd.extend(["-b:v", bitrate, temp_output_path])

    log_frame("🟡 Encode", f"Using {enc_name} @ {bitrate} (pass 1: Main + Visualizer)", YELLOW)

    try:
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True, encoding="utf-8")
        with tqdm(total=total_frames_estimate, desc=f"🟡 Encode-{tag_prefix}", unit="frame", colour="yellow") as pbar:
            for line in process.stderr:
                if match := re.search(r"frame=\s*(\d+)", line):
                    current_frame = int(match.group(1))
                    pbar.update(current_frame - pbar.n)
                    if progress_callback:
                        percent = 60 + int(35 * (current_frame / total_frames_estimate))
                        progress_callback(percent, f"{tag_prefix}_encode")

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stderr=process.stderr.read())

    except subprocess.CalledProcessError as e:
        logger.error("--- FFmpeg Command Failed ---")
        logger.error(f"Command: {' '.join(e.cmd)}")
        logger.error(f"Stderr: {e.stderr}")
        raise

    perf_summary['Encode'] = time.time() - start_time

    if scrolling_text:
        start_overlay_time = time.time()
        if progress_callback:
            progress_callback(95, f"{tag_prefix}_overlay")

        apply_scrolling_text_overlay(
            temp_output_path,
            output_path,
            scrolling_text,
            total_duration,
            resolution,
            enc_name,
            enc_opts,
            bitrate
        )
        perf_summary['Overlay'] = time.time() - start_overlay_time
    else:
        shutil.move(temp_output_path, output_path)

    # ✅ Cleanup temporary dirs
    try:
        if vis_frames_pattern:
            shutil.rmtree(os.path.dirname(vis_frames_pattern), ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        log_frame("🧹 Cleanup", f"Removed temp dirs for {tag_prefix}", YELLOW)
    except Exception as e:
        log_frame("⚠️ Cleanup", f"Failed to remove temp dirs: {e}", YELLOW)

    log_frame("✅ Done", f"Video created at {output_path}", GREEN)

    if progress_callback:
        progress_callback(100, f"{tag_prefix}_done")
    save_disclaimer(mp3_path, artist_name, output_path)
    return perf_summary
