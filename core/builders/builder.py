import os
import tempfile
import platform
import shutil
import subprocess
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue
from os.path import basename, splitext
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from moviepy.video.fx import all as vfx

from moviepy.editor import (
    ImageClip, concatenate_videoclips, AudioFileClip,
    CompositeVideoClip, TextClip, VideoFileClip, VideoClip 
)
from moviepy.video.fx import all as vfx
from moviepy.config import change_settings
from core.utils import list_images
from core.logger import get_logger
from tqdm import tqdm

logger = get_logger("sikabayan")

# =========================
# Console Frame Logging
# =========================
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def log_frame(stage, message, color=""):
    border = "═" * (len(stage) + len(message) + 5)
    tqdm.write(f"╔{border}╗")
    tqdm.write(f"║ {color}{stage}{RESET} | {message} ║")
    tqdm.write(f"╚{border}╝")

# =========================
# Utilities
# =========================
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

def create_blurred_background(img_path, resolution, blur_level, temp_dir):
    img = Image.open(img_path).convert("RGB")
    img_bg = img.resize(resolution, Image.LANCZOS)
    radius = max(1, int((11 - max(1, min(10, blur_level))) * 4))
    img_bg = img_bg.filter(ImageFilter.GaussianBlur(radius))
    temp_path = os.path.join(temp_dir, f"blur_{os.path.basename(img_path)}.jpg")
    img_bg.save(temp_path, quality=95)
    return temp_path

def _select_bitrate_for_resolution(resolution):
    w, h = resolution
    max_dim = max(w, h)
    if max_dim >= 3840:
        return "50000k"
    if max_dim >= 2560:
        return "20000k"
    return "8000k"

# =========================
# Image Animations (Updated to use selected animations)
# =========================
IMAGE_ANIMATIONS = [
    # Original animations
    "zoom-in", "zoom-out", 
    "pan-left", "pan-right", "pan-up", "pan-down",
    "rotate-ccw", "rotate-cw",
    "fade-in", "fade-out",
    
    # New animations (20 total)
    "bounce", "glow-pulse", "tilt-swing", "color-shift", "pixelate",
    "squeeze", "radial-blur", "shake", "parallax", "shadow-pulse",
    "mosaic", "lens-flare", "watercolor", "neon-edge", "glitch",
    "warp", "duotone", "texture", "circle-reveal","time-freeze"
]

def _apply_image_animation(image_clip, duration, resolution, animations=None):
    """Apply random animation to the foreground image only"""
    available_anims = animations if animations is not None else IMAGE_ANIMATIONS
    if not available_anims:
        return image_clip  # No animations selected
    
    animation = random.choice(available_anims)
    W, H = resolution
    
    # Original animations (unchanged)
    if animation.startswith("zoom"):
        if animation == "zoom-in":
            scale_start = 0.8
            scale_end = 1.0
        else:  # zoom-out
            scale_start = 1.0
            scale_end = 0.8
            
        def scale_func(t):
            progress = min(1.0, t / duration)
            return scale_start + (scale_end - scale_start) * progress
            
        return image_clip.resize(scale_func)
    
    elif animation.startswith("pan"):
        pan_amount = 0.15
        direction = animation.split("-")[1]
        
        if direction == "left":
            x_start = W * pan_amount
            x_end = -W * pan_amount
            y_start = 0
            y_end = 0
        elif direction == "right":
            x_start = -W * pan_amount
            x_end = W * pan_amount
            y_start = 0
            y_end = 0
        elif direction == "up":
            x_start = 0
            x_end = 0
            y_start = H * pan_amount
            y_end = -H * pan_amount
        else:  # down
            x_start = 0
            x_end = 0
            y_start = -H * pan_amount
            y_end = H * pan_amount
            
        def pos_func(t):
            progress = min(1.0, t / duration)
            x = x_start + (x_end - x_start) * progress
            y = y_start + (y_end - y_start) * progress
            return (x, y)
            
        return image_clip.set_position(pos_func)
    
    elif animation.startswith("rotate"):
        max_angle = 3
        direction = 1 if animation.endswith("cw") else -1
        
        def rotate_func(t):
            progress = min(1.0, t / duration)
            angle = direction * max_angle * math.sin(progress * math.pi)
            return angle
            
        return image_clip.rotate(rotate_func)
    
    elif animation.startswith("fade"):
        if animation == "fade-in":
            return image_clip.fadein(duration)
        else:  # fade-out
            return image_clip.fadeout(duration)
    
    # Fixed animations
    elif animation == "bounce":
        def pos_func(t):
            progress = min(1.0, t / duration)
            bounce = H * 0.05 * math.sin(progress * math.pi * 4)
            return ("center", H//2 - bounce)
        return image_clip.set_position(pos_func)
    
    elif animation == "glow-pulse":
        def opacity_func(t):
            progress = min(1.0, t / duration)
            return float(0.6 + 0.4 * math.sin(progress * math.pi * 2))
        
        new_clip = image_clip.copy()
        
        # Only apply mask effect if mask exists
        if new_clip.mask is not None:
            new_clip.mask = new_clip.mask.fl(lambda gf, t: opacity_func(t) * gf(t))
        
        return new_clip.set_opacity(opacity_func)
    
    elif animation == "tilt-swing":
        def rotate_func(t):
            progress = min(1.0, t / duration)
            return 5 * math.sin(progress * math.pi * 2)
        return image_clip.rotate(rotate_func)
    
    elif animation == "color-shift":
        def apply_color_shift(get_frame, t):
            progress = min(1.0, t / duration)
            factor = 1.0 + progress * 0.3
            frame = get_frame(t)
            return np.minimum(255, (frame * factor)).astype(np.uint8)
        
        if hasattr(image_clip, 'fl'):
            return image_clip.fl(apply_color_shift)
        return image_clip
    
    elif animation == "pixelate":
        def size_func(t):
            progress = min(1.0, t / duration)
            size = max(2, int(20 * (1 - progress)))  # Ensure minimum size of 2
            return (size, size)
        
        # Add validation
        try:
            return image_clip.fx(vfx.resize, lambda t: size_func(t))
        except Exception as e:
            logger.error(f"Pixelate animation failed: {e}")
            return image_clip  # Fallback to original
    
    elif animation == "squeeze":
        def resize_func(t):
            progress = min(1.0, t / duration)
            scale_y = 1.0 + 0.2 * math.sin(progress * math.pi * 2)
            return (1.0, scale_y)
        return image_clip.resize(resize_func)
    
    elif animation == "radial-blur":
        def apply_radial_blur(get_frame, t):
            progress = min(1.0, t / duration)
            radius = int(10 * progress)
            frame = get_frame(t)
            
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(frame)
            # Apply Gaussian blur
            blurred_img = pil_img.filter(ImageFilter.GaussianBlur(radius))
            # Convert back to numpy array
            return np.array(blurred_img)
        
        if hasattr(image_clip, 'fl'):
            return image_clip.fl(apply_radial_blur)
        return image_clip
    
    elif animation == "shake":
        def pos_func(t):
            dx = W * 0.01 * random.uniform(-1, 1)
            dy = H * 0.01 * random.uniform(-1, 1)
            return ("center" + dx, "center" + dy)
        return image_clip.set_position(pos_func)
    
    elif animation == "parallax":
        def pos_func(t):
            progress = min(1.0, t / duration)
            offset = W * 0.1 * progress
            return ("center" + offset, "center")
        return image_clip.set_position(pos_func)
    
    elif animation == "shadow-pulse":
        def opacity_func(t):
            progress = min(1.0, t / duration)
            return float(0.8 + 0.2 * math.sin(progress * math.pi * 2))
        
        new_clip = image_clip.copy()
        if new_clip.mask is None:
            new_clip.mask = ImageClip(np.ones((resolution[1], resolution[0]), dtype=np.uint8)*255,
                                    ismask=True).set_duration(image_clip.duration)
        return new_clip.set_opacity(opacity_func)
        
    elif animation == "mosaic":
        def fx_func(t):
            try:
                # Handle both scalar and array time inputs
                if isinstance(t, np.ndarray):
                    t = t.item() if t.size == 1 else t[0]
                
                progress = min(1.0, float(t) / duration)
                tiles = max(2, int(20 * (1 - progress)))
                
                def pixelate_frame(frame):
                    h, w = frame.shape[:2]
                    small = frame[::tiles, ::tiles]
                    return np.kron(small, np.ones((tiles, tiles, 1), dtype=np.uint8))
                
                return pixelate_frame
            except Exception as e:
                logger.error(f"Mosaic effect failed: {e}")
                return lambda frame: frame  # Return original frame on error
        
        return image_clip.fl_image(fx_func(0))  # Initialize with time=0
    
    elif animation == "lens-flare":
        def flare_func(get_frame, t):
            progress = min(1.0, t / duration)
            flare_opacity = float(0.7 * math.sin(progress * math.pi))
            frame = get_frame(t)
            h, w = frame.shape[:2]
            overlay = np.zeros_like(frame)
            y, x = np.ogrid[:h, :w]
            center_x, center_y = w//2, h//2
            radius = min(w, h) // 8
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            overlay[mask] = 255
            return np.clip(frame + overlay * flare_opacity, 0, 255).astype(np.uint8)
        
        if hasattr(image_clip, 'fl'):
            try:
                return image_clip.fl(flare_func)
            except Exception as e:
                logger.warning(f"Lens flare effect failed: {e}")
                return image_clip
        return image_clip
    
    elif animation == "watercolor":
        return image_clip.fx(vfx.gaussian_blur, 3).fx(vfx.posterize, 4)
    
    elif animation == "neon-edge":
        def apply_neon_edge(get_frame, t):
            try:
                progress = min(1.0, t / duration)
                radius = 1 + 2 * math.sin(progress * math.pi * 2)
                frame = get_frame(t)
                
                # Convert to PIL, apply filter, convert back
                pil_img = Image.fromarray(frame)
                blurred_img = pil_img.filter(ImageFilter.GaussianBlur(radius))
                return np.array(blurred_img)
            except Exception as e:
                logger.warning(f"Neon edge effect failed: {e}")
                return get_frame(t)  # fallback to original frame
        
        if hasattr(image_clip, 'fl'):
            return image_clip.fl(apply_neon_edge)
        return image_clip.fx(vfx.gaussian_blur, 2)  # fallback
    
    elif animation == "glitch":
        def apply_glitch(get_frame, t):
            progress = min(1.0, t / duration)
            offset = int(W * 0.01 * math.sin(progress * math.pi * 10))
            frame = get_frame(t)
            return np.roll(frame, offset, axis=1)
        
        if hasattr(image_clip, 'fl'):
            return image_clip.fl(apply_glitch)
        return image_clip
    
    elif animation == "warp":
        def warp_effect(get_frame, t):
            progress = min(1.0, t / duration)
            frame = get_frame(t)
            h, w = frame.shape[:2]
            y, x = np.indices((h, w))
            warp_amount = 10 * math.sin(progress * math.pi)
            x_warped = x + (warp_amount * np.sin(y * 0.05)).astype(int)
            x_warped = np.clip(x_warped, 0, w-1)
            y_warped = y + (warp_amount * np.sin(x * 0.05)).astype(int)
            y_warped = np.clip(y_warped, 0, h-1)
            return frame[y_warped, x_warped]
        
        if hasattr(image_clip, 'fl'):
            return image_clip.fl(warp_effect)
        return image_clip
    
    elif animation == "duotone":
        def apply_duotone(frame):
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            colored = np.minimum(255, (1.5 * frame)).astype(np.uint8)
            bw = np.dot(colored[...,:3], [0.2989, 0.5870, 0.1140])
            return np.dstack([bw, bw, bw])
        
        return image_clip.fl_image(apply_duotone)
    
    elif animation == "texture":
        def grain_func(t):
            progress = min(1.0, t / duration)
            noise_img = Image.effect_noise(resolution, int(progress * 50))
            return np.array(noise_img.convert('RGB'))
        
        overlay = ImageClip(grain_func(0), ismask=False).set_opacity(0.3)
        comp = CompositeVideoClip([image_clip, overlay])
        if comp.mask is None and image_clip.mask is not None:
            comp.mask = image_clip.mask.copy()
        return comp
    
    elif animation == "circle-reveal":
        def mask_func(t):
            progress = min(1.0, t / duration)
            radius = int(max(W,H) * progress)
            mask = Image.new("L", (W,H), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((W//2-radius, H//2-radius, W//2+radius, H//2+radius), fill=255)
            return np.array(mask)
        
        # Create a VideoClip that generates the mask frames
        mask_clip = VideoClip(lambda t: mask_func(t), ismask=True).set_duration(duration)
        return image_clip.set_mask(mask_clip)
    
    elif animation == "time-freeze":
        def apply_time_freeze(get_frame, t):
            progress = min(1.0, t / duration)
            if progress < 0.5:
                blur_radius = 10 * progress
            else:
                blur_radius = 10 * (1 - progress)
            
            frame = get_frame(t)
            
            # Convert to PIL Image for blurring
            pil_img = Image.fromarray(frame)
            blurred_img = pil_img.filter(ImageFilter.GaussianBlur(blur_radius))
            return np.array(blurred_img)
        
        if hasattr(image_clip, 'fl'):
            return image_clip.fl(apply_time_freeze)
        return image_clip
    
    return image_clip  # Default: no animation

# =========================
# Transitions (Randomized)
# =========================
TRANSITIONS = ["crossfade", "slide", "zoom", "wipe", "dissolve"]

def _transition_duration(per_image_duration):
    # cap transition at 1s or 33% of per-image duration (whichever lower)
    return max(0.2, min(1.0, per_image_duration * 0.33))

def _wipe_mask_clip(resolution, total_duration, trans_duration, direction="right"):
    W, H = resolution
    def make_frame(t):
        # t is relative to next_clip start (mask clip timeline)
        import numpy as _np
        if t <= trans_duration:
            prog = max(0.0, min(1.0, t / trans_duration))
        else:
            prog = 1.0
        if direction in ("right", "left"):
            x = _np.linspace(0, 1, W, dtype=float)
            if direction == "right":
                col = (x <= prog).astype(float)
            else:  # left
                col = (x >= 1.0 - prog).astype(float)
            frame = _np.tile(col, (H, 1))
        else:
            y = _np.linspace(0, 1, H, dtype=float).reshape(H, 1)
            if direction == "down":
                row = (y <= prog).astype(float)
            else:  # up
                row = (y >= 1.0 - prog).astype(float)
            frame = _np.tile(row, (1, W))
        return frame
    from moviepy.editor import VideoClip
    return VideoClip(make_frame=make_frame, ismask=True).set_duration(total_duration)

def _apply_random_transition(prev_clip, next_clip, per_image_duration, resolution):
    """
    Returns a tuple (prev_timed, next_timed, overlap_duration)
    prev_timed and next_timed are clips with proper .start, fades, masks, positions, etc.
    """
    trans = random.choice(TRANSITIONS)
    td = _transition_duration(per_image_duration)
    start_time_next = prev_clip.start + prev_clip.duration - td

    # Defaults: no effect, simple overlap with small crossfade to avoid seams
    prev = prev_clip.fadeout(td)
    nxt = next_clip.set_start(start_time_next).fadein(td)

    if trans == "crossfade" or trans == "dissolve":
        # dissolve == crossfade (soft)
        prev = prev_clip.crossfadeout(td)
        nxt = next_clip.set_start(start_time_next).crossfadein(td)

    elif trans == "slide":
        direction = random.choice(["left", "right", "up", "down"])
        W, H = resolution
        s0 = start_time_next
        def pos_func(t):
            # t is absolute time
            rel = max(0.0, min(td, t - s0))
            prog = rel / td if td > 0 else 1.0
            if direction == "left":
                x = int((1.0 - prog) * W)  # from right edge to center
                return (x, "center")
            if direction == "right":
                x = int((-1.0 + prog) * W)  # from left offscreen to center
                return (x, "center")
            if direction == "up":
                y = int((1.0 - prog) * H)  # from bottom to center
                return ("center", y)
            if direction == "down":
                y = int((-1.0 + prog) * H)  # from top to center
                return ("center", y)
            return ("center", "center")
        prev = prev_clip.fadeout(td)
        nxt = next_clip.set_start(s0).set_position(pos_func).fadein(td)

    elif trans == "zoom":
        s0 = start_time_next
        def scale_func(t):
            rel = max(0.0, min(td, t - s0))
            prog = rel / td if td > 0 else 1.0
            # zoom-out from 1.2 -> 1.0 during transition
            return 1.2 - 0.2 * prog
        prev = prev_clip.fadeout(td)
        nxt = next_clip.set_start(s0).resize(scale_func).fadein(td)

    elif trans == "wipe":
        direction = random.choice(["left", "right", "up", "down"])
        mask = _wipe_mask_clip(resolution, next_clip.duration, td, direction=direction)
        prev = prev_clip.fadeout(td)
        nxt = next_clip.set_start(start_time_next).set_mask(mask)

    return prev, nxt, td

# =========================
# Worker: Render chunk
# =========================
def _render_chunk(chunk_imgs, duration_per_image, bg_mode, blur_level, resolution, temp_dir, idx, animations=None):
    start_time = time.time()
    log_frame("🟢 Render", f"PID {os.getpid()} | Chunk {idx} | {len(chunk_imgs)} images", GREEN)

    # Build per-image base clips with animations
    base_clips = []
    for img in chunk_imgs:
        # Create the main image clip with animation
        main_clip = ImageClip(img).set_duration(duration_per_image)
        animated_clip = _apply_image_animation(main_clip, duration_per_image, resolution, animations)
        
        if bg_mode == "Blur":
            blur_path = create_blurred_background(img, resolution, blur_level, temp_dir)
            bg_clip = ImageClip(blur_path).set_duration(duration_per_image)
            frame_clip = CompositeVideoClip(
                [bg_clip, animated_clip.set_position("center")], 
                size=resolution
            ).set_duration(duration_per_image)
        else:
            frame_clip = CompositeVideoClip(
                [animated_clip.on_color(
                    size=resolution, 
                    color=(0, 0, 0), 
                    pos=("center", "center")
                )], 
                size=resolution
            ).set_duration(duration_per_image)
            
        base_clips.append(frame_clip)

    # Build a timeline with overlapping transitions
    timeline = []
    if not base_clips:
        raise RuntimeError("No images for this chunk")

    # First clip starts at 0
    t_cursor = 0.0
    first = base_clips[0].set_start(t_cursor)
    timeline.append(first)
    t_cursor += duration_per_image

    for i in range(1, len(base_clips)):
        prev = timeline[-1]  # last added visible clip (could already have fx)
        base_next = base_clips[i]
        prev_timed, next_timed, td = _apply_random_transition(prev, base_next, duration_per_image, resolution)

        # Replace last element with the transitioned version for prev
        timeline[-1] = prev_timed
        # Add the transitioned next
        timeline.append(next_timed)

        # Advance the cursor: add only the non-overlapped extra time
        t_cursor += duration_per_image - td

    # Compose the timeline (overlaps are baked via .start)
    final_chunk = CompositeVideoClip(timeline, size=resolution)
    # Ensure duration equals computed cursor (avoid dangling tails)
    final_chunk = final_chunk.set_duration(t_cursor)

    part_path = os.path.join(temp_dir, f"part_{idx}.mp4")
    final_chunk.write_videofile(
        part_path, fps=30, codec="libx264", audio=False,
        preset="ultrafast", bitrate=_select_bitrate_for_resolution(resolution),
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

# =========================
# Stage 2: Merge process
# =========================
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
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
         "-c", "copy", merged_path],
        check=True
    )
    log_frame("✅ Merge Done", f"Merged {total_chunks} chunks → {os.path.basename(merged_path)}", BLUE)

# =========================
# Main build
# =========================
def build_video_multithread(mp3_path, bg_folder, cover_path, song_name, artist_name,
                           output_path, resolution=(1280, 720), num_threads=None,
                           bg_mode="Blur", blur_level=3, progress_callback=None,
                           animations=None):

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

    images = list_images(bg_folder)
    if not images:
        logger.error("No background images found.")
        if progress_callback:
            progress_callback(0)
        return

    logger.info(f"Found {len(images)} background images")
    logger.info(f"Resolution: {resolution}")
    logger.info(f"Selected animations: {animations if animations else 'All'}")

    total_duration = AudioFileClip(mp3_path).duration
    duration_per_image = total_duration / len(images)
    logger.info(f"Each image will be shown for {duration_per_image:.2f} seconds")

    song_text = f"{splitext(basename(mp3_path))[0]} - {artist_name}"
    temp_dir = tempfile.mkdtemp(prefix="sikabayan_")
    merged_video_path = os.path.join(temp_dir, "merged.mp4")
    concat_list_path = os.path.join(temp_dir, "concat.txt")
    queue = Queue()

    try:
        # Stage 2 merger process
        total_cores = os.cpu_count()
        chunk_size = max(1, math.ceil(len(images) / total_cores))
        chunks = [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]

        merger_proc = Process(target=merge_worker, args=(queue, concat_list_path, merged_video_path, len(chunks)))
        merger_proc.start()

        # Stage 1 render
        with ProcessPoolExecutor(max_workers=total_cores) as executor:
            futures = {executor.submit(
                _render_chunk, chunk, duration_per_image, bg_mode,
                blur_level, resolution, temp_dir, idx, animations
            ): idx for idx, chunk in enumerate(chunks)}

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Chunks", unit="chunk")):
                part_path = future.result()
                queue.put(part_path)
                if progress_callback:
                    progress_callback((i + 1) * 40 // len(futures))

        # Tell merger to finish
        queue.put(None)
        merger_proc.join()

        # Stage 3 overlay & audio
        log_frame("🟡 Overlay", "Starting final video creation...", YELLOW)
        audio_clip = AudioFileClip(mp3_path)
        final_video = VideoFileClip(merged_video_path).set_audio(audio_clip)

        try:
            text_clip = TextClip(song_text, fontsize=40, color="white", font="Arial-Bold")
            text_width, text_height = text_clip.w, text_clip.h
            scrolling_text = text_clip.set_position(
                lambda t: (resolution[0] - (t * 100) % (text_width + resolution[0]),
                           resolution[1] - text_height - 20)
            ).set_duration(total_duration)
            final_video = CompositeVideoClip([final_video, scrolling_text])
        except Exception as e:
            logger.warning(f"Text overlay failed: {e}")

        final_video.write_videofile(
            output_path, fps=30, codec=encoder,
            preset="ultrafast", bitrate=_select_bitrate_for_resolution(resolution),
            threads=total_cores, verbose=True, logger="bar"
        )
        final_video.close()
        audio_clip.close()
        log_frame("✅ Done", f"Final video saved as {output_path}", GREEN)

        if progress_callback:
            progress_callback(100)

    except Exception as e:
        logger.exception(f"Error during build_video_multithread: {e}")
        if progress_callback:
            progress_callback(0)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)