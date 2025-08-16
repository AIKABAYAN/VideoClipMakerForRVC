import math
import random
import logging
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from moviepy.editor import ImageClip, VideoClip, CompositeVideoClip, TextClip

logger = logging.getLogger(__name__)

IMAGE_ANIMATIONS = [
    "zoom-in", "zoom-out",
    "pan-left", "pan-right", "pan-up", "pan-down",
    "rotate-ccw", "rotate-cw",
    "fade-in", "fade-out",
    "bounce", "glow-pulse", "tilt-swing", "color-shift", "pixelate",
    "squeeze", "radial-blur", "shake", "parallax", "shadow-pulse",
    "mosaic", "lens-flare", "watercolor", "neon-edge", "glitch",
    "warp", "duotone", "texture", "circle-reveal", "time-freeze"
]

def apply_image_animation(image_clip, duration, resolution, animations=None):
    available_anims = animations if animations is not None else IMAGE_ANIMATIONS
    if not available_anims:
        logger.debug("No animations selected.")
        return image_clip
    
    animation = random.choice(available_anims)
    W, H = resolution
    logger.info(f"Applying animation '{animation}' | duration={duration:.2f}s | resolution={W}x{H}")

    if animation.startswith("zoom"):
        scale_start, scale_end = (0.8, 1.0) if animation == "zoom-in" else (1.0, 0.8)
        return image_clip.resize(lambda t: scale_start + (scale_end - scale_start) * min(1.0, t / duration))

    elif animation.startswith("pan"):
        pan_amount = 0.15
        direction = animation.split("-")[1]
        if direction == "left":
            x_start, x_end, y_start, y_end = W * pan_amount, -W * pan_amount, 0, 0
        elif direction == "right":
            x_start, x_end, y_start, y_end = -W * pan_amount, W * pan_amount, 0, 0
        elif direction == "up":
            x_start, x_end, y_start, y_end = 0, 0, H * pan_amount, -H * pan_amount
        else:
            x_start, x_end, y_start, y_end = 0, 0, -H * pan_amount, H * pan_amount

        return image_clip.set_position(lambda t: (
            x_start + (x_end - x_start) * min(1.0, t / duration),
            y_start + (y_end - y_start) * min(1.0, t / duration)
        ))

    elif animation.startswith("rotate"):
        direction = 1 if animation.endswith("cw") else -1
        return image_clip.rotate(lambda t: direction * 3 * math.sin(min(1.0, t / duration) * math.pi))

    elif animation.startswith("fade"):
        return image_clip.fadein(duration) if animation == "fade-in" else image_clip.fadeout(duration)

    elif animation == "bounce":
        return image_clip.set_position(lambda t: ("center", H // 2 - (H * 0.05 * math.sin(min(1.0, t / duration) * math.pi * 4))))

    elif animation == "glow-pulse":
        def opacity_func(t):
            progress = min(1.0, t / duration)
            return 0.6 + 0.4 * math.sin(progress * math.pi * 2)
        new_clip = image_clip.copy()
        if new_clip.mask is None:
            new_clip.mask = ImageClip(
                np.ones((H, W), dtype=np.uint8) * 255,
                ismask=True
            ).set_duration(image_clip.duration)
        new_clip.mask = new_clip.mask.fl(lambda gf, t: opacity_func(t) * gf(t))
        return new_clip

    elif animation == "tilt-swing":
        return image_clip.rotate(lambda t: 5 * math.sin(min(1.0, t / duration) * math.pi * 2))

    elif animation == "color-shift":
        def apply_color_shift(get_frame, t):
            progress = min(1.0, t / duration)
            factor = 1.0 + progress * 0.3
            return np.minimum(255, get_frame(t) * factor).astype(np.uint8)
        return image_clip.fl(apply_color_shift) if hasattr(image_clip, 'fl') else image_clip

    elif animation == "pixelate":
        def size_func(t):
            progress = min(1.0, t / duration)
            return max(2, int(20 * (1 - progress)))
        def apply_pixelate(get_frame, t):
            size = size_func(t)
            pil_img = Image.fromarray(get_frame(t))
            small = pil_img.resize((pil_img.width // size, pil_img.height // size), Image.NEAREST)
            return np.array(small.resize(pil_img.size, Image.NEAREST))
        return image_clip.fl(apply_pixelate)

    elif animation == "squeeze":
        return image_clip.resize(lambda t: (1.0, 1.0 + 0.2 * math.sin(min(1.0, t / duration) * math.pi * 2)))

    elif animation == "radial-blur":
        def apply_radial_blur(get_frame, t):
            radius = int(10 * min(1.0, t / duration))
            return np.array(Image.fromarray(get_frame(t)).filter(ImageFilter.GaussianBlur(radius)))
        return image_clip.fl(apply_radial_blur)

    elif animation == "shake":
        return image_clip.set_position(lambda t: ("center", "center"))

    elif animation == "parallax":
        return image_clip.set_position(lambda t: ("center", "center"))

    elif animation == "shadow-pulse":
        def opacity_func(t):
            progress = min(1.0, t / duration)
            return 0.8 + 0.2 * math.sin(progress * math.pi * 2)
        new_clip = image_clip.copy()
        if new_clip.mask is None:
            new_clip.mask = ImageClip(
                np.ones((H, W), dtype=np.uint8) * 255,
                ismask=True
            ).set_duration(image_clip.duration)
        new_clip.mask = new_clip.mask.fl(lambda gf, t: opacity_func(t) * gf(t))
        return new_clip

    elif animation == "mosaic":
        def mosaic_frame(frame, tiles):
            h, w = frame.shape[:2]
            small = frame[::tiles, ::tiles]
            return np.kron(small, np.ones((tiles, tiles, 1), dtype=np.uint8))
        def apply_mosaic(get_frame, t):
            tiles = max(2, int(20 * (1 - min(1.0, t / duration))))
            return mosaic_frame(get_frame(t), tiles)
        return image_clip.fl(apply_mosaic)

    elif animation == "lens-flare":
        start_x = random.choice([0.2, 0.8])
        start_y, end_y = 0.1, 0.9
        end_x = 1 - start_x
        control_x, control_y = random.uniform(0.3, 0.7), random.uniform(0.3, 0.7)
        def flare_func(get_frame, t):
            progress = min(1.0, t / duration)
            flare_opacity = 0.7 * math.sin(progress * math.pi)
            frame = get_frame(t)
            h, w = frame.shape[:2]
            y, x = np.ogrid[:h, :w]
            bezier_progress = progress**0.5
            center_x = (1-bezier_progress)**2 * start_x * w + 2*(1-bezier_progress)*bezier_progress * control_x * w + bezier_progress**2 * end_x * w
            center_y = (1-bezier_progress)**2 * start_y * h + 2*(1-bezier_progress)*bezier_progress * control_y * h + bezier_progress**2 * end_y * h
            radius = min(w, h) // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            gradient = np.clip(1 - distance / radius, 0, 1)
            overlay = np.zeros_like(frame)
            overlay[..., 0] = 255 * gradient * random.uniform(0.7, 1.0)
            overlay[..., 1] = 255 * gradient * random.uniform(0.4, 0.7)
            overlay[..., 2] = 255 * gradient * 0.3
            return np.clip(frame + overlay * flare_opacity, 0, 255).astype(np.uint8)
        return image_clip.fl(flare_func)

    elif animation == "watercolor":
        def apply_watercolor(get_frame, t):
            pil_img = Image.fromarray(get_frame(t))
            blurred = pil_img.filter(ImageFilter.GaussianBlur(3))
            arr = np.array(blurred)
            mask = arr > 128
            arr[mask] = 255
            arr[~mask] = 0
            return arr
        return image_clip.fl(apply_watercolor)

    elif animation == "neon-edge":
        def apply_neon_edge(get_frame, t):
            radius = 1 + 2 * math.sin(min(1.0, t / duration) * math.pi * 2)
            return np.array(Image.fromarray(get_frame(t)).filter(ImageFilter.GaussianBlur(radius)))
        return image_clip.fl(apply_neon_edge)

    elif animation == "glitch":
        def apply_glitch(get_frame, t):
            offset = int(W * 0.01 * math.sin(min(1.0, t / duration) * math.pi * 10))
            return np.roll(get_frame(t), offset, axis=1)
        return image_clip.fl(apply_glitch)

    elif animation == "warp":
        def warp_effect(get_frame, t):
            frame = get_frame(t)
            h, w = frame.shape[:2]
            y, x = np.indices((h, w))
            warp_amount = 10 * math.sin(min(1.0, t / duration) * math.pi)
            x_warped = np.clip(x + (warp_amount * np.sin(y * 0.05)).astype(int), 0, w-1)
            y_warped = np.clip(y + (warp_amount * np.sin(x * 0.05)).astype(int), 0, h-1)
            return frame[y_warped, x_warped]
        return image_clip.fl(warp_effect)

    elif animation == "duotone":
        def apply_duotone(frame):
            colored = np.minimum(255, (1.5 * frame)).astype(np.uint8)
            bw = np.dot(colored[...,:3], [0.2989, 0.5870, 0.1140])
            return np.dstack([bw, bw, bw])
        return image_clip.fl_image(apply_duotone)

    elif animation == "texture":
        def grain_func(t):
            noise_img = Image.effect_noise(resolution, int(min(1.0, t / duration) * 50))
            return np.array(noise_img.convert('RGB'))
        overlay = ImageClip(grain_func(0), ismask=False).set_opacity(0.3)
        comp = CompositeVideoClip([image_clip, overlay])
        if comp.mask is None and image_clip.mask is not None:
            comp.mask = image_clip.mask.copy()
        return comp

    elif animation == "circle-reveal":
        def mask_func(t):
            radius = int(max(W, H) * min(1.0, t / duration))
            mask = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((W//2-radius, H//2-radius, W//2+radius, H//2+radius), fill=255)
            return np.array(mask)
        mask_clip = VideoClip(lambda t: mask_func(t), ismask=True).set_duration(duration)
        return image_clip.set_mask(mask_clip)

    elif animation == "time-freeze":
        def apply_time_freeze(get_frame, t):
            progress = min(1.0, t / duration)
            blur_radius = 10 * (progress if progress < 0.5 else 1 - progress)
            return np.array(Image.fromarray(get_frame(t)).filter(ImageFilter.GaussianBlur(blur_radius)))
        return image_clip.fl(apply_time_freeze)

    return image_clip
