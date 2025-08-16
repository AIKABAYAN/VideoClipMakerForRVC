import random
import math
import numpy as np
from PIL import Image, ImageFilter
from moviepy.editor import VideoClip

TRANSITIONS = ["crossfade", "slide", "zoom", "wipe", "dissolve"]

def transition_duration(per_image_duration):
    return max(0.2, min(1.0, per_image_duration * 0.33))

def create_wipe_mask(resolution, progress, direction="right"):
    W, H = resolution
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    
    if direction == "right":
        width = int(W * progress)
        draw.rectangle([0, 0, width, H], fill=255)
    elif direction == "left":
        width = int(W * (1 - progress))
        draw.rectangle([width, 0, W, H], fill=255)
    elif direction == "down":
        height = int(H * progress)
        draw.rectangle([0, 0, W, height], fill=255)
    elif direction == "up":
        height = int(H * (1 - progress))
        draw.rectangle([0, height, W, H], fill=255)
    
    return np.array(mask)

def wipe_mask_clip(resolution, total_duration, trans_duration, direction="right"):
    def make_frame(t):
        if t <= trans_duration:
            prog = max(0.0, min(1.0, t / trans_duration))
        else:
            prog = 1.0
        return create_wipe_mask(resolution, prog, direction)
    return VideoClip(make_frame=make_frame, ismask=True).set_duration(total_duration)

def apply_random_transition(prev_clip, next_clip, per_image_duration, resolution):
    trans = random.choice(TRANSITIONS)
    td = transition_duration(per_image_duration)
    start_time_next = prev_clip.start + prev_clip.duration - td

    # Default transition (crossfade)
    prev = prev_clip.fadeout(td)
    nxt = next_clip.set_start(start_time_next).fadein(td)

    if trans == "slide":
        direction = random.choice(["left", "right", "up", "down"])
        W, H = resolution
        s0 = start_time_next
        
        def pos_func(t):
            rel = max(0.0, min(td, t - s0))
            prog = rel / td if td > 0 else 1.0
            if direction == "left":
                x = int((1.0 - prog) * W)
                return (x, "center")
            elif direction == "right":
                x = int((-1.0 + prog) * W)
                return (x, "center")
            elif direction == "up":
                y = int((1.0 - prog) * H)
                return ("center", y)
            else:  # down
                y = int((-1.0 + prog) * H)
                return ("center", y)
        
        prev = prev_clip.fadeout(td)
        nxt = next_clip.set_start(s0).set_position(pos_func).fadein(td)

    elif trans == "zoom":
        s0 = start_time_next
        def scale_func(t):
            rel = max(0.0, min(td, t - s0))
            prog = rel / td if td > 0 else 1.0
            return 1.2 - 0.2 * prog
        prev = prev_clip.fadeout(td)
        nxt = next_clip.set_start(s0).resize(scale_func).fadein(td)

    elif trans == "wipe":
        direction = random.choice(["left", "right", "up", "down"])
        mask = wipe_mask_clip(resolution, next_clip.duration, td, direction=direction)
        prev = prev_clip.fadeout(td)
        nxt = next_clip.set_start(start_time_next).set_mask(mask)

    return prev, nxt, td