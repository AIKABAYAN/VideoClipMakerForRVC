from pathlib import Path
from mutagen.mp3 import MP3
from PIL import Image
import os

def get_mp3_duration_seconds(mp3_path: Path) -> float:
    audio = MP3(str(mp3_path))
    return float(audio.info.length)

def collect_images(images_folder: Path) -> list:
    exts = [".jpg", ".jpeg", ".png"]
    return sorted([p for p in images_folder.iterdir() if p.suffix.lower() in exts])

def resize_and_pad_image(image_path: Path, resolution=(1920, 1080)):
    """
    Opens an image, resizes it to fit within resolution, and pads with black if needed.
    Returns a PIL.Image object.
    """
    img = Image.open(image_path)
    img.thumbnail(resolution, Image.LANCZOS)

    background = Image.new("RGB", resolution, (0, 0, 0))
    pos = ((resolution[0] - img.width) // 2, (resolution[1] - img.height) // 2)
    background.paste(img, pos)
    return background

def list_images(folder):
    """
    Returns a list of image file paths from the given folder.
    Supported formats: jpg, jpeg, png
    """
    if not os.path.exists(folder):
        raise ValueError(f"Folder not found: {folder}")

    supported_ext = (".jpg", ".jpeg", ".png")
    images = [os.path.join(folder, f) for f in os.listdir(folder)
              if f.lower().endswith(supported_ext)]

    # Sort for consistent order
    images.sort()
    return images