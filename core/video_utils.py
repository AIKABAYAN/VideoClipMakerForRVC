from PIL import Image

if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

import os
import platform
import subprocess
from moviepy.config import change_settings
from tqdm import tqdm

from core.logger import get_logger

logger = get_logger("sikabayan")

# Console colors
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RESET = "\033[0m"
CYAN = "\033[96m"

# --- Resource Management ---
CPU_USAGE_PERCENT = 0.40
MAX_WORKERS = max(1, int(os.cpu_count() * CPU_USAGE_PERCENT))


def log_frame(stage, message, color=""):
    """Logs a formatted message to the console using tqdm for clean output."""
    border = "═" * (len(stage) + len(message) + 5)
    tqdm.write(f"╔{border}╗")
    tqdm.write(f"║ {color}{stage}{RESET} | {message} ║")
    tqdm.write(f"╚{border}╝")


def check_ffmpeg():
    """Checks if FFmpeg is installed and available in the system's PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except Exception:
        raise RuntimeError("FFmpeg not found. Please install it and ensure it's in your system's PATH.")


def detect_best_encoder():
    """Detects the fastest available hardware encoder."""
    check_ffmpeg()
    try:
        enc_list = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True
        ).stdout
    except Exception as e:
        raise RuntimeError(f"Failed to query FFmpeg encoders: {e}")

    if "h264_nvenc" in enc_list:
        try:
            if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
                logger.info("NVIDIA NVENC encoder detected.")
                return "h264_nvenc", {"preset": "p1", "rc": "vbr"}
        except Exception:
            pass
    if "h264_qsv" in enc_list:
        logger.info("Intel QuickSync (QSV) encoder detected.")
        return "h264_qsv", {"preset": "veryfast"}
    if "h264_amf" in enc_list:
        logger.info("AMD AMF encoder detected.")
        return "h264_amf", {"quality": "speed"}
    if "libx264" in enc_list:
        logger.info("Using CPU-based libx264 encoder (ultrafast preset).")
        return "libx264", {"preset": "ultrafast", "tune": "zerolatency"}
    raise RuntimeError("No suitable H.264 encoder found.")


def configure_imagemagick():
    """Configures the path to the ImageMagick binary for Windows."""
    if platform.system() == "Windows":
        for path in [
            r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe",
            r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
        ]:
            if os.path.exists(path):
                try:
                    change_settings({"IMAGEMAGICK_BINARY": path})
                    logger.info(f"ImageMagick binary configured at: {path}")
                    return True
                except Exception as e:
                    logger.error(f"Error configuring ImageMagick: {e}")
        logger.warning("ImageMagick binary not found. Text features may fail.")
        return False
    return True


def select_bitrate_for_resolution(resolution):
    """Selects a video bitrate based on resolution."""
    pixels = resolution[0] * resolution[1]
    if pixels > 1920 * 1080:
        return "12000k"
    if pixels > 1280 * 720:
        return "8000k"
    return "5000k"


def get_font_path():
    """Finds a usable TrueType font on the system."""
    if platform.system() == "Windows":
        return "C:/Windows/Fonts/arial.ttf"
    elif platform.system() == "Linux":
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        ]:
            if os.path.exists(path):
                return path
    elif platform.system() == "Darwin":  # macOS
        return "/System/Library/Fonts/Supplemental/Arial.ttf"
    return None
