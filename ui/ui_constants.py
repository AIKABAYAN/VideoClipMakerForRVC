import re

# ===== Defaults =====
DEFAULT_MP3 = r"C:/Users/SIKABAYAN/Desktop/mini 2 Hati-hati Dijalan - Tulus.MP3"
DEFAULT_BATCH_FOLDER = r"C:/Users/SIKABAYAN/Desktop/dummy"
DEFAULT_BG_FOLDER = r"C:/Users/SIKABAYAN/Pictures/bg"
DEFAULT_OUTPUT = r"C:/Users/SIKABAYAN/Desktop/result"
DEFAULT_SONG = "Hati-hati di Jalan"
DEFAULT_ARTIST = "RVC by Sharkoded"
DEFAULT_COVER = r"C:/Users/SIKABAYAN/Pictures/bg/cover kau adalah.jpg"
DEFAULT_BG_MODE = "Blur"
DEFAULT_BLUR_LEVEL = 10
DEFAULT_VISUALIZER_HEIGHT = "90%"


YOUTUBE_RESOLUTIONS = {
    "HD (1280x720)": (1280, 720),
    "2K (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160)
}

def sanitize_filename(name: str) -> str:
    """Remove illegal characters for filenames"""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()
