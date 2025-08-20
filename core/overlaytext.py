# overlaytext.py
import os
import shutil
import subprocess
from moviepy.editor import VideoFileClip
from core.logger import get_logger

logger = get_logger("sikabayan")

def apply_scrolling_text_overlay(
    input_video_path,
    output_video_path,
    song_text,
    total_duration,
    resolution,
    encoder_name,
    encoder_opts,
    bitrate,
    font_color="white",
    stroke_color="black",
    stroke_width=2
):
    """
    Menambahkan teks overlay berjalan ke file video dengan FFmpeg.
    
    - Landscape (W >= H):
        teks di bawah, naik 10% dari bawah layar, font normal
    - Short (portrait, H > W):
        font lebih kecil (40% dari normal)
        teks 35% dari bawah (y = h*0.65)
    """

    try:
        logger.info("Memulai proses overlay teks dengan FFmpeg langsung...")

        # Amankan teks agar aman dipakai di command line
        sanitized_text = song_text.replace("'", "'\\''")

        # Kecepatan gulir dalam piksel per detik
        scroll_speed = 150
        
        # Resolusi video
        W, H = resolution
        margin_bottom = 20
        offset = int(H * 0.1)

        # Deteksi mode short (portrait)
        if H > W:
            # 📌 Short video
            fontsize = int(H * 0.05 * 0.4)  # 40% dari default (lebih kecil)
            y_expr = f"h*0.65"              # 35% dari bawah
        else:
            # 📌 Landscape
            fontsize = int(H * 0.05)        # default 5% tinggi video
            y_expr = f"h-th-{margin_bottom + offset}"  # naik 10% dari bawah

        # Konstruksi filter drawtext
        video_filter = (
            f"drawtext="
            f"fontfile='/path/to/your/font/Arial.ttf':"  # ⚠️ Ganti dengan path font valid
            f"text='{sanitized_text}':"
            f"fontsize={fontsize}:"
            f"fontcolor={font_color}:"
            f"bordercolor={stroke_color}:"
            f"borderw={stroke_width}:"
            f"y={y_expr}:"
            f"x='w-mod(t*{scroll_speed},w+text_w)'"
        )
        
        # Perintah FFmpeg
        command = [
            'ffmpeg',
            '-i', input_video_path,
            '-vf', video_filter,
            '-c:v', encoder_name,
            '-b:v', bitrate,
            '-preset', 'fast',
            '-c:a', 'copy',
            '-y',
            output_video_path
        ]
        
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        logger.info("Proses overlay teks FFmpeg selesai.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Gagal menambahkan teks overlay dengan FFmpeg. Error: {e.stderr}", exc_info=True)
        if not os.path.exists(output_video_path):
            shutil.copy(input_video_path, output_video_path)
    except Exception as e:
        logger.error(f"Kesalahan tak terduga: {e}", exc_info=True)
        if not os.path.exists(output_video_path):
            shutil.copy(input_video_path, output_video_path)

# --- PENTING ---
# Ganti '/path/to/your/font/Arial.ttf' dengan path font yang sesuai di sistem Anda.
#
# Contoh Path Font:
# - Windows: 'C\\:/Windows/Fonts/arialbd.ttf'
# - Linux:   '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
# - macOS:   '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
