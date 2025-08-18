# overlaytext.py
import os
import shutil
import subprocess
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from core.logger import get_logger

logger = get_logger("sikabayan")

def apply_scrolling_text_overlay(input_video_path, output_video_path, song_text, total_duration, resolution, encoder_name, encoder_opts, bitrate):
    """
    Menambahkan teks overlay berjalan ke file video yang sudah ada menggunakan FFmpeg secara langsung untuk performa maksimal.
    Catatan: Argumen 'total_duration' dan 'encoder_opts' tidak digunakan dalam implementasi ini.

    Args:
        input_video_path (str): Path ke video input.
        output_video_path (str): Path untuk menyimpan video output.
        song_text (str): Teks yang akan berjalan.
        total_duration (float): Durasi total video (tidak digunakan).
        resolution (tuple): Resolusi video (lebar, tinggi).
        encoder_name (str): Nama encoder FFmpeg yang akan digunakan (misalnya, 'libx264', 'h264_nvenc').
        encoder_opts (dict): Opsi untuk encoder (tidak digunakan, preset 'fast' digunakan secara default).
        bitrate (str): Bitrate video.
    """
    try:
        logger.info("Memulai proses overlay teks dengan FFmpeg langsung (Performa Tinggi)...")

        # Amankan teks untuk digunakan dalam command line
        sanitized_text = song_text.replace("'", "'\\''")

        # Kecepatan gulir dalam piksel per detik
        scroll_speed = 150
        
        # Ukuran font dan margin dari bawah
        fontsize = int(resolution[1] * 0.05)
        margin_bottom = 20

        # Konstruksi filter `drawtext` FFmpeg.
        # x='w-mod(t*speed,w+text_w)' menciptakan efek gulir dari kanan ke kiri.
        video_filter = (
            f"drawtext="
            f"fontfile='/path/to/your/font/Arial.ttf':"  # PENTING: Ganti dengan path font yang valid
            f"text='{sanitized_text}':"
            f"fontsize={fontsize}:"
            f"fontcolor=white:"
            f"y=h-th-{margin_bottom}:"
            f"x='w-mod(t*{scroll_speed},w+text_w)'"
        )
        
        # Perintah FFmpeg
        # -c:a copy: Salin stream audio tanpa re-encoding (sangat cepat)
        command = [
            'ffmpeg',
            '-i', input_video_path,
            '-vf', video_filter,
            '-c:v', encoder_name,
            '-b:v', bitrate,
            '-preset', 'fast',
            '-c:a', 'copy',
            '-y', # Timpa file output jika sudah ada
            output_video_path
        ]
        
        # Jalankan perintah FFmpeg. check=True akan memunculkan error jika FFmpeg gagal.
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        logger.info("Proses overlay teks FFmpeg selesai.")

    except subprocess.CalledProcessError as e:
        # Menangkap error spesifik dari FFmpeg dan menampilkannya di log
        logger.error(f"Gagal menambahkan teks overlay dengan FFmpeg. Error: {e.stderr}", exc_info=True)
        if not os.path.exists(output_video_path):
            shutil.copy(input_video_path, output_video_path)
    except Exception as e:
        logger.error(f"Terjadi kesalahan tak terduga: {e}", exc_info=True)
        if not os.path.exists(output_video_path):
            shutil.copy(input_video_path, output_video_path)

# --- PENTING ---
# Anda harus mengganti '/path/to/your/font/Arial.ttf' dengan path
# yang benar ke file font di sistem Anda.
#
# Contoh Path Font:
# - Windows: 'C\\:/Windows/Fonts/arialbd.ttf'
# - Linux:   '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
# - macOS:   '/System/Library/Fonts/Supplemental/Arial Bold.ttf'