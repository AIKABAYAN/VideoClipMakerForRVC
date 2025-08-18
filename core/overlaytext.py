# overlaytext.py
import os
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from .logger import get_logger

logger = get_logger("sikabayan")

def apply_scrolling_text_overlay(input_video_path, output_video_path, song_text, total_duration, resolution, encoder_name, encoder_opts, bitrate):
    """
    Menambahkan teks overlay berjalan ke file video yang sudah ada menggunakan MoviePy.

    Args:
        input_video_path (str): Path ke video input.
        output_video_path (str): Path untuk menyimpan video output.
        song_text (str): Teks yang akan berjalan.
        total_duration (float): Durasi total video.
        resolution (tuple): Resolusi video (lebar, tinggi).
        encoder_name (str): Nama encoder FFmpeg yang akan digunakan.
        encoder_opts (dict): Opsi untuk encoder.
        bitrate (str): Bitrate video.
    """
    base_video = None
    main_video = None
    try:
        logger.info("Memulai proses overlay teks dengan MoviePy...")
        
        # Muat video yang sudah dirender sebagai dasar
        base_video = VideoFileClip(input_video_path)

        # Buat klip teks berjalan sesuai dengan logika yang Anda berikan
        text_clip = TextClip(song_text, fontsize=40, color="white", font="Arial-Bold")
        text_width, text_height = text_clip.w, text_clip.h
        
        scrolling_text = text_clip.set_position(
            lambda t: (resolution[0] - (t * 100) % (text_width + resolution[0]),
                       resolution[1] - text_height - 20)
        ).set_duration(total_duration)

        # Gabungkan video dasar dengan teks berjalan
        main_video = CompositeVideoClip([base_video, scrolling_text], size=resolution)
        
        # Tulis video final dengan audio dari video dasar
        main_video.write_videofile(
            output_video_path,
            codec=encoder_name,
            bitrate=bitrate,
            audio=base_video.audio, # Gunakan audio dari video asli
            threads=os.cpu_count(),
            preset=encoder_opts.get("preset", "ultrafast"),
            logger='bar'
        )
        logger.info("Proses overlay teks MoviePy selesai.")

    except Exception as e:
        logger.error(f"Gagal menambahkan teks overlay dengan MoviePy: {e}", exc_info=True)
        # Jika gagal, salin saja video asli tanpa overlay
        if not os.path.exists(output_video_path):
            shutil.copy(input_video_path, output_video_path)
    finally:
        # Pastikan semua klip ditutup untuk melepaskan file
        if base_video:
            base_video.close()
        if main_video:
            main_video.close()
