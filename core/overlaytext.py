# overlaytext.py
from .logger import get_logger

logger = get_logger("sikabayan")

def create_scrolling_text_filter(scrolling_text, resolution, font_path, scroll_speed=100):
    """
    Membuat string filter 'drawtext' untuk FFmpeg untuk membuat teks berjalan.

    Args:
        scrolling_text (str): Teks yang akan ditampilkan.
        resolution (tuple): Resolusi video (lebar, tinggi).
        font_path (str): Path ke file font TrueType.
        scroll_speed (int): Kecepatan teks berjalan dalam piksel per detik.

    Returns:
        str: String filter FFmpeg 'drawtext' atau string kosong jika tidak ada teks/font.
    """
    if not scrolling_text or not font_path:
        return ""

    try:
        # Menyesuaikan path font untuk filter FFmpeg di Windows
        escaped_font_path = font_path.replace('\\', '/').replace(':', '\\\\:')
        font_size = int(resolution[1] / 25)
        
        # Ekspresi untuk posisi x yang membuat teks berjalan dari kanan ke kiri
        x_pos = f"w-mod(t*{scroll_speed},w+tw)"
        
        # Membuat string filter
        filter_string = (
            f"drawtext="
            f"text='{scrolling_text}':"
            f"fontfile='{escaped_font_path}':"
            f"fontsize={font_size}:"
            f"fontcolor=white@0.8:"
            f"x={x_pos}:"
            f"y=h-th-20" # Posisi di bagian bawah
        )
        return filter_string
        
    except Exception as e:
        logger.warning(f"Gagal membuat filter teks overlay: {e}")
        return ""
