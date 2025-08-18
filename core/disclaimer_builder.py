# disclaimer_builder.py
import os
from os.path import basename
from .logger import get_logger

logger = get_logger("sikabayan")

def save_disclaimer(mp3_path, artist_name, output_path):
    """
    Generates a disclaimer by reading a template from 'doc/disclaimer.txt',
    replacing tags, and saving it to a text file.
    """
    base_name = os.path.splitext(basename(mp3_path))[0]
    output_txt = os.path.splitext(output_path)[0] + ".txt"
    disclaimer_template_path = "doc/disclaimer.txt"

    try:
        if os.path.exists(disclaimer_template_path):
            with open(disclaimer_template_path, "r", encoding="utf-8") as f:
                template = f.read()

            # --- Logika baru untuk memecah judul lagu dan artis asli ---
            song_title_hashtag = ""
            ori_artist_hashtag = ""

            # Cek apakah nama file mengandung ' - ' untuk dipecah
            if ' - ' in base_name:
                # Pecah pada tanda hubung terakhir untuk memisahkan judul utama dari bagian artis terakhir
                parts = base_name.rsplit(' - ', 1)
                if len(parts) == 2:
                    song_title = parts[0].strip()
                    ori_artist = parts[1].strip()
                    
                    song_title_hashtag = f"#{song_title.replace(' ', '')}"
                    ori_artist_hashtag = f"#{ori_artist.replace(' ', '')}"

            # Ganti placeholder standar
            disclaimer = template.replace("{base_name}", base_name)
            disclaimer = disclaimer.replace("{artist_name}", artist_name)
            disclaimer = disclaimer.replace('#{artist_name.replace(" ", "")}', f"#{artist_name.replace(' ', '')}")
            disclaimer = disclaimer.replace('#{base_name.replace(" ", "")}', f"#{base_name.replace(' ', '')}")

            # Ganti tagar baru
            disclaimer = disclaimer.replace("#song_title", song_title_hashtag)
            disclaimer = disclaimer.replace("#ori_artist", ori_artist_hashtag)

        else:
            logger.warning(f"Disclaimer template not found at '{disclaimer_template_path}'. Using a default disclaimer.")
            disclaimer = f"🎵 {base_name} - {artist_name}\n⚠️ Fan-made AI cover for hobby purposes.\n#AICover #RVC"

        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(disclaimer)

    except Exception as e:
        logger.error(f"Failed to create disclaimer file: {e}")
