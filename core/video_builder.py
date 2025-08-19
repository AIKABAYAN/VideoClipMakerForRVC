import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from core.utils import list_images
from core.logger import get_logger
from core.video_utils import (
    log_frame, check_ffmpeg, configure_imagemagick,
    MAX_WORKERS, BLUE, CYAN
)
from core.frame_pipeline import _build_single_output_hybrid

logger = get_logger("sikabayan")


def build_video_multithread(settings, progress_callback=None):
    """Public entry point that orchestrates concurrent video builds."""
    song_name = settings["song_name"]
    output_folder = settings["output_folder"]
    export_youtube = settings.get("export_youtube", True)
    export_shorts = settings.get("export_shorts", False)
    cover_image = settings.get("cover_image")

    try:
        res_parts = settings.get("resolution", (1280, 720))
        yt_resolution = (int(res_parts[0]), int(res_parts[1]))
    except Exception:
        yt_resolution = (1280, 720)
    shorts_resolution = tuple(settings.get("shorts_resolution", (1080, 1920)))

    check_ffmpeg()
    configure_imagemagick()

    images = list_images(settings.get("bg_folder")) if settings.get("bg_folder") and os.path.isdir(settings.get("bg_folder")) else []
    if cover_image and os.path.isfile(cover_image):
        images.insert(0, cover_image)
    if not images:
        images = ["__BLACK__"]

    os.makedirs(output_folder, exist_ok=True)

    tasks_to_run = []
    if export_youtube:
        yt_settings = settings.copy()
        yt_settings.update({"images": images,
                            "output_path": os.path.join(output_folder, f"{song_name}.mp4"),
                            "resolution": yt_resolution})
        tasks_to_run.append(("youtube", yt_settings))
    if export_shorts:
        shorts_settings = settings.copy()
        shorts_settings.update({"images": images,
                                "output_path": os.path.join(output_folder, f"{song_name}_Shorts.mp4"),
                                "resolution": shorts_resolution})
        tasks_to_run.append(("shorts", shorts_settings))

    overall_start_time = time.time()
    log_frame("🚀 Process Start", f"Started at: {datetime.fromtimestamp(overall_start_time).strftime('%Y/%m/%d %H:%M:%S')}", BLUE)

    all_summaries = []
    with ThreadPoolExecutor(max_workers=min(len(tasks_to_run), MAX_WORKERS) or 1) as executor:
        futures = {
            executor.submit(_build_single_output_hybrid, task_settings, tag, progress_callback): (tag, task_settings)
            for tag, task_settings in tasks_to_run
        }
        for future in as_completed(futures):
            tag, task_settings = futures[future]
            try:
                perf_summary = future.result()
                all_summaries.append((tag, task_settings["song_name"], perf_summary))
            except Exception as e:
                logger.error(f"Error building '{tag}' video: {e}", exc_info=False)
                if progress_callback:
                    progress_callback(0, f"{tag}_error")

    overall_end_time = time.time()
    log_frame("📊 Overall Summary", "All tasks completed.", CYAN)

    for tag, name, summary in sorted(all_summaries):
        total_task_time = sum(summary.values())
        tqdm.write(f"\n  --- Breakdown for {tag.capitalize()} [{name}] ({total_task_time:.1f}s) ---")
        for stage, duration in summary.items():
            tqdm.write(f"   - {stage:<12}: {duration:.1f}s")

    tqdm.write(f"\n\n  Start Time: {datetime.fromtimestamp(overall_start_time).strftime('%Y/%m/%d %H:%M:%S')}")
    tqdm.write(f"  End Time  : {datetime.fromtimestamp(overall_end_time).strftime('%Y/%m/%d %H:%M:%S')}")
    tqdm.write(f"  Total Duration: {overall_end_time - overall_start_time:.1f}s")
    tqdm.write("\nProcess done!!!")
