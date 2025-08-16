import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
import time

from core import build_video_multithread, get_logger
from .ui_constants import (
    DEFAULT_MP3, DEFAULT_BG_FOLDER, DEFAULT_OUTPUT, DEFAULT_SONG, DEFAULT_ARTIST,
    DEFAULT_COVER, DEFAULT_BG_MODE, DEFAULT_BLUR_LEVEL, DEFAULT_VISUALIZER_HEIGHT,
    YOUTUBE_RESOLUTIONS, sanitize_filename
)
from .ui_animation_panel import AnimationPanel
from .ui_progress_panel import ProgressPanel

logger = get_logger("sikabayan")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("SiKabayan Video Clip Maker")

        self._create_main_widgets()

        # Panels
        self.animation_panel = AnimationPanel(self.root)
        self.progress_panel = ProgressPanel(self.root)

        tk.Button(root, text="Start", command=self.start_process).grid(row=16, column=1, pady=10)

        self.is_running = False
        self.start_time = None
        self.current_progress = 0
        self.current_task_name = ""

    def _create_main_widgets(self):
        # MP3 File
        tk.Label(self.root, text="MP3 File:").grid(row=0, column=0, sticky="w")
        self.mp3_path = tk.Entry(self.root, width=50)
        self.mp3_path.grid(row=0, column=1)
        self.mp3_path.insert(0, DEFAULT_MP3)
        tk.Button(self.root, text="Browse", command=self.browse_mp3).grid(row=0, column=2)

        # Add Background
        self.add_background = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Add Background", variable=self.add_background, command=self.toggle_background).grid(row=1, column=0, sticky="w")

        # Background Folder
        tk.Label(self.root, text="Background Folder:").grid(row=2, column=0, sticky="w")
        self.bg_folder = tk.Entry(self.root, width=50)
        self.bg_folder.grid(row=2, column=1)
        self.bg_folder.insert(0, DEFAULT_BG_FOLDER)
        self.bg_browse_btn = tk.Button(self.root, text="Browse", command=self.browse_bg)
        self.bg_browse_btn.grid(row=2, column=2)

        # Output Folder
        tk.Label(self.root, text="Output Folder:").grid(row=3, column=0, sticky="w")
        self.output_folder = tk.Entry(self.root, width=50)
        self.output_folder.grid(row=3, column=1)
        self.output_folder.insert(0, DEFAULT_OUTPUT)
        tk.Button(self.root, text="Browse", command=self.browse_output).grid(row=3, column=2)

        # Song Info
        # tk.Label(self.root, text="Song Name:").grid(row=4, column=0, sticky="w")
        # self.song_name = tk.Entry(self.root, width=50)
        # self.song_name.grid(row=4, column=1)
        # self.song_name.insert(0, DEFAULT_SONG)

        tk.Label(self.root, text="Artist Name:").grid(row=5, column=0, sticky="w")
        self.artist_name = tk.Entry(self.root, width=50)
        self.artist_name.grid(row=5, column=1)
        self.artist_name.insert(0, DEFAULT_ARTIST)

        # Cover Image
        tk.Label(self.root, text="Cover Image:").grid(row=6, column=0, sticky="w")
        self.cover_path = tk.Entry(self.root, width=50)
        self.cover_path.grid(row=6, column=1)
        self.cover_path.insert(0, DEFAULT_COVER)
        tk.Button(self.root, text="Browse", command=self.browse_cover).grid(row=6, column=2)

        # Video Settings
        tk.Label(self.root, text="Background Mode:").grid(row=7, column=0, sticky="w")
        self.bg_mode = ttk.Combobox(self.root, values=["Black", "Blur", "Darken"], width=10, state="readonly")
        self.bg_mode.grid(row=7, column=1, sticky="w")
        self.bg_mode.set(DEFAULT_BG_MODE)

        tk.Label(self.root, text="Blur/Darken Strength:").grid(row=8, column=0, sticky="w")
        self.blur_level = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.blur_level.grid(row=8, column=1, sticky="w")
        self.blur_level.set(DEFAULT_BLUR_LEVEL)

        # Export Options
        tk.Label(self.root, text="Export Format:").grid(row=9, column=0, sticky="w")
        self.export_youtube = tk.BooleanVar(value=True)
        self.export_shorts = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="YouTube", variable=self.export_youtube).grid(row=9, column=1, sticky="w")

        self.youtube_res_select = ttk.Combobox(self.root, values=list(YOUTUBE_RESOLUTIONS.keys()), width=15, state="readonly")
        self.youtube_res_select.grid(row=9, column=2, sticky="w")
        self.youtube_res_select.set("HD (1280x720)")

        tk.Checkbutton(self.root, text="Shorts/TikTok (9:16)", variable=self.export_shorts).grid(row=10, column=1, sticky="w")

        # Include visualizer
        self.include_visualizer = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Include Audio Visualizer", variable=self.include_visualizer).grid(row=10, column=2, sticky="w")

        # Visualizer Height
        tk.Label(self.root, text="Visualizer Height:").grid(row=10, column=3, sticky="w")
        self.visualizer_height_var = tk.StringVar(value=DEFAULT_VISUALIZER_HEIGHT)
        self.visualizer_height_select = ttk.Combobox(
            self.root,
            values=["90%","80%", "60%", "40%", "20%"],
            textvariable=self.visualizer_height_var,
            width=5,
            state="readonly"
        )
        self.visualizer_height_select.grid(row=10, column=4, sticky="w")

    def toggle_background(self):
        state = "normal" if self.add_background.get() else "disabled"
        self.bg_folder.config(state=state)
        self.bg_browse_btn.config(state=state)

    def browse_mp3(self):
        path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
        if path:
            self.mp3_path.delete(0, tk.END)
            self.mp3_path.insert(0, path)

    def browse_bg(self):
        path = filedialog.askdirectory()
        if path:
            self.bg_folder.delete(0, tk.END)
            self.bg_folder.insert(0, path)

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_folder.delete(0, tk.END)
            self.output_folder.insert(0, path)

    def browse_cover(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if path:
            self.cover_path.delete(0, tk.END)
            self.cover_path.insert(0, path)

    def start_process(self):
        self.progress_panel.update_progress(0, "youtube", 1, 0)
        self.progress_panel.update_progress(0, "shorts", 1, 0)
        self.progress_panel.update_time("Waktu: 0 detik")

        if not os.path.isfile(self.mp3_path.get()):
            messagebox.showerror("Error", "MP3 file tidak ditemukan.")
            return

        if self.add_background.get():
            if not os.path.isdir(self.bg_folder.get()):
                messagebox.showerror("Error", "Folder background tidak ditemukan.")
                return

        if not os.path.isdir(self.output_folder.get()):
            try:
                os.makedirs(self.output_folder.get(), exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuat output folder:\n{e}")
                return

        self.start_time = time.time()
        self.task_start_time = self.start_time
        self.is_running = True
        self.update_elapsed_time()
        threading.Thread(target=self.run_builder, daemon=True).start()

    def update_elapsed_time(self):
        if self.is_running:
            elapsed = int(time.time() - self.task_start_time)
            eta_text = ""
            if self.current_progress > 0:
                total_estimated_time = elapsed / (self.current_progress / 100)
                remaining = max(0, total_estimated_time - elapsed)
                eta_text = f" | Perkiraan sisa: {self.format_time(remaining)}"
            self.progress_panel.update_time(
                f"[{self.current_task_name}] Waktu: {self.format_time(elapsed)} sudah berlalu{eta_text}"
            )
            self.root.after(1000, self.update_elapsed_time)

    def run_builder(self):
        mp3 = self.mp3_path.get()
        add_background = self.add_background.get()
        bg_folder = self.bg_folder.get() if add_background else ""
        output_folder = self.output_folder.get()
        # song_name = self.song_name.get().strip() or "Untitled"
        artist_name = self.artist_name.get().strip() or "Unknown"
        cover_path = self.cover_path.get()
        bg_mode = self.bg_mode.get() if add_background else "Black"
        blur_level = self.blur_level.get()
        export_youtube = self.export_youtube.get()
        export_shorts = self.export_shorts.get()
        include_visualizer = self.include_visualizer.get()

        height_str = self.visualizer_height_var.get().strip("%")
        try:
            visualizer_height_fraction = float(height_str) / 100.0
        except ValueError:
            visualizer_height_fraction = 0.4

        selected_anims = self.animation_panel.get_selected()
        if not selected_anims:
            messagebox.showwarning("Warning", "No animations selected! Using default set.")
            selected_anims = None

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(mp3))[0]
        safe_artist = sanitize_filename(artist_name)
        total_tasks = int(bool(export_youtube)) + int(bool(export_shorts))
        done_tasks = 0

        try:
            if export_youtube:
                yt_res = YOUTUBE_RESOLUTIONS[self.youtube_res_select.get()]
                self.current_task_name = "YouTube"
                self.task_start_time = time.time()
                self.current_progress = 0
                yt_filename = f"{base_name} - {safe_artist} - YT.mp4"
                yt_output_path = os.path.join(output_folder, yt_filename)
                logger.info(f"Start YT: {yt_output_path}")
                build_video_multithread(
                    mp3_path=mp3, bg_folder=bg_folder, cover_path=cover_path,
                    song_name="", artist_name=artist_name,
                    output_path=yt_output_path,
                    resolution=yt_res,
                    bg_mode=bg_mode, blur_level=blur_level,
                    progress_callback=lambda p: self.update_progress(p, "youtube", total_tasks, done_tasks),
                    animations=selected_anims,
                    include_visualizer=include_visualizer,
                    visualizer_height=visualizer_height_fraction
                )
                done_tasks += 1

            if export_shorts:
                self.current_task_name = "Shorts/TikTok"
                self.task_start_time = time.time()
                self.current_progress = 0
                shorts_filename = f"{base_name} - {safe_artist} - Short.mp4"
                shorts_output_path = os.path.join(output_folder, shorts_filename)
                logger.info(f"Start Shorts: {shorts_output_path}")
                build_video_multithread(
                    mp3_path=mp3, bg_folder=bg_folder, cover_path=cover_path,
                    song_name="", artist_name=artist_name,
                    output_path=shorts_output_path,
                    resolution=(1080, 1920),
                    bg_mode=bg_mode, blur_level=blur_level,
                    progress_callback=lambda p: self.update_progress(p, "shorts", total_tasks, done_tasks),
                    animations=selected_anims,
                    include_visualizer=include_visualizer,
                    visualizer_height=visualizer_height_fraction
                )
                done_tasks += 1

            total_elapsed = time.time() - self.start_time
            self.is_running = False
            self.progress_panel.update_time(f"Selesai dalam {self.format_time(total_elapsed)}")
            messagebox.showinfo("Selesai", "Rendering selesai 🎉")

        except Exception as e:
            self.is_running = False
            logger.exception("Kesalahan saat rendering: %s", e)
            messagebox.showerror("Error", f"Terjadi kesalahan saat rendering:\n{e}")

    @staticmethod
    def format_time(seconds):
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        return f"{m} menit {s} detik" if m else f"{s} detik"

    def update_progress(self, percent, task_type, total_tasks, done_tasks):
        self.current_progress = max(0, min(100, percent))
        self.progress_panel.update_progress(self.current_progress, task_type, total_tasks, done_tasks)
        self.root.update_idletasks()
