import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
import time

from core import build_video_multithread
from core.logger import get_logger # Menggunakan get_logger langsung
from .ui_constants import (
    DEFAULT_MP3, DEFAULT_BG_FOLDER, DEFAULT_OUTPUT, DEFAULT_SONG, DEFAULT_ARTIST, DEFAULT_BATCH_FOLDER,
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

        # Start Button
        self.start_button = tk.Button(root, text="Start", command=self.start_process)
        self.start_button.grid(row=16, column=1, pady=10)

        # Progress di bawah tombol Start
        self.progress_panel.frame.grid(row=17, column=0, columnspan=3, pady=10, sticky="ew")

        self.is_running = False
        self.start_time = None
        
        self.progress_vars = {
            "youtube": tk.DoubleVar(),
            "shorts": tk.DoubleVar()
        }

    def _create_main_widgets(self):
        # Render Mode (paling atas)
        tk.Label(self.root, text="Render Mode:").grid(row=0, column=0, sticky="w")
        self.render_mode = tk.StringVar(value="single")
        tk.Radiobutton(self.root, text="Single File", variable=self.render_mode, value="single",
                       command=self.toggle_mode_fields).grid(row=0, column=1, sticky="w")
        tk.Radiobutton(self.root, text="Batch Folder", variable=self.render_mode, value="batch",
                       command=self.toggle_mode_fields).grid(row=0, column=2, sticky="w")

        # MP3 File
        tk.Label(self.root, text="MP3 File:").grid(row=1, column=0, sticky="w")
        self.mp3_path = tk.Entry(self.root, width=50)
        self.mp3_path.grid(row=1, column=1)
        self.mp3_path.insert(0, DEFAULT_MP3)
        tk.Button(self.root, text="Browse", command=self.browse_mp3).grid(row=1, column=2)

        # MP3 Folder (Batch)
        tk.Label(self.root, text="MP3 Folder:").grid(row=2, column=0, sticky="w")
        self.mp3_folder = tk.Entry(self.root, width=50)
        self.mp3_folder.grid(row=2, column=1)
        self.mp3_folder.insert(0, DEFAULT_BATCH_FOLDER)
        self.mp3_folder_btn = tk.Button(self.root, text="Browse", command=self.browse_mp3_folder)
        self.mp3_folder_btn.grid(row=2, column=2)

        # Add Background
        self.add_background = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Add Background", variable=self.add_background,
                       command=self.toggle_background).grid(row=3, column=0, sticky="w")

        # Background Folder
        tk.Label(self.root, text="Background Folder:").grid(row=4, column=0, sticky="w")
        self.bg_folder = tk.Entry(self.root, width=50)
        self.bg_folder.grid(row=4, column=1)
        self.bg_folder.insert(0, DEFAULT_BG_FOLDER)
        self.bg_browse_btn = tk.Button(self.root, text="Browse", command=self.browse_bg)
        self.bg_browse_btn.grid(row=4, column=2)

        # Output Folder
        tk.Label(self.root, text="Output Folder:").grid(row=5, column=0, sticky="w")
        self.output_folder = tk.Entry(self.root, width=50)
        self.output_folder.grid(row=5, column=1)
        self.output_folder.insert(0, DEFAULT_OUTPUT)
        tk.Button(self.root, text="Browse", command=self.browse_output).grid(row=5, column=2)

        # Artist Name
        tk.Label(self.root, text="Artist Name:").grid(row=6, column=0, sticky="w")
        self.artist_name = tk.Entry(self.root, width=50)
        self.artist_name.grid(row=6, column=1)
        self.artist_name.insert(0, DEFAULT_ARTIST)

        # Cover Image
        tk.Label(self.root, text="Cover Image:").grid(row=7, column=0, sticky="w")
        self.cover_path = tk.Entry(self.root, width=50)
        self.cover_path.grid(row=7, column=1)
        self.cover_path.insert(0, DEFAULT_COVER)
        tk.Button(self.root, text="Browse", command=self.browse_cover).grid(row=7, column=2)

        # Scrolling Text Overlay
        tk.Label(self.root, text="Scrolling Text:").grid(row=8, column=0, sticky="w")
        self.scrolling_text = tk.Entry(self.root, width=50)
        self.scrolling_text.grid(row=8, column=1)
        self.scrolling_text.insert(0, "RVC by Sharkoded")

        # Video Settings
        tk.Label(self.root, text="Background Mode:").grid(row=9, column=0, sticky="w")
        self.bg_mode = ttk.Combobox(self.root, values=["Black", "Blur", "Darken"], width=10, state="readonly")
        self.bg_mode.grid(row=9, column=1, sticky="w")
        self.bg_mode.set(DEFAULT_BG_MODE)

        tk.Label(self.root, text="Blur/Darken Strength:").grid(row=10, column=0, sticky="w")
        self.blur_level = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.blur_level.grid(row=10, column=1, sticky="w")
        self.blur_level.set(DEFAULT_BLUR_LEVEL)

        # Export Options
        tk.Label(self.root, text="Export Format:").grid(row=11, column=0, sticky="w")
        self.export_youtube = tk.BooleanVar(value=True)
        self.export_shorts = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="YouTube", variable=self.export_youtube).grid(row=11, column=1, sticky="w")

        self.youtube_res_select = ttk.Combobox(self.root, values=list(YOUTUBE_RESOLUTIONS.keys()), width=15,
                                               state="readonly")
        self.youtube_res_select.grid(row=11, column=2, sticky="w")
        self.youtube_res_select.set("HD (1280x720)")

        tk.Checkbutton(self.root, text="Shorts/TikTok (9:16)", variable=self.export_shorts).grid(row=12, column=1,
                                                                                                 sticky="w")

        # Include visualizer
        self.include_visualizer = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Include Audio Visualizer", variable=self.include_visualizer).grid(row=12,
                                                                                                         column=2,
                                                                                                         sticky="w")

        # Visualizer Height
        tk.Label(self.root, text="Visualizer Height:").grid(row=12, column=3, sticky="w")
        self.visualizer_height_var = tk.StringVar(value=DEFAULT_VISUALIZER_HEIGHT)
        self.visualizer_height_select = ttk.Combobox(
            self.root,
            values=["90%", "80%", "60%", "40%", "20%"],
            textvariable=self.visualizer_height_var,
            width=5,
            state="readonly"
        )
        self.visualizer_height_select.grid(row=12, column=4, sticky="w")

        self.toggle_mode_fields()


    def toggle_background(self):
        state = "normal" if self.add_background.get() else "disabled"
        self.bg_folder.config(state=state)
        self.bg_browse_btn.config(state=state)

    def toggle_mode_fields(self):
        if self.render_mode.get() == "single":
            self.mp3_path.config(state="normal")
            self.mp3_folder.config(state="disabled")
            self.mp3_folder_btn.config(state="disabled")
        else:
            self.mp3_path.config(state="disabled")
            self.mp3_folder.config(state="normal")
            self.mp3_folder_btn.config(state="normal")

    def browse_mp3(self):
        path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
        if path:
            self.mp3_path.delete(0, tk.END)
            self.mp3_path.insert(0, path)

    def browse_mp3_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.mp3_folder.delete(0, tk.END)
            self.mp3_folder.insert(0, path)

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
        self.start_button.config(state="disabled")
        
        self.progress_vars["youtube"].set(0)
        self.progress_vars["shorts"].set(0)
        self.progress_panel.update_progress(0, "youtube_idle")
        self.progress_panel.update_progress(0, "shorts_idle")
        self.progress_panel.update_time("Waktu: 0 detik")

        if self.render_mode.get() == "single":
            if not os.path.isfile(self.mp3_path.get()):
                messagebox.showerror("Error", "MP3 file tidak ditemukan.")
                self.start_button.config(state="normal")
                return
        else:
            if not os.path.isdir(self.mp3_folder.get()):
                messagebox.showerror("Error", "Folder MP3 tidak ditemukan.")
                self.start_button.config(state="normal")
                return

        if self.add_background.get():
            if not os.path.isdir(self.bg_folder.get()):
                messagebox.showerror("Error", "Folder background tidak ditemukan.")
                self.start_button.config(state="normal")
                return

        if not os.path.isdir(self.output_folder.get()):
            try:
                os.makedirs(self.output_folder.get(), exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuat output folder:\n{e}")
                self.start_button.config(state="normal")
                return

        self.start_time = time.time()
        self.is_running = True
        self.update_elapsed_time()
        threading.Thread(target=self.run_builder, daemon=True).start()

    def update_elapsed_time(self):
        if self.is_running:
            elapsed = int(time.time() - self.start_time)
            avg_progress = sum(var.get() for var in self.progress_vars.values()) / len(self.progress_vars)
            
            eta_text = ""
            if avg_progress > 1:
                total_estimated_time = elapsed / (avg_progress / 100)
                remaining = max(0, total_estimated_time - elapsed)
                eta_text = f" | Perkiraan sisa: {self.format_time(remaining)}"
                
            self.progress_panel.update_time(
                f"Total Waktu Berjalan: {self.format_time(elapsed)}{eta_text}"
            )
            self.root.after(1000, self.update_elapsed_time)

    def run_builder(self):
        try:
            tasks = []
            if self.render_mode.get() == "single":
                tasks.append(self.mp3_path.get())
            else:
                for file in os.listdir(self.mp3_folder.get()):
                    if file.lower().endswith(".mp3"):
                        tasks.append(os.path.join(self.mp3_folder.get(), file))

            total_mp3s = len(tasks)
            for i, mp3_file in enumerate(tasks):
                logger.info(f"--- Memproses file {i+1} dari {total_mp3s}: {os.path.basename(mp3_file)} ---")
                
                output_safe_name = sanitize_filename(os.path.splitext(os.path.basename(mp3_file))[0])
                
                settings = {
                    "mp3_file": mp3_file,
                    "bg_folder": self.bg_folder.get() if self.add_background.get() else None,
                    "output_folder": self.output_folder.get(),
                    "song_name": output_safe_name,
                    "artist_name": self.artist_name.get(),
                    "cover_image": self.cover_path.get(),
                    "bg_mode": self.bg_mode.get(),
                    "blur_level": self.blur_level.get(),
                    "export_youtube": self.export_youtube.get(),
                    "export_shorts": self.export_shorts.get(),
                    "resolution": YOUTUBE_RESOLUTIONS[self.youtube_res_select.get()],
                    "include_visualizer": self.include_visualizer.get(),
                    "visualizer_height": self.visualizer_height_var.get(),
                    "intro_duration": 2,
                    "scrolling_text": self.scrolling_text.get()
                }

                try:
                    build_video_multithread(settings, progress_callback=self.update_progress)
                except Exception as e:
                    logger.error(f"Error processing {mp3_file}: {e}")
                    messagebox.showerror("Error", f"Gagal memproses {mp3_file}:\n{e}")

        finally:
            self.is_running = False
            self.start_button.config(state="normal")
            logger.info("--- Semua proses selesai! ---")

    @staticmethod
    def format_time(seconds):
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        return f"{m} menit {s} detik" if m else f"{s} detik"

    def update_progress(self, percent, stage):
        """Callback untuk memperbarui progress bar dari thread lain."""
        task_key = "youtube"
        if "shorts" in stage:
            task_key = "shorts"
            
        self.progress_vars[task_key].set(percent)
        self.progress_panel.update_progress(percent, stage)
        self.root.update_idletasks()
