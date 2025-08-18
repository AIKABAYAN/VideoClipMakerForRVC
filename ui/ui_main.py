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
        # Configure grid to allow expansion
        self.root.grid_columnconfigure(1, weight=1)

        self._create_main_widgets()

        # Panels
        self.animation_panel = AnimationPanel(self.root, self.start_process)
        self.progress_panel = ProgressPanel(self.root)

        # Progress di bawah tombol Start
        self.progress_panel.frame.grid(row=16, column=0, columnspan=4, pady=10, sticky="ew")

        self.is_running = False
        self.start_time = None
        
        self.progress_vars = {
            "youtube": tk.DoubleVar(),
            "shorts": tk.DoubleVar()
        }

    def _create_main_widgets(self):
        # --- Row 0: Render Mode ---
        tk.Label(self.root, text="Render Mode:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        mode_frame = tk.Frame(self.root)
        mode_frame.grid(row=0, column=1, columnspan=2, sticky="w")
        self.render_mode = tk.StringVar(value="single")
        tk.Radiobutton(mode_frame, text="Single File", variable=self.render_mode, value="single",
                       command=self.toggle_mode_fields).pack(side="left")
        tk.Radiobutton(mode_frame, text="Batch Folder", variable=self.render_mode, value="batch",
                       command=self.toggle_mode_fields).pack(side="left", padx=10)

        # --- Row 1-7: File Paths and Metadata ---
        self._create_path_entry("MP3 File:", 1, self.browse_mp3)
        self._create_path_entry("MP3 Folder:", 2, self.browse_mp3_folder)
        self._create_path_entry("Background Folder:", 4, self.browse_bg)
        self._create_path_entry("Output Folder:", 5, self.browse_output)
        self._create_path_entry("Cover Image:", 7, self.browse_cover)

        self.mp3_path.insert(0, DEFAULT_MP3)
        self.mp3_folder.insert(0, DEFAULT_BATCH_FOLDER)
        self.bg_folder.insert(0, DEFAULT_BG_FOLDER)
        self.output_folder.insert(0, DEFAULT_OUTPUT)
        self.cover_path.insert(0, DEFAULT_COVER)

        tk.Label(self.root, text="Artist Name:").grid(row=6, column=0, sticky="w", padx=5, pady=2)
        self.artist_name = tk.Entry(self.root)
        self.artist_name.grid(row=6, column=1, columnspan=2, sticky="ew", padx=5)
        self.artist_name.insert(0, DEFAULT_ARTIST)
        
        # --- Row 3: Add Background Checkbox ---
        self.add_background = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="Add Background", variable=self.add_background,
                       command=self.toggle_background).grid(row=3, column=0, sticky="w", padx=5)

        # --- Row 8-9: Video Settings ---
        tk.Label(self.root, text="Background Mode:").grid(row=8, column=0, sticky="w", padx=5, pady=2)
        self.bg_mode = ttk.Combobox(self.root, values=["Black", "Blur", "Darken"], width=10, state="readonly")
        self.bg_mode.grid(row=8, column=1, sticky="w", padx=5)
        self.bg_mode.set(DEFAULT_BG_MODE)

        tk.Label(self.root, text="Blur/Darken Strength:").grid(row=9, column=0, sticky="w", padx=5, pady=2)
        self.blur_level = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.blur_level.grid(row=9, column=1, columnspan=2, sticky="ew", padx=5)
        self.blur_level.set(DEFAULT_BLUR_LEVEL)

        # --- Row 10-12: Export Options ---
        tk.Label(self.root, text="Export Format:", font=("", 10, "bold")).grid(row=10, column=0, sticky="w", padx=5, pady=(10,2))

        # YouTube Export
        self.export_youtube = tk.BooleanVar(value=True)
        tk.Checkbutton(self.root, text="YouTube (16:9)", variable=self.export_youtube).grid(row=11, column=0, sticky="w", padx=5)
        self.youtube_res_select = ttk.Combobox(self.root, values=list(YOUTUBE_RESOLUTIONS.keys()), width=15, state="readonly")
        self.youtube_res_select.grid(row=11, column=1, sticky="w", padx=5)
        self.youtube_res_select.set("HD (1280x720)")

        # Shorts Export
        self.export_shorts = tk.BooleanVar(value=False)
        tk.Checkbutton(self.root, text="Shorts/TikTok (9:16)", variable=self.export_shorts).grid(row=12, column=0, sticky="w", padx=5)
        
        # Visualizer Options (Cleaned up layout)
        visualizer_frame = tk.Frame(self.root)
        visualizer_frame.grid(row=12, column=1, columnspan=2, sticky="w", padx=5)

        self.include_visualizer = tk.BooleanVar(value=True)
        tk.Checkbutton(visualizer_frame, text="Include Audio Visualizer", variable=self.include_visualizer).pack(side="left")

        tk.Label(visualizer_frame, text="Height:").pack(side="left", padx=(10, 2))
        self.visualizer_height_var = tk.StringVar(value=DEFAULT_VISUALIZER_HEIGHT)
        self.visualizer_height_select = ttk.Combobox(
            visualizer_frame,
            values=["90%", "80%", "60%", "40%", "20%"],
            textvariable=self.visualizer_height_var,
            width=5,
            state="readonly"
        )
        self.visualizer_height_select.pack(side="left")

        self.toggle_mode_fields()

    def _create_path_entry(self, label_text, row, command):
        """Helper to create a Label, Entry, and Button row for file/folder paths."""
        tk.Label(self.root, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        
        entry = tk.Entry(self.root)
        entry.grid(row=row, column=1, sticky="ew", padx=5)
        
        button = tk.Button(self.root, text="Browse", command=command)
        button.grid(row=row, column=2, sticky="e", padx=5)

        # Store references to the created widgets
        if "MP3 File" in label_text: self.mp3_path = entry
        elif "MP3 Folder" in label_text: self.mp3_folder, self.mp3_folder_btn = entry, button
        elif "Background Folder" in label_text: self.bg_folder, self.bg_browse_btn = entry, button
        elif "Output Folder" in label_text: self.output_folder = entry
        elif "Cover Image" in label_text: self.cover_path = entry
        
        return entry, button

    def toggle_background(self):
        state = "normal" if self.add_background.get() else "disabled"
        self.bg_folder.config(state=state)
        self.bg_browse_btn.config(state=state)

    def toggle_mode_fields(self):
        is_single = self.render_mode.get() == "single"
        self.mp3_path.config(state="normal" if is_single else "disabled")
        self.mp3_folder.config(state="disabled" if is_single else "normal")
        self.mp3_folder_btn.config(state="disabled" if is_single else "normal")

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
        self.animation_panel.start_button.config(state="disabled")
        
        self.progress_vars["youtube"].set(0)
        self.progress_vars["shorts"].set(0)
        self.progress_panel.update_progress(0, "youtube_idle")
        self.progress_panel.update_progress(0, "shorts_idle")
        self.progress_panel.update_time("Waktu: 0 detik")
        self.progress_panel.update_file_header("Preparing to process...")

        if self.render_mode.get() == "single":
            if not os.path.isfile(self.mp3_path.get()):
                messagebox.showerror("Error", "MP3 file tidak ditemukan.")
                self.animation_panel.start_button.config(state="normal")
                return
        else:
            if not os.path.isdir(self.mp3_folder.get()):
                messagebox.showerror("Error", "Folder MP3 tidak ditemukan.")
                self.animation_panel.start_button.config(state="normal")
                return

        if self.add_background.get() and not os.path.isdir(self.bg_folder.get()):
            messagebox.showerror("Error", "Folder background tidak ditemukan.")
            self.animation_panel.start_button.config(state="normal")
            return

        if not os.path.isdir(self.output_folder.get()):
            try:
                os.makedirs(self.output_folder.get(), exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuat output folder:\n{e}")
                self.animation_panel.start_button.config(state="normal")
                return

        self.start_time = time.time()
        self.is_running = True
        self.update_elapsed_time()
        threading.Thread(target=self.run_builder, daemon=True).start()

    def update_elapsed_time(self):
        if self.is_running:
            elapsed = int(time.time() - self.start_time)
            avg_progress = (self.progress_vars["youtube"].get() + self.progress_vars["shorts"].get()) / 2
            
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
        total_mp3s = 0
        try:
            tasks = []
            if self.render_mode.get() == "single":
                tasks.append(self.mp3_path.get())
            else:
                tasks = [os.path.join(self.mp3_folder.get(), f) for f in os.listdir(self.mp3_folder.get()) if f.lower().endswith(".mp3")]

            total_mp3s = len(tasks)
            if total_mp3s == 0:
                self.progress_panel.update_file_header("No MP3 files found to process.")
                return

            for i, mp3_file in enumerate(tasks):
                file_name = os.path.basename(mp3_file)
                header_text = f"[{i + 1} of {total_mp3s}] - {file_name}"
                self.progress_panel.update_file_header(header_text)
                
                logger.info(f"--- Memproses file {i+1} dari {total_mp3s}: {file_name} ---")
                
                output_safe_name = sanitize_filename(os.path.splitext(file_name)[0])
                artist = self.artist_name.get()
                dynamic_scrolling_text = f"[{output_safe_name}] - [{artist}]"
                
                settings = {
                    "mp3_file": mp3_file,
                    "bg_folder": self.bg_folder.get() if self.add_background.get() else None,
                    "output_folder": self.output_folder.get(),
                    "song_name": output_safe_name,
                    "artist_name": artist,
                    "cover_image": self.cover_path.get(),
                    "bg_mode": self.bg_mode.get(),
                    "blur_level": self.blur_level.get(),
                    "export_youtube": self.export_youtube.get(),
                    "export_shorts": self.export_shorts.get(),
                    "resolution": YOUTUBE_RESOLUTIONS[self.youtube_res_select.get()],
                    "include_visualizer": self.include_visualizer.get(),
                    "visualizer_height": self.visualizer_height_var.get(),
                    "intro_duration": 2,
                    "scrolling_text": dynamic_scrolling_text,
                    "animations": self.animation_panel.get_selected()
                }

                try:
                    build_video_multithread(settings, progress_callback=self.update_progress)
                except Exception as e:
                    logger.error(f"Error processing {mp3_file}: {e}")
                    messagebox.showerror("Error", f"Gagal memproses {mp3_file}:\n{e}")

        finally:
            self.is_running = False
            self.animation_panel.start_button.config(state="normal")
            if total_mp3s > 0:
                self.progress_panel.update_file_header(f"Finished processing {total_mp3s} file(s).")
            logger.info("--- Semua proses selesai! ---")

    @staticmethod
    def format_time(seconds):
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        return f"{m} menit {s} detik" if m else f"{s} detik"

    def update_progress(self, percent, stage):
        task_key = "youtube" if "youtube" in stage else "shorts"
        self.progress_vars[task_key].set(percent)
        self.progress_panel.update_progress(percent, stage)
        self.root.update_idletasks()
