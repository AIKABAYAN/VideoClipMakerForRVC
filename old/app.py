import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
import time
import re

from core.video_builder import build_video_multithread
from core.logger import get_logger

logger = get_logger("sikabayan")

# ===== Defaults =====
DEFAULT_MP3 = r"C:/Users/SIKABAYAN/Desktop/mini hati hati.MP3"
DEFAULT_BG_FOLDER = r"C:/Users/SIKABAYAN/Pictures/bg"
DEFAULT_OUTPUT = r"C:/Users/SIKABAYAN/Desktop/result"
DEFAULT_SONG = "Hati-hati di Jalan"
DEFAULT_ARTIST = "RVC by Sharkoded"
DEFAULT_COVER = r"C:/Users/SIKABAYAN/Pictures/photo_2025-07-13_00-41-59.jpg"
DEFAULT_BG_MODE = "Blur"
DEFAULT_BLUR_LEVEL = 3
DEFAULT_VISUALIZER_HEIGHT = "40%"

YOUTUBE_RESOLUTIONS = {
    "HD (1280x720)": (1280, 720),
    "2K (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160)
}

def sanitize_filename(name: str) -> str:
    """Remove illegal characters for filenames"""
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("SiKabayan Video Clip Maker")

        # MP3 File
        tk.Label(root, text="MP3 File:").grid(row=0, column=0, sticky="w")
        self.mp3_path = tk.Entry(root, width=50)
        self.mp3_path.grid(row=0, column=1)
        self.mp3_path.insert(0, DEFAULT_MP3)
        tk.Button(root, text="Browse", command=self.browse_mp3).grid(row=0, column=2)

        # Add Background checkbox
        self.add_background = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Add Background", variable=self.add_background, command=self.toggle_background).grid(row=1, column=0, sticky="w")

        # Background Folder
        tk.Label(root, text="Background Folder:").grid(row=2, column=0, sticky="w")
        self.bg_folder = tk.Entry(root, width=50)
        self.bg_folder.grid(row=2, column=1)
        self.bg_folder.insert(0, DEFAULT_BG_FOLDER)
        self.bg_browse_btn = tk.Button(root, text="Browse", command=self.browse_bg)
        self.bg_browse_btn.grid(row=2, column=2)

        # Output Folder
        tk.Label(root, text="Output Folder:").grid(row=3, column=0, sticky="w")
        self.output_folder = tk.Entry(root, width=50)
        self.output_folder.grid(row=3, column=1)
        self.output_folder.insert(0, DEFAULT_OUTPUT)
        tk.Button(root, text="Browse", command=self.browse_output).grid(row=3, column=2)

        # Song Info
        tk.Label(root, text="Song Name:").grid(row=4, column=0, sticky="w")
        self.song_name = tk.Entry(root, width=50)
        self.song_name.grid(row=4, column=1)
        self.song_name.insert(0, DEFAULT_SONG)

        tk.Label(root, text="Artist Name:").grid(row=5, column=0, sticky="w")
        self.artist_name = tk.Entry(root, width=50)
        self.artist_name.grid(row=5, column=1)
        self.artist_name.insert(0, DEFAULT_ARTIST)

        # Cover Image
        tk.Label(root, text="Cover Image:").grid(row=6, column=0, sticky="w")
        self.cover_path = tk.Entry(root, width=50)
        self.cover_path.grid(row=6, column=1)
        self.cover_path.insert(0, DEFAULT_COVER)
        tk.Button(root, text="Browse", command=self.browse_cover).grid(row=6, column=2)

        # Video Settings
        tk.Label(root, text="Background Mode:").grid(row=7, column=0, sticky="w")
        self.bg_mode = ttk.Combobox(root, values=["Black", "Blur", "Darken"], width=10, state="readonly")
        self.bg_mode.grid(row=7, column=1, sticky="w")
        self.bg_mode.set(DEFAULT_BG_MODE)

        tk.Label(root, text="Blur/Darken Strength:").grid(row=8, column=0, sticky="w")
        self.blur_level = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL)
        self.blur_level.grid(row=8, column=1, sticky="w")
        self.blur_level.set(DEFAULT_BLUR_LEVEL)

        # Export Options
        tk.Label(root, text="Export Format:").grid(row=9, column=0, sticky="w")
        self.export_youtube = tk.BooleanVar(value=True)
        self.export_shorts = tk.BooleanVar(value=False)
        tk.Checkbutton(root, text="YouTube", variable=self.export_youtube).grid(row=9, column=1, sticky="w")

        self.youtube_res_select = ttk.Combobox(root, values=list(YOUTUBE_RESOLUTIONS.keys()), width=15, state="readonly")
        self.youtube_res_select.grid(row=9, column=2, sticky="w")
        self.youtube_res_select.set("HD (1280x720)")

        tk.Checkbutton(root, text="Shorts/TikTok (9:16)", variable=self.export_shorts).grid(row=10, column=1, sticky="w")

        # Include visualizer
        self.include_visualizer = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Include Audio Visualizer", variable=self.include_visualizer).grid(row=10, column=2, sticky="w")

        # NEW: Visualizer Height
        tk.Label(root, text="Visualizer Height:").grid(row=10, column=3, sticky="w")
        self.visualizer_height_var = tk.StringVar(value=DEFAULT_VISUALIZER_HEIGHT)
        self.visualizer_height_select = ttk.Combobox(
            root,
            values=["80%", "60%", "40%", "20%"],
            textvariable=self.visualizer_height_var,
            width=5,
            state="readonly"
        )
        self.visualizer_height_select.grid(row=10, column=4, sticky="w")

        # Progress Tracking
        tk.Label(root, text="YouTube Progress:").grid(row=11, column=0, sticky="w")
        self.youtube_progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.youtube_progress.grid(row=11, column=1, columnspan=2)

        tk.Label(root, text="Shorts/TikTok Progress:").grid(row=12, column=0, sticky="w")
        self.shorts_progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.shorts_progress.grid(row=12, column=1, columnspan=2)

        tk.Label(root, text="Total Progress:").grid(row=13, column=0, sticky="w")
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.grid(row=13, column=1, columnspan=2)

        self.time_label = tk.Label(root, text="Waktu: 0 detik")
        self.time_label.grid(row=14, column=0, columnspan=3)

        # Animation Effects Frame
        anim_frame = tk.LabelFrame(root, text="Animation Effects", padx=5, pady=5)
        anim_frame.grid(row=15, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        self.anim_vars = {
            "zoom-in": tk.BooleanVar(value=True),
            "zoom-out": tk.BooleanVar(value=True),
            "pan-left": tk.BooleanVar(value=True),
            "pan-right": tk.BooleanVar(value=True),
            "pan-up": tk.BooleanVar(value=True),
            "pan-down": tk.BooleanVar(value=True),
            "rotate-ccw": tk.BooleanVar(value=True),
            "rotate-cw": tk.BooleanVar(value=True),
            "fade-in": tk.BooleanVar(value=True),
            "fade-out": tk.BooleanVar(value=True),
            "bounce": tk.BooleanVar(value=True),
            "glow-pulse": tk.BooleanVar(value=True),
            "tilt-swing": tk.BooleanVar(value=True),
            "color-shift": tk.BooleanVar(value=True),
            "pixelate": tk.BooleanVar(value=True),
            "squeeze": tk.BooleanVar(value=True),
            "radial-blur": tk.BooleanVar(value=True),
            "shake": tk.BooleanVar(value=True),
            "parallax": tk.BooleanVar(value=True),
            "shadow-pulse": tk.BooleanVar(value=True),
            "mosaic": tk.BooleanVar(value=False),
            "lens-flare": tk.BooleanVar(value=False),
            "watercolor": tk.BooleanVar(value=False),
            "neon-edge": tk.BooleanVar(value=False),
            "glitch": tk.BooleanVar(value=False),
            "warp": tk.BooleanVar(value=False),
            "duotone": tk.BooleanVar(value=False),
            "texture": tk.BooleanVar(value=False),
            "circle-reveal": tk.BooleanVar(value=False),
            "time-freeze": tk.BooleanVar(value=False)
        }

        col_frames = [tk.Frame(anim_frame) for _ in range(3)]
        for idx, frame in enumerate(col_frames):
            frame.grid(row=0, column=idx, sticky="nw")
        for i, (anim_name, var) in enumerate(self.anim_vars.items()):
            cb = tk.Checkbutton(col_frames[i // 10], text=anim_name, variable=var)
            cb.pack(anchor="w")

        btn_frame = tk.Frame(anim_frame)
        btn_frame.grid(row=1, column=0, columnspan=3, pady=(5, 0))
        tk.Button(btn_frame, text="Select All", command=self.select_all_animations).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Select None", command=self.deselect_all_animations).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Basic Only", command=self.select_basic_animations).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Advanced Only", command=self.select_advanced_animations).pack(side="left", padx=5)

        tk.Button(root, text="Start", command=self.start_process).grid(row=16, column=1, pady=10)

        self.is_running = False
        self.start_time = None
        self.current_progress = 0
        self.current_task_name = ""

    def toggle_background(self):
        state = "normal" if self.add_background.get() else "disabled"
        self.bg_folder.config(state=state)
        self.bg_browse_btn.config(state=state)

    def select_all_animations(self):
        for var in self.anim_vars.values():
            var.set(True)

    def deselect_all_animations(self):
        for var in self.anim_vars.values():
            var.set(False)

    def select_basic_animations(self):
        basic_anims = [
            "zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down",
            "rotate-ccw", "rotate-cw", "fade-in", "fade-out"
        ]
        for anim_name, var in self.anim_vars.items():
            var.set(anim_name in basic_anims)

    def select_advanced_animations(self):
        basic_anims = [
            "zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down",
            "rotate-ccw", "rotate-cw", "fade-in", "fade-out"
        ]
        for anim_name, var in self.anim_vars.items():
            var.set(anim_name not in basic_anims)

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
        self.progress["value"] = 0
        self.youtube_progress["value"] = 0
        self.shorts_progress["value"] = 0
        self.time_label.config(text="Waktu: 0 detik")

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
            self.time_label.config(
                text=f"[{self.current_task_name}] Waktu: {self.format_time(elapsed)} sudah berlalu{eta_text}"
            )
            self.root.after(1000, self.update_elapsed_time)

    def run_builder(self):
        mp3 = self.mp3_path.get()
        add_background = self.add_background.get()
        bg_folder = self.bg_folder.get() if add_background else ""
        output_folder = self.output_folder.get()
        song_name = self.song_name.get().strip() or "Untitled"
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

        selected_anims = [anim for anim, var in self.anim_vars.items() if var.get()]
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
                    song_name=song_name, artist_name=artist_name,
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
                    song_name=song_name, artist_name=artist_name,
                    output_path=shorts_output_path,
                    resolution=(720, 1280),
                    bg_mode=bg_mode, blur_level=blur_level,
                    progress_callback=lambda p: self.update_progress(p, "shorts", total_tasks, done_tasks),
                    animations=selected_anims,
                    include_visualizer=include_visualizer,
                    visualizer_height=visualizer_height_fraction
                )
                done_tasks += 1

            total_elapsed = time.time() - self.start_time
            self.is_running = False
            self.time_label.config(text=f"Selesai dalam {self.format_time(total_elapsed)}")
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
        if task_type == "youtube":
            self.youtube_progress["value"] = self.current_progress
        elif task_type == "shorts":
            self.shorts_progress["value"] = self.current_progress
        total_percent = (done_tasks / max(1, total_tasks) * 100) + (self.current_progress / max(1, total_tasks))
        self.progress["value"] = total_percent
        self.root.update_idletasks()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
