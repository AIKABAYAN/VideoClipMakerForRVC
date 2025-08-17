import tkinter as tk
from tkinter import ttk


class ProgressPanel:
    def __init__(self, parent):
        # Bungkus semua di frame supaya gampang diposisikan
        self.frame = tk.LabelFrame(parent, text="Progress", padx=5, pady=5)

        # Progress YouTube
        tk.Label(self.frame, text="YouTube:").grid(row=0, column=0, sticky="w")
        self.progress_youtube = ttk.Progressbar(self.frame, length=300)
        self.progress_youtube.grid(row=0, column=1, padx=5, pady=2)
        self.progress_youtube["value"] = 0
        self.youtube_status = tk.Label(self.frame, text="0%")
        self.youtube_status.grid(row=0, column=2, sticky="w")
        self.youtube_stage = tk.Label(self.frame, text="⚙️ Idle")
        self.youtube_stage.grid(row=0, column=3, sticky="w", padx=(10, 0))

        # Progress Shorts
        tk.Label(self.frame, text="Shorts:").grid(row=1, column=0, sticky="w")
        self.progress_shorts = ttk.Progressbar(self.frame, length=300)
        self.progress_shorts.grid(row=1, column=1, padx=5, pady=2)
        self.progress_shorts["value"] = 0
        self.shorts_status = tk.Label(self.frame, text="0%")
        self.shorts_status.grid(row=1, column=2, sticky="w")
        self.shorts_stage = tk.Label(self.frame, text="⚙️ Idle")
        self.shorts_stage.grid(row=1, column=3, sticky="w", padx=(10, 0))

        # ETA / Waktu
        self.time_label = tk.Label(self.frame, text="Waktu: 0 detik")
        self.time_label.grid(row=2, column=0, columnspan=4, sticky="w", pady=(5, 0))

    def update_progress(self, percent, task_type, total_tasks=1, done_tasks=0):
        """Update progress bar, percent, dan tahap render"""

        percent = max(0, min(100, percent))  # clamp 0-100

        # mapping task_type jadi label & ikon
        task_labels = {
            "setup": "⚙️ Setup",
            "render": "🎨 Render",
            "merge": "🧩 Merge",
            "overlay": "✨ Overlay",
            "finalize": "✅ Finalize",
            "error": "❌ Error",
        }
        label = task_labels.get(task_type, f"🔹 {task_type.capitalize()}")

        # update ke YouTube
        if task_type.lower().startswith("youtube") or task_type in task_labels:
            self.progress_youtube["value"] = percent
            self.youtube_status.config(text=f"{percent:.1f}% ({done_tasks}/{total_tasks})")
            self.youtube_stage.config(text=label)

        # update ke Shorts
        if task_type.lower().startswith("shorts"):
            self.progress_shorts["value"] = percent
            self.shorts_status.config(text=f"{percent:.1f}% ({done_tasks}/{total_tasks})")
            self.shorts_stage.config(text=label)

    def update_time(self, text):
        self.time_label.config(text=text)
