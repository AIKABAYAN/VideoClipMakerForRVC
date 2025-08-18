import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

class ProgressPanel:
    def __init__(self, parent):
        self.frame = tk.LabelFrame(parent, text="Progress", padx=5, pady=5)

        # Progress YouTube
        tk.Label(self.frame, text="YouTube:").grid(row=0, column=0, sticky="w")
        self.progress_youtube = ttk.Progressbar(self.frame, length=300)
        self.progress_youtube.grid(row=0, column=1, padx=5, pady=2)
        self.youtube_status = tk.Label(self.frame, text="0.0%")
        self.youtube_status.grid(row=0, column=2, sticky="w", padx=5)
        self.youtube_stage = tk.Label(self.frame, text="⚙️ Idle")
        self.youtube_stage.grid(row=0, column=3, sticky="w", padx=(10, 0))

        # Progress Shorts
        tk.Label(self.frame, text="Shorts:").grid(row=1, column=0, sticky="w")
        self.progress_shorts = ttk.Progressbar(self.frame, length=300)
        self.progress_shorts.grid(row=1, column=1, padx=5, pady=2)
        self.shorts_status = tk.Label(self.frame, text="0.0%")
        self.shorts_status.grid(row=1, column=2, sticky="w", padx=5)
        self.shorts_stage = tk.Label(self.frame, text="⚙️ Idle")
        self.shorts_stage.grid(row=1, column=3, sticky="w", padx=(10, 0))

        # ETA / Waktu
        self.time_label = tk.Label(self.frame, text="Waktu: 0 detik")
        self.time_label.grid(row=2, column=0, columnspan=4, sticky="w", pady=(5, 0))

    def update_progress(self, percent, stage):
        """Memperbarui bilah kemajuan, persentase, dan tahap render."""
        percent = max(0, min(100, percent))
        
        task_labels = {
            "preprocess": "⚙️ Preprocessing",
            "frames": "🖼️ Building Frames",
            "visualizer": "🎵 Rendering Visualizer",
            "encode": "🎞️ Encoding Video",
            "done": "✅ Done",
            "idle": "⚙️ Idle",
            "error": "❌ Error",
        }
        
        stage_key = stage.split('_')[-1]
        label = task_labels.get(stage_key, f"🔹 {stage_key.capitalize()}")

        if stage.startswith("youtube"):
            self.progress_youtube["value"] = percent
            self.youtube_status.config(text=f"{percent:.1f}%")
            self.youtube_stage.config(text=label)
        elif stage.startswith("shorts"):
            self.progress_shorts["value"] = percent
            self.shorts_status.config(text=f"{percent:.1f}%")
            self.shorts_stage.config(text=label)

    def update_time(self, text):
        self.time_label.config(text=text)
