import tkinter as tk
from tkinter import ttk

class ProgressPanel:
    def __init__(self, parent):
        # YouTube progress
        tk.Label(parent, text="YouTube Progress:").grid(row=11, column=0, sticky="w")
        self.youtube_progress = ttk.Progressbar(parent, orient="horizontal", length=400, mode="determinate")
        self.youtube_progress.grid(row=11, column=1, columnspan=2)

        # Shorts progress
        tk.Label(parent, text="Shorts/TikTok Progress:").grid(row=12, column=0, sticky="w")
        self.shorts_progress = ttk.Progressbar(parent, orient="horizontal", length=400, mode="determinate")
        self.shorts_progress.grid(row=12, column=1, columnspan=2)

        # Total progress
        tk.Label(parent, text="Total Progress:").grid(row=13, column=0, sticky="w")
        self.progress = ttk.Progressbar(parent, orient="horizontal", length=400, mode="determinate")
        self.progress.grid(row=13, column=1, columnspan=2)

        # Elapsed time label
        self.time_label = tk.Label(parent, text="Waktu: 0 detik")
        self.time_label.grid(row=14, column=0, columnspan=3)

    def update_progress(self, percent, task_type, total_tasks, done_tasks):
        """
        Update progress bar values.
        :param percent: Current percent (0-100) for the task
        :param task_type: "youtube" or "shorts"
        :param total_tasks: Total number of export tasks
        :param done_tasks: Number of completed tasks
        """
        if task_type == "youtube":
            self.youtube_progress["value"] = percent
        elif task_type == "shorts":
            self.shorts_progress["value"] = percent

        total_percent = (done_tasks / max(1, total_tasks) * 100) + (percent / max(1, total_tasks))
        self.progress["value"] = total_percent

    def update_time(self, text):
        """Update the elapsed time label."""
        self.time_label.config(text=text)
