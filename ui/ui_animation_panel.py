import tkinter as tk

class AnimationPanel:
    def __init__(self, parent, start_command):
        # Default animation states
        self.anim_vars = {
            "zoom-in": tk.BooleanVar(value=False),
            "zoom-out": tk.BooleanVar(value=False),
            "pan-left": tk.BooleanVar(value=False),
            "pan-right": tk.BooleanVar(value=False),
            "pan-up": tk.BooleanVar(value=False),
            "pan-down": tk.BooleanVar(value=False),
            "rotate-ccw": tk.BooleanVar(value=True),  # ✅ default aktif
            "rotate-cw": tk.BooleanVar(value=True),   # ✅ default aktif
            "fade-in": tk.BooleanVar(value=False),
            "fade-out": tk.BooleanVar(value=False),
            "bounce": tk.BooleanVar(value=False),
            "glow-pulse": tk.BooleanVar(value=False),
            "tilt-swing": tk.BooleanVar(value=False),
            "color-shift": tk.BooleanVar(value=False),
            "pixelate": tk.BooleanVar(value=False),
            "squeeze": tk.BooleanVar(value=False),
            "radial-blur": tk.BooleanVar(value=False),
            "shake": tk.BooleanVar(value=True),     # ✅ default aktif
            "parallax": tk.BooleanVar(value=False),
            "shadow-pulse": tk.BooleanVar(value=False),
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

        anim_frame = tk.LabelFrame(parent, text="Animation Effects", padx=5, pady=5)
        anim_frame.grid(row=15, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        anim_frame.grid_columnconfigure(0, weight=1) # Make column expandable

        # Buat 5 kolom biar rapi
        checkbox_frame = tk.Frame(anim_frame)
        checkbox_frame.grid(row=0, column=0, columnspan=5, sticky="ew")

        col_frames = [tk.Frame(checkbox_frame) for _ in range(5)]
        for idx, frame in enumerate(col_frames):
            frame.pack(side="left", fill="x", expand=True, anchor="n")

        # Tambahkan checkbox ke masing-masing kolom
        for i, (anim_name, var) in enumerate(self.anim_vars.items()):
            cb = tk.Checkbutton(col_frames[i % 5], text=anim_name, variable=var)
            cb.pack(anchor="w")

        # Buttons untuk quick select
        btn_frame = tk.Frame(anim_frame)
        btn_frame.grid(row=1, column=0, columnspan=5, pady=(10, 0))
        tk.Button(btn_frame, text="Select All", command=self.select_all).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Select None", command=self.deselect_all).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Basic Only", command=self.select_basic).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Advanced Only", command=self.select_advanced).pack(side="left", padx=5)

        # Start Button now at the bottom of this panel
        self.start_button = tk.Button(anim_frame, text="Start", command=start_command)
        self.start_button.grid(row=2, column=0, columnspan=5, pady=(10, 5), sticky="ew")


    def select_all(self):
        for var in self.anim_vars.values():
            var.set(True)

    def deselect_all(self):
        for var in self.anim_vars.values():
            var.set(False)

    def select_basic(self):
        basic_anims = [
            "zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down",
            "rotate-ccw", "rotate-cw", "fade-in", "fade-out", "shake"
        ]
        for anim_name, var in self.anim_vars.items():
            var.set(anim_name in basic_anims)

    def select_advanced(self):
        basic_anims = [
            "zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down",
            "rotate-ccw", "rotate-cw", "fade-in", "fade-out", "shake"
        ]
        for anim_name, var in self.anim_vars.items():
            var.set(anim_name not in basic_anims)

    def get_selected(self):
        """Return a list of selected animation names."""
        return [anim for anim, var in self.anim_vars.items() if var.get()]
