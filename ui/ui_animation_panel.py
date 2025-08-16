import tkinter as tk

class AnimationPanel:
    def __init__(self, parent):
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

        anim_frame = tk.LabelFrame(parent, text="Animation Effects", padx=5, pady=5)
        anim_frame.grid(row=15, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Create columns for checkboxes
        col_frames = [tk.Frame(anim_frame) for _ in range(3)]
        for idx, frame in enumerate(col_frames):
            frame.grid(row=0, column=idx, sticky="nw")

        # Add checkboxes to columns
        for i, (anim_name, var) in enumerate(self.anim_vars.items()):
            cb = tk.Checkbutton(col_frames[i // 10], text=anim_name, variable=var)
            cb.pack(anchor="w")

        # Buttons for quick selection
        btn_frame = tk.Frame(anim_frame)
        btn_frame.grid(row=1, column=0, columnspan=3, pady=(5, 0))
        tk.Button(btn_frame, text="Select All", command=self.select_all).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Select None", command=self.deselect_all).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Basic Only", command=self.select_basic).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Advanced Only", command=self.select_advanced).pack(side="left", padx=5)

    def select_all(self):
        for var in self.anim_vars.values():
            var.set(True)

    def deselect_all(self):
        for var in self.anim_vars.values():
            var.set(False)

    def select_basic(self):
        basic_anims = [
            "zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down",
            "rotate-ccw", "rotate-cw", "fade-in", "fade-out"
        ]
        for anim_name, var in self.anim_vars.items():
            var.set(anim_name in basic_anims)

    def select_advanced(self):
        basic_anims = [
            "zoom-in", "zoom-out", "pan-left", "pan-right", "pan-up", "pan-down",
            "rotate-ccw", "rotate-cw", "fade-in", "fade-out"
        ]
        for anim_name, var in self.anim_vars.items():
            var.set(anim_name not in basic_anims)

    def get_selected(self):
        """Return a list of selected animation names."""
        return [anim for anim, var in self.anim_vars.items() if var.get()]
