import tkinter as tk

class AnimationPanel:
    def __init__(self, parent):
        # Default animation states (semua False dulu, nanti kita pilih default yg 3 aja)
        self.anim_vars = {
            "zoom-in": tk.BooleanVar(value=False),
            "zoom-out": tk.BooleanVar(value=False),
            "pan-left": tk.BooleanVar(value=False),
            "pan-right": tk.BooleanVar(value=False),
            "pan-up": tk.BooleanVar(value=False),
            "pan-down": tk.BooleanVar(value=False),
            "rotate-ccw": tk.BooleanVar(value=False),
            "rotate-cw": tk.BooleanVar(value=False),
            "fade-in": tk.BooleanVar(value=True),   # ✅ default aktif
            "fade-out": tk.BooleanVar(value=True),  # ✅ default aktif
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

        # Buat 5 kolom biar rapi
        col_frames = [tk.Frame(anim_frame) for _ in range(5)]
        for idx, frame in enumerate(col_frames):
            frame.grid(row=0, column=idx, sticky="nw")

        # Tambahkan checkbox ke masing-masing kolom
        for i, (anim_name, var) in enumerate(self.anim_vars.items()):
            cb = tk.Checkbutton(col_frames[i % 5], text=anim_name, variable=var)
            cb.pack(anchor="w")

        # Buttons untuk quick select
        btn_frame = tk.Frame(anim_frame)
        btn_frame.grid(row=1, column=0, columnspan=5, pady=(5, 0))
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
