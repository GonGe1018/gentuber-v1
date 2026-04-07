"""
settings_gui.py — Tkinter settings dialog shown before pipeline starts.

Loads/saves settings from settings.json. Returns updated Config when
the user clicks Start.
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent / "settings.json"

# Keys that map to config fields, with display info
SETTINGS_SCHEMA = [
    {
        "section": "Input",
        "fields": [
            {
                "key": "video_source",
                "label": "Source",
                "type": "entry",
                "default": "assets/test_input.mp4",
                "tooltip": "Video file path or webcam index (0, 1, ...)",
            },
            {
                "key": "reference_image",
                "label": "Reference Image",
                "type": "file",
                "default": "assets/reference.png",
                "tooltip": "Character reference image for IP-Adapter",
            },
        ],
    },
    {
        "section": "Engine",
        "fields": [
            {
                "key": "engine_backend",
                "label": "Backend",
                "type": "combo",
                "default": "ip_adapter",
                "options": [
                    "ip_adapter",
                    "lcm_graph",
                    "sdturbo_graph",
                    "sdturbo",
                    "t2i",
                    "controlnet",
                ],
            },
            {
                "key": "output_size",
                "label": "Resolution",
                "type": "combo",
                "default": "384",
                "options": ["256", "384", "512"],
            },
            {
                "key": "num_inference_steps",
                "label": "Steps",
                "type": "int_slider",
                "default": 4,
                "min": 1,
                "max": 12,
            },
            {
                "key": "seed",
                "label": "Seed",
                "type": "entry",
                "default": "42",
                "tooltip": "-1 = random",
            },
        ],
    },
    {
        "section": "IP-Adapter / ControlNet",
        "fields": [
            {
                "key": "ip_adapter_scale",
                "label": "IP-Adapter Scale",
                "type": "slider",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "resolution": 0.05,
            },
            {
                "key": "controlnet_conditioning_scale",
                "label": "ControlNet Scale",
                "type": "slider",
                "default": 1.5,
                "min": 0.5,
                "max": 3.0,
                "step": 0.1,
                "resolution": 0.1,
            },
            {
                "key": "guidance_scale",
                "label": "Guidance Scale",
                "type": "slider",
                "default": 1.0,
                "min": 1.0,
                "max": 3.0,
                "step": 0.1,
                "resolution": 0.1,
            },
            {
                "key": "temporal_feedback_strength",
                "label": "Feedback Strength",
                "type": "slider",
                "default": 0.3,
                "min": 0.1,
                "max": 1.0,
                "step": 0.05,
                "resolution": 0.05,
            },
        ],
    },
    {
        "section": "Display",
        "fields": [
            {
                "key": "show_skeleton_overlay",
                "label": "Show Skeleton",
                "type": "check",
                "default": True,
            },
            {"key": "show_fps", "label": "Show FPS", "type": "check", "default": True},
            {
                "key": "detect_hands",
                "label": "Detect Hands",
                "type": "check",
                "default": True,
            },
            {
                "key": "interp_alpha",
                "label": "Smoothing",
                "type": "slider",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "resolution": 0.05,
            },
        ],
    },
    {
        "section": "Output",
        "fields": [
            {
                "key": "output_file",
                "label": "Save to MP4",
                "type": "entry",
                "default": "",
                "tooltip": "Leave empty for GUI mode, or set path for headless recording",
            },
        ],
    },
]


def _load_settings() -> dict:
    """Load saved settings from JSON file."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_settings(data: dict) -> None:
    """Save settings to JSON file."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class SettingsGUI:
    """
    Tkinter settings dialog. Call .run() to show the window.
    Returns a dict of settings when Start is clicked, or None if cancelled.
    """

    def __init__(self):
        self.result: dict | None = None
        self._vars: dict[str, tk.Variable] = {}

    def run(self) -> dict | None:
        saved = _load_settings()

        root = tk.Tk()
        root.title("Realtime Live2D - Settings")
        root.resizable(False, False)

        # Dark theme colors
        bg = "#2b2b2b"
        fg = "#e0e0e0"
        accent = "#4a9eff"
        entry_bg = "#3c3c3c"
        section_fg = "#4a9eff"

        root.configure(bg=bg)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=bg, foreground=fg, fieldbackground=entry_bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 9))
        style.configure(
            "TButton",
            background=accent,
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
        )
        style.map("TButton", background=[("active", "#3a8aee")])
        style.configure(
            "TCheckbutton", background=bg, foreground=fg, font=("Segoe UI", 9)
        )
        style.configure("TCombobox", fieldbackground=entry_bg, foreground=fg)
        style.configure(
            "Section.TLabel",
            background=bg,
            foreground=section_fg,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TLabelframe", background=bg, foreground=section_fg)
        style.configure(
            "TLabelframe.Label",
            background=bg,
            foreground=section_fg,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TScale", background=bg, troughcolor=entry_bg)

        main_frame = ttk.Frame(root, padding=15)
        main_frame.pack(fill="both", expand=True)

        row = 0
        for section in SETTINGS_SCHEMA:
            # Section header
            lf = ttk.LabelFrame(main_frame, text=section["section"], padding=(10, 5))
            lf.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(5, 5))
            row += 1

            for i, field in enumerate(section["fields"]):
                key = field["key"]
                default = field["default"]
                val = saved.get(key, default)

                label = ttk.Label(lf, text=field["label"])
                label.grid(row=i, column=0, sticky="w", padx=(5, 10), pady=3)

                if field["type"] == "slider":
                    var = tk.DoubleVar(value=float(val))
                    self._vars[key] = var

                    val_label = ttk.Label(lf, text=f"{float(val):.2f}", width=5)
                    val_label.grid(row=i, column=2, padx=(5, 5))

                    def make_slider_cb(v, vl):
                        def cb(x):
                            vl.config(text=f"{v.get():.2f}")

                        return cb

                    scale = ttk.Scale(
                        lf,
                        from_=field["min"],
                        to=field["max"],
                        variable=var,
                        orient="horizontal",
                        length=200,
                        command=make_slider_cb(var, val_label),
                    )
                    scale.grid(row=i, column=1, sticky="ew", padx=5, pady=3)

                elif field["type"] == "int_slider":
                    var = tk.IntVar(value=int(val))
                    self._vars[key] = var

                    val_label = ttk.Label(lf, text=f"{int(val)}", width=5)
                    val_label.grid(row=i, column=2, padx=(5, 5))

                    def make_int_slider_cb(v, vl):
                        def cb(x):
                            v.set(round(float(x)))
                            vl.config(text=f"{v.get()}")

                        return cb

                    scale = ttk.Scale(
                        lf,
                        from_=field["min"],
                        to=field["max"],
                        variable=var,
                        orient="horizontal",
                        length=200,
                        command=make_int_slider_cb(var, val_label),
                    )
                    scale.grid(row=i, column=1, sticky="ew", padx=5, pady=3)

                elif field["type"] == "combo":
                    var = tk.StringVar(value=str(val))
                    self._vars[key] = var
                    combo = ttk.Combobox(
                        lf,
                        textvariable=var,
                        values=field["options"],
                        state="readonly",
                        width=20,
                    )
                    combo.grid(
                        row=i, column=1, columnspan=2, sticky="ew", padx=5, pady=3
                    )

                elif field["type"] == "check":
                    var = tk.BooleanVar(value=bool(val))
                    self._vars[key] = var
                    chk = ttk.Checkbutton(lf, variable=var)
                    chk.grid(row=i, column=1, sticky="w", padx=5, pady=3)

                elif field["type"] == "file":
                    var = tk.StringVar(value=str(val))
                    self._vars[key] = var
                    entry = ttk.Entry(lf, textvariable=var, width=25)
                    entry.grid(row=i, column=1, sticky="ew", padx=5, pady=3)

                    def make_browse(v):
                        def browse():
                            path = filedialog.askopenfilename(
                                filetypes=[
                                    ("Images", "*.png *.jpg *.jpeg *.bmp"),
                                    ("All", "*.*"),
                                ]
                            )
                            if path:
                                v.set(path)

                        return browse

                    btn = ttk.Button(lf, text="...", width=3, command=make_browse(var))
                    btn.grid(row=i, column=2, padx=(0, 5), pady=3)

                elif field["type"] == "entry":
                    var = tk.StringVar(value=str(val))
                    self._vars[key] = var
                    entry = ttk.Entry(lf, textvariable=var, width=28)
                    entry.grid(
                        row=i, column=1, columnspan=2, sticky="ew", padx=5, pady=3
                    )

                lf.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=(15, 5))

        def on_start():
            self.result = self._collect()
            _save_settings(self.result)
            root.destroy()

        def on_reset():
            for section in SETTINGS_SCHEMA:
                for field in section["fields"]:
                    key = field["key"]
                    if key in self._vars:
                        self._vars[key].set(field["default"])

        start_btn = ttk.Button(btn_frame, text="  Start  ", command=on_start)
        start_btn.pack(side="right", padx=5)

        reset_btn = ttk.Button(btn_frame, text="  Reset  ", command=on_reset)
        reset_btn.pack(side="right", padx=5)

        # Center window
        root.update_idletasks()
        w = root.winfo_width()
        h = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (w // 2)
        y = (root.winfo_screenheight() // 2) - (h // 2)
        root.geometry(f"+{x}+{y}")

        # Enter key starts
        root.bind("<Return>", lambda e: on_start())
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        root.mainloop()
        return self.result

    def _collect(self) -> dict:
        """Collect all variable values into a dict."""
        result = {}
        for section in SETTINGS_SCHEMA:
            for field in section["fields"]:
                key = field["key"]
                if key in self._vars:
                    val = self._vars[key].get()
                    # Type coercion
                    if field["type"] == "slider":
                        val = round(float(val), 3)
                    elif field["type"] == "int_slider":
                        val = int(round(float(val)))
                    elif field["type"] == "check":
                        val = bool(val)
                    result[key] = val
        return result


def apply_gui_settings(settings: dict, cfg) -> str | None:
    """
    Apply GUI settings dict to a Config object.
    Returns output_file path or None for GUI mode.
    """
    if "video_source" in settings:
        src = settings["video_source"]
        try:
            cfg.video_source = int(src)
        except ValueError:
            cfg.video_source = src

    if "reference_image" in settings:
        cfg.reference_image = settings["reference_image"]

    if "engine_backend" in settings:
        cfg.engine_backend = settings["engine_backend"]

    if "output_size" in settings:
        s = int(settings["output_size"])
        cfg.capture_width = cfg.capture_height = s
        cfg.output_width = cfg.output_height = s

    if "num_inference_steps" in settings:
        cfg.num_inference_steps = int(settings["num_inference_steps"])

    if "seed" in settings:
        try:
            cfg.seed = int(settings["seed"])
        except ValueError:
            cfg.seed = 42

    if "ip_adapter_scale" in settings:
        cfg.ip_adapter_scale = float(settings["ip_adapter_scale"])

    if "controlnet_conditioning_scale" in settings:
        cfg.controlnet_conditioning_scale = float(
            settings["controlnet_conditioning_scale"]
        )

    if "guidance_scale" in settings:
        cfg.guidance_scale = float(settings["guidance_scale"])

    if "temporal_feedback_strength" in settings:
        cfg.temporal_feedback_strength = float(settings["temporal_feedback_strength"])

    if "show_skeleton_overlay" in settings:
        cfg.show_skeleton_overlay = bool(settings["show_skeleton_overlay"])

    if "show_fps" in settings:
        cfg.show_fps = bool(settings["show_fps"])

    if "detect_hands" in settings:
        cfg.detect_hands = bool(settings["detect_hands"])

    if "interp_alpha" in settings:
        cfg.interp_alpha = float(settings["interp_alpha"])

    output = settings.get("output_file", "")
    return output if output else None


def show_settings_gui(cfg) -> str | None:
    """
    Show settings GUI, apply to cfg, return output path or None.
    Returns None for GUI mode, path string for headless mode.
    If user closes window without clicking Start, exits the program.
    """
    gui = SettingsGUI()
    result = gui.run()
    if result is None:
        print("[Settings] Cancelled.")
        raise SystemExit(0)
    return apply_gui_settings(result, cfg)
