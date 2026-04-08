"""
settings_gui.py — Tkinter settings dialog shown before pipeline starts.

- Loads/saves current settings from settings.json
- Keeps run history in settings_history.json (auto-named presets)
- History panel on the right side for one-click preset loading
"""

import json
import tkinter as tk
from tkinter import ttk, filedialog
from datetime import datetime
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"
HISTORY_FILE = Path(__file__).parent.parent / "settings_history.json"
MAX_HISTORY = 20

SETTINGS_SCHEMA = [
    {
        "section": "Input",
        "fields": [
            {
                "key": "video_source",
                "label": "Source",
                "type": "entry",
                "default": "assets/test_input.mp4",
            },
            {
                "key": "reference_image",
                "label": "Reference Image",
                "type": "file",
                "default": "assets/reference.png",
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
                "options": ["ip_adapter"],
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
            {"key": "seed", "label": "Seed", "type": "entry", "default": "42"},
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
            },
            {
                "key": "controlnet_conditioning_scale",
                "label": "ControlNet Scale",
                "type": "slider",
                "default": 1.5,
                "min": 0.5,
                "max": 3.0,
            },
            {
                "key": "guidance_scale",
                "label": "Guidance Scale",
                "type": "slider",
                "default": 1.0,
                "min": 1.0,
                "max": 3.0,
            },
            {
                "key": "temporal_feedback_strength",
                "label": "Feedback Strength",
                "type": "slider",
                "default": 0.3,
                "min": 0.1,
                "max": 1.0,
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
                "key": "half_body",
                "label": "Half Body (VTuber)",
                "type": "check",
                "default": False,
            },
            {
                "key": "no_interp",
                "label": "No Interpolation",
                "type": "check",
                "default": False,
            },
            {
                "key": "interp_alpha",
                "label": "Smoothing",
                "type": "slider",
                "default": 0.3,
                "min": 0.0,
                "max": 1.0,
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
            },
        ],
    },
]


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_settings(data: dict) -> None:
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_history(history: list[dict]) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-MAX_HISTORY:], f, indent=2, ensure_ascii=False)


def _make_preset_name(settings: dict) -> str:
    """Auto-generate a short name from key parameters."""
    backend = settings.get("engine_backend", "?")[:6]
    ip = settings.get("ip_adapter_scale", 0.5)
    cn = settings.get("controlnet_conditioning_scale", 1.5)
    fb = settings.get("temporal_feedback_strength", 0.3)
    gs = settings.get("guidance_scale", 1.0)
    st = settings.get("num_inference_steps", 4)
    sz = settings.get("output_size", "384")
    return f"{backend} | {sz}px st{st} ip{ip:.1f} cn{cn:.1f} fb{fb:.1f} gs{gs:.1f}"


class SettingsGUI:
    def __init__(self):
        self.result: dict | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._history: list[dict] = []

    def run(self) -> dict | None:
        saved = _load_settings()
        self._history = _load_history()

        root = tk.Tk()
        root.title("Realtime Live2D - Settings")
        root.resizable(False, False)

        # Dark theme
        bg = "#2b2b2b"
        fg = "#e0e0e0"
        accent = "#4a9eff"
        entry_bg = "#3c3c3c"
        section_fg = "#4a9eff"
        hist_bg = "#1e1e1e"
        hist_sel = "#3a3a5c"

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
        style.configure("TLabelframe", background=bg, foreground=section_fg)
        style.configure(
            "TLabelframe.Label",
            background=bg,
            foreground=section_fg,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TScale", background=bg, troughcolor=entry_bg)
        style.configure("Small.TButton", font=("Segoe UI", 8))

        # ── Layout: left (settings) + right (history) ─────────────────────
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)

        left_frame = ttk.Frame(outer)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        right_frame = ttk.LabelFrame(outer, text="History", padding=(5, 5))
        right_frame.pack(side="right", fill="both", padx=(5, 0))

        # ── Left: settings fields ─────────────────────────────────────────
        row = 0
        for section in SETTINGS_SCHEMA:
            lf = ttk.LabelFrame(left_frame, text=section["section"], padding=(10, 5))
            lf.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(3, 3))
            row += 1

            for i, field in enumerate(section["fields"]):
                key = field["key"]
                default = field["default"]
                val = saved.get(key, default)

                label = ttk.Label(lf, text=field["label"])
                label.grid(row=i, column=0, sticky="w", padx=(5, 10), pady=2)

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
                    scale.grid(row=i, column=1, sticky="ew", padx=5, pady=2)

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
                    scale.grid(row=i, column=1, sticky="ew", padx=5, pady=2)

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
                        row=i, column=1, columnspan=2, sticky="ew", padx=5, pady=2
                    )

                elif field["type"] == "check":
                    var = tk.BooleanVar(value=bool(val))
                    self._vars[key] = var
                    chk = ttk.Checkbutton(lf, variable=var)
                    chk.grid(row=i, column=1, sticky="w", padx=5, pady=2)

                elif field["type"] == "file":
                    var = tk.StringVar(value=str(val))
                    self._vars[key] = var
                    entry = ttk.Entry(lf, textvariable=var, width=25)
                    entry.grid(row=i, column=1, sticky="ew", padx=5, pady=2)

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
                    btn.grid(row=i, column=2, padx=(0, 5), pady=2)

                elif field["type"] == "entry":
                    var = tk.StringVar(value=str(val))
                    self._vars[key] = var
                    entry = ttk.Entry(lf, textvariable=var, width=28)
                    entry.grid(
                        row=i, column=1, columnspan=2, sticky="ew", padx=5, pady=2
                    )

                lf.columnconfigure(1, weight=1)

        # ── Left: buttons ─────────────────────────────────────────────────
        btn_frame = ttk.Frame(left_frame)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=(10, 5))

        def on_start():
            self.result = self._collect()
            _save_settings(self.result)
            # Add to history
            entry = {
                "name": _make_preset_name(self.result),
                "time": datetime.now().strftime("%m/%d %H:%M"),
                "settings": self.result,
            }
            self._history.append(entry)
            _save_history(self._history)
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

        # ── Right: history listbox ────────────────────────────────────────
        hist_list = tk.Listbox(
            right_frame,
            bg=hist_bg,
            fg=fg,
            selectbackground=hist_sel,
            selectforeground="#ffffff",
            font=("Consolas", 8),
            width=38,
            height=28,
            borderwidth=0,
            highlightthickness=0,
            activestyle="none",
        )
        hist_list.pack(side="top", fill="both", expand=True, pady=(0, 5))

        # Populate history (newest first)
        for h in reversed(self._history):
            display = f"{h.get('time', '')}  {h.get('name', '?')}"
            hist_list.insert("end", display)

        if not self._history:
            hist_list.insert("end", "  (no history yet)")

        def on_load_history():
            sel = hist_list.curselection()
            if not sel:
                return
            idx = len(self._history) - 1 - sel[0]  # reversed order
            if 0 <= idx < len(self._history):
                preset = self._history[idx].get("settings", {})
                self._apply_preset(preset)

        def on_delete_history():
            sel = hist_list.curselection()
            if not sel:
                return
            idx = len(self._history) - 1 - sel[0]
            if 0 <= idx < len(self._history):
                self._history.pop(idx)
                _save_history(self._history)
                hist_list.delete(sel[0])

        hist_btn_frame = ttk.Frame(right_frame)
        hist_btn_frame.pack(side="bottom", fill="x")

        load_btn = ttk.Button(
            hist_btn_frame, text="Load", command=on_load_history, style="Small.TButton"
        )
        load_btn.pack(side="left", padx=2)

        del_btn = ttk.Button(
            hist_btn_frame,
            text="Delete",
            command=on_delete_history,
            style="Small.TButton",
        )
        del_btn.pack(side="left", padx=2)

        # Double-click to load
        hist_list.bind("<Double-1>", lambda e: on_load_history())

        # Center window
        root.update_idletasks()
        w = root.winfo_width()
        h = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (w // 2)
        y = (root.winfo_screenheight() // 2) - (h // 2)
        root.geometry(f"+{x}+{y}")

        root.bind("<Return>", lambda e: on_start())
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        root.mainloop()
        return self.result

    def _apply_preset(self, preset: dict) -> None:
        """Load a preset dict into the GUI variables."""
        for section in SETTINGS_SCHEMA:
            for field in section["fields"]:
                key = field["key"]
                if key in preset and key in self._vars:
                    self._vars[key].set(preset[key])

    def _collect(self) -> dict:
        result = {}
        for section in SETTINGS_SCHEMA:
            for field in section["fields"]:
                key = field["key"]
                if key in self._vars:
                    val = self._vars[key].get()
                    if field["type"] == "slider":
                        val = round(float(val), 3)
                    elif field["type"] == "int_slider":
                        val = int(round(float(val)))
                    elif field["type"] == "check":
                        val = bool(val)
                    result[key] = val
        return result


def apply_gui_settings(settings: dict, cfg) -> str | None:
    """Apply GUI settings dict to a Config object.
    Returns output_file path or None for GUI mode."""
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
    if "half_body" in settings:
        cfg.half_body = bool(settings["half_body"])
    if "no_interp" in settings and settings["no_interp"]:
        cfg.interp_alpha = 1.0
    elif "interp_alpha" in settings:
        cfg.interp_alpha = float(settings["interp_alpha"])

    output = settings.get("output_file", "")
    return output if output else None


def show_settings_gui(cfg) -> str | None:
    """Show settings GUI, apply to cfg, return output path or None.
    If user closes window without clicking Start, exits the program."""
    gui = SettingsGUI()
    result = gui.run()
    if result is None:
        print("[Settings] Cancelled.")
        raise SystemExit(0)
    return apply_gui_settings(result, cfg)
