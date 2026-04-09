"""
Runtime parameter tuning panel.
Runs a tkinter window in a separate thread alongside the main pipeline.
Sliders update cfg values live; engine reads cfg each frame.
"""

import threading
import tkinter as tk
from tkinter import ttk


_SLIDERS = [
    ("ip_adapter_scale", "IP-Adapter Scale", 0.0, 1.0, 0.01),
    ("controlnet_conditioning_scale", "ControlNet Scale", 0.0, 3.0, 0.01),
    ("guidance_scale", "Guidance Scale", 1.0, 3.0, 0.01),
    ("temporal_feedback_strength", "Feedback Strength", 0.05, 1.0, 0.01),
    ("motion_lo", "Motion Lo (jitter)", 0.001, 0.05, 0.001),
    ("motion_hi", "Motion Hi (reset)", 0.01, 0.15, 0.001),
    ("motion_max_strength", "Motion Max Strength", 0.3, 1.0, 0.01),
]


class TuningPanel:
    """Tkinter panel that edits cfg in-place."""

    def __init__(self, cfg):
        self._cfg = cfg
        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        root = tk.Tk()
        self._root = root
        root.title("GenTuber v1 — Tuning")
        root.configure(bg="#2b2b2b")
        root.resizable(False, False)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background="#2b2b2b")
        style.configure(
            "Dark.TLabel",
            background="#2b2b2b",
            foreground="#e0e0e0",
            font=("Segoe UI", 9),
        )
        style.configure(
            "Header.TLabel",
            background="#2b2b2b",
            foreground="#80cbc4",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure("Apply.TButton", font=("Segoe UI", 10, "bold"))

        main = ttk.Frame(root, style="Dark.TFrame", padding=10)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Runtime Tuning", style="Header.TLabel").pack(pady=(0, 8))

        self._vars: dict[str, tk.DoubleVar] = {}
        self._value_labels: dict[str, ttk.Label] = {}

        for key, label, lo, hi, step in _SLIDERS:
            current = getattr(self._cfg, key, lo)
            var = tk.DoubleVar(value=current)
            self._vars[key] = var

            row = ttk.Frame(main, style="Dark.TFrame")
            row.pack(fill="x", pady=2)

            ttk.Label(row, text=label, style="Dark.TLabel", width=22, anchor="w").pack(
                side="left"
            )

            val_label = ttk.Label(
                row, text=f"{current:.3f}", style="Dark.TLabel", width=7, anchor="e"
            )
            val_label.pack(side="right")
            self._value_labels[key] = val_label

            scale = tk.Scale(
                row,
                from_=lo,
                to=hi,
                variable=var,
                orient="horizontal",
                resolution=step,
                showvalue=False,
                bg="#2b2b2b",
                fg="#e0e0e0",
                troughcolor="#404040",
                highlightthickness=0,
                bd=0,
                length=200,
                command=lambda v, k=key: self._on_slide(k),
            )
            scale.pack(side="right", fill="x", expand=True, padx=(4, 4))

        # Buttons
        btn_frame = ttk.Frame(main, style="Dark.TFrame")
        btn_frame.pack(fill="x", pady=(12, 0))

        apply_btn = ttk.Button(
            btn_frame, text="  Apply  ", style="Apply.TButton", command=self._apply
        )
        apply_btn.pack(side="left", padx=4)

        reset_btn = ttk.Button(btn_frame, text="  Reset  ", command=self._reset)
        reset_btn.pack(side="left", padx=4)

        # Status
        self._status_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self._status_var, style="Dark.TLabel").pack(
            side="right", padx=4
        )

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.mainloop()

    def _on_slide(self, key: str):
        val = self._vars[key].get()
        self._value_labels[key].config(text=f"{val:.3f}")

    def _apply(self):
        for key, var in self._vars.items():
            setattr(self._cfg, key, var.get())
        self._status_var.set("Applied!")
        if self._root:
            self._root.after(1500, lambda: self._status_var.set(""))

    def _reset(self):
        for key, var in self._vars.items():
            current = getattr(self._cfg, key)
            var.set(current)
            self._value_labels[key].config(text=f"{current:.3f}")
        self._status_var.set("Reset to current")
        if self._root:
            self._root.after(1500, lambda: self._status_var.set(""))

    def _on_close(self):
        if self._root:
            self._root.destroy()
            self._root = None
