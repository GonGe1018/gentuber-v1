"""
interpolator.py — Temporal frame blending to smooth out diffusion jitter.
"""

import cv2
import numpy as np


class FrameInterpolator:
    """
    Exponential moving average blender using cv2.addWeighted (SIMD, uint8).

    alpha=1.0  → raw generated frame (no smoothing)
    alpha=0.3  → 30% new + 70% previous (recommended)
    alpha=0.0  → frozen on first frame
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def blend(self, frame: np.ndarray) -> np.ndarray:
        if self._prev is None or self.alpha >= 1.0:
            self._prev = frame.copy()
            return frame

        blended = cv2.addWeighted(frame, self.alpha, self._prev, 1.0 - self.alpha, 0)
        self._prev = blended
        return blended

    def reset(self) -> None:
        self._prev = None
