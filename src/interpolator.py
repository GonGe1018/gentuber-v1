"""
interpolator.py — Temporal frame blending to smooth out diffusion jitter.

When the diffusion engine produces frames at ~10-20 FPS but the display
runs at 30+ FPS, simple alpha-blending between the previous and current
generated frame reduces flickering without adding complex optical-flow cost.
"""

import numpy as np


class FrameInterpolator:
    """
    Exponential moving average blender.

    alpha=0.0  → always show previous frame (frozen)
    alpha=1.0  → always show raw generated frame (no smoothing)
    alpha=0.3  → recommended: 30% new + 70% previous
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._prev: np.ndarray | None = None

    def blend(self, frame: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        frame : np.ndarray  RGB uint8 (H×W×3)

        Returns
        -------
        blended : np.ndarray  RGB uint8 (H×W×3)
        """
        if self._prev is None or self.alpha >= 1.0:
            self._prev = frame.copy()
            return frame

        blended = (
            self.alpha * frame.astype(np.float32)
            + (1.0 - self.alpha) * self._prev.astype(np.float32)
        ).astype(np.uint8)

        self._prev = blended
        return blended

    def reset(self) -> None:
        self._prev = None
