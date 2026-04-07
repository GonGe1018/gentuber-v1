"""
renderer.py — OpenCV display window with FPS overlay and optional skeleton.
"""

import time
from collections import deque

import cv2
import numpy as np


class Renderer:
    """
    Displays generated frames in an OpenCV window.

    Parameters
    ----------
    title : str
    show_fps : bool
    show_skeleton : bool   — if True, blend the skeleton control map on top
    skeleton_alpha : float — opacity of skeleton overlay (0–1)
    max_fps : float        — cap display rate (0 = uncapped)
    """

    def __init__(
        self,
        title: str = "Realtime Live2D",
        show_fps: bool = True,
        show_skeleton: bool = True,
        skeleton_alpha: float = 0.25,
        max_fps: float = 60.0,
    ):
        self.title = title
        self.show_fps = show_fps
        self.show_skeleton = show_skeleton
        self.skeleton_alpha = skeleton_alpha
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_show = 0.0

        self._fps_counter = _FPSCounter()
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(
        self,
        frame_rgb: np.ndarray,
        skeleton_rgb: np.ndarray | None = None,
        source_rgb: np.ndarray | None = None,
    ) -> bool:
        # Rate cap — skip display if we're ahead of monitor refresh
        now = time.perf_counter()
        if self._min_interval > 0 and (now - self._last_show) < self._min_interval:
            key = cv2.waitKey(1) & 0xFF
            return key != ord("q")
        self._last_show = now

        display = frame_rgb.copy()

        # Skeleton overlay — use cv2.addWeighted on masked region
        if self.show_skeleton and skeleton_rgb is not None:
            mask = skeleton_rgb.sum(axis=2) > 0
            if mask.any():
                # Blend only where skeleton is non-zero (faster than full-frame blend)
                display[mask] = cv2.addWeighted(
                    skeleton_rgb,
                    self.skeleton_alpha,
                    display,
                    1.0 - self.skeleton_alpha,
                    0,
                )[mask]

        # FPS counter
        fps = self._fps_counter.tick()
        if self.show_fps:
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Side-by-side: source (left) | generated (right)
        if source_rgb is not None:
            h, w = display.shape[:2]
            src_resized = cv2.resize(source_rgb, (w, h))
            # Add label to source
            cv2.putText(
                src_resized,
                "Source",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            combined = np.hstack([src_resized, display])
            bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        else:
            bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

        cv2.imshow(self.title, bgr)

        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()


class _FPSCounter:
    def __init__(self, window: int = 60):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])
