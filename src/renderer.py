"""
renderer.py — OpenCV display window with FPS overlay and optional skeleton.
"""

import time

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
    """

    def __init__(
        self,
        title: str = "Realtime Live2D",
        show_fps: bool = True,
        show_skeleton: bool = True,
        skeleton_alpha: float = 0.25,
    ):
        self.title = title
        self.show_fps = show_fps
        self.show_skeleton = show_skeleton
        self.skeleton_alpha = skeleton_alpha

        self._fps_counter = _FPSCounter()
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(
        self,
        frame_rgb: np.ndarray,
        skeleton_rgb: np.ndarray | None = None,
    ) -> bool:
        """
        Render one frame.

        Parameters
        ----------
        frame_rgb    : np.ndarray  RGB uint8 (H×W×3) — generated anime frame
        skeleton_rgb : np.ndarray  RGB uint8 (H×W×3) — pose control map (optional)

        Returns
        -------
        bool — False if the user pressed 'q' or closed the window
        """
        display = frame_rgb.copy()

        # Blend skeleton overlay
        if self.show_skeleton and skeleton_rgb is not None:
            mask = skeleton_rgb.sum(axis=2) > 0
            overlay = display.copy()
            overlay[mask] = (
                self.skeleton_alpha * skeleton_rgb[mask].astype(np.float32)
                + (1 - self.skeleton_alpha) * display[mask].astype(np.float32)
            ).astype(np.uint8)
            display = overlay

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

        # Convert RGB → BGR for OpenCV
        bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.title, bgr)

        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()


class _FPSCounter:
    def __init__(self, window: int = 30):
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])
