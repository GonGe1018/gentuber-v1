"""
capture.py — Non-blocking video/webcam reader with a ring buffer.

A background thread continuously grabs frames so the main pipeline
never blocks waiting for I/O.  Only the latest `queue_size` frames
are kept; older ones are dropped to prevent latency build-up.
"""

import threading
import time
from collections import deque

import cv2
import numpy as np


class FrameBuffer:
    """Thread-safe ring buffer that always holds the freshest frames."""

    def __init__(self, maxlen: int = 2):
        self._buf: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append(frame)
        self._event.set()

    def get(self, timeout: float = 1.0) -> np.ndarray | None:
        """Block until a new frame is available, then return it."""
        if not self._event.wait(timeout):
            return None
        with self._lock:
            if not self._buf:
                return None
            frame = self._buf[-1]  # always take the newest
            self._buf.clear()
            self._event.clear()
        return frame

    def empty(self) -> bool:
        with self._lock:
            return len(self._buf) == 0


class VideoCapture:
    """
    Wraps cv2.VideoCapture in a daemon thread.

    Parameters
    ----------
    source : int | str
        Webcam index (int) or video file path (str).
    width, height : int
        Target resolution; frames are resized to this.
    queue_size : int
        Ring-buffer depth.  Keep small (2) to minimise latency.
    loop : bool
        If True, video files restart when they reach the end.
    """

    def __init__(
        self,
        source,
        width: int = 512,
        height: int = 512,
        queue_size: int = 2,
        loop: bool = True,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.loop = loop
        self.buffer = FrameBuffer(maxlen=queue_size)

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Hint the driver to use a small internal buffer (webcam only)
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._running = False
        self._thread: threading.Thread | None = None

    # ── public API ──────────────────────────────────────────────────────────

    def start(self) -> "VideoCapture":
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._cap.release()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Return the latest BGR frame (H×W×3 uint8), or None on timeout."""
        return self.buffer.get(timeout=timeout)

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ── internal ────────────────────────────────────────────────────────────

    def _reader(self) -> None:
        # For video files, pace reads to match the source FPS
        is_file = isinstance(self.source, str)
        if is_file:
            src_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1.0 / src_fps
        else:
            frame_interval = 0.0  # webcam: read as fast as possible

        while self._running:
            t0 = time.perf_counter()

            ok, frame = self._cap.read()
            if not ok:
                if self.loop and is_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self._running = False
                    break

            # Resize to target resolution
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(
                    frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
                )

            self.buffer.put(frame)

            # Sleep to match source FPS (video files only)
            if frame_interval > 0:
                elapsed = time.perf_counter() - t0
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
