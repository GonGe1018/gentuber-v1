"""
pose_process.py — Pose extraction in a separate OS process.

Running MediaPipe in its own process gives it a dedicated GIL, so it
never blocks the diffusion thread.  Frames are exchanged via
multiprocessing.shared_memory for zero-copy transfer.
"""

import multiprocessing as mp
import multiprocessing.shared_memory as shm
import numpy as np


def _pose_loop(
    shm_in_name: str,
    shm_out_name: str,
    frame_shape: tuple,
    frame_ready: mp.Event,
    result_ready: mp.Event,
    stop_flag: mp.Event,
    ready_event: mp.Event,
    width: int,
    height: int,
):
    """Runs inside the child process."""
    import sys, pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    from src.pose_extractor import PoseExtractor

    extractor = PoseExtractor(width=width, height=height)

    mem_in = shm.SharedMemory(name=shm_in_name)
    mem_out = shm.SharedMemory(name=shm_out_name)

    buf_in = np.ndarray(frame_shape, dtype=np.uint8, buffer=mem_in.buf)
    buf_out = np.ndarray(frame_shape, dtype=np.uint8, buffer=mem_out.buf)

    # Signal host that we are ready to accept frames
    ready_event.set()

    while not stop_flag.is_set():
        if not frame_ready.wait(timeout=0.05):
            continue
        frame_ready.clear()

        frame_bgr = buf_in.copy()
        control_rgb, _ = extractor.process(frame_bgr)
        np.copyto(buf_out, control_rgb)
        result_ready.set()

    extractor.close()
    mem_in.close()
    mem_out.close()


class PoseProcess:
    """
    Manages a child process that runs MediaPipe pose extraction.

    Parameters
    ----------
    width, height : int
    """

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self._shape = (height, width, 3)
        nbytes = int(np.prod(self._shape))

        self._shm_in = shm.SharedMemory(create=True, size=nbytes)
        self._shm_out = shm.SharedMemory(create=True, size=nbytes)

        self._buf_in = np.ndarray(self._shape, dtype=np.uint8, buffer=self._shm_in.buf)
        self._buf_out = np.ndarray(
            self._shape, dtype=np.uint8, buffer=self._shm_out.buf
        )

        self._frame_ready = mp.Event()
        self._result_ready = mp.Event()
        self._stop_flag = mp.Event()
        self._ready_event = mp.Event()

        self._proc: mp.Process | None = None

    def start(self, timeout: float = 30.0) -> "PoseProcess":
        self._proc = mp.Process(
            target=_pose_loop,
            args=(
                self._shm_in.name,
                self._shm_out.name,
                self._shape,
                self._frame_ready,
                self._result_ready,
                self._stop_flag,
                self._ready_event,
                self.width,
                self.height,
            ),
            daemon=True,
            name="pose-proc",
        )
        self._proc.start()
        if not self._ready_event.wait(timeout=timeout):
            raise RuntimeError("PoseProcess failed to initialise within timeout")
        print("[PoseProcess] Ready.")
        return self

    def push_frame(self, bgr_frame: np.ndarray) -> None:
        np.copyto(self._buf_in, bgr_frame)
        self._result_ready.clear()
        self._frame_ready.set()

    def get_control(self, timeout: float = 0.1) -> np.ndarray | None:
        if not self._result_ready.wait(timeout=timeout):
            return None
        self._result_ready.clear()
        return self._buf_out.copy()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._proc:
            self._proc.join(timeout=5.0)
            if self._proc.is_alive():
                self._proc.kill()
        self._shm_in.close()
        self._shm_in.unlink()
        self._shm_out.close()
        self._shm_out.unlink()
