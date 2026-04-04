"""
scripts/test_stage1.py — Stage 1 smoke test: pose extraction only.

Reads the test video, runs MediaPipe pose extraction on every frame,
and writes a side-by-side comparison video (original | skeleton).

Usage:
    uv run python scripts/test_stage1.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time

import cv2
import numpy as np

from src.pose_extractor import PoseExtractor

INPUT = "assets/test_input.mp4"
OUTPUT = Path("assets/stage1_skeleton.mp4")
WIDTH, HEIGHT = 512, 512


def main() -> None:
    # Use cv2 directly for sequential (non-dropping) frame processing
    cap = cv2.VideoCapture(INPUT)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {INPUT}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ext = PoseExtractor(width=WIDTH, height=HEIGHT)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, 30, (WIDTH * 2, HEIGHT))

    frame_count = 0
    t_total = 0.0

    print(f"Processing {total} frames …")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_bgr.shape[1] != WIDTH or frame_bgr.shape[0] != HEIGHT:
            frame_bgr = cv2.resize(frame_bgr, (WIDTH, HEIGHT))

        t0 = time.perf_counter()
        skeleton_rgb, _ = ext.process(frame_bgr)
        t_total += time.perf_counter() - t0

        skeleton_bgr = cv2.cvtColor(skeleton_rgb, cv2.COLOR_RGB2BGR)
        side_by_side = np.concatenate([frame_bgr, skeleton_bgr], axis=1)
        writer.write(side_by_side)
        frame_count += 1

        if frame_count % 30 == 0:
            avg_ms = (t_total / frame_count) * 1000
            print(
                f"  frame {frame_count:3d}/{total}  avg pose latency: {avg_ms:.1f} ms"
            )

    writer.release()
    cap.release()
    ext.close()

    avg_ms = (t_total / max(frame_count, 1)) * 1000
    print(f"\nDone. {frame_count} frames processed.")
    print(f"Average pose extraction: {avg_ms:.1f} ms/frame  ({1000 / avg_ms:.1f} FPS)")
    print(f"Output saved → {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
