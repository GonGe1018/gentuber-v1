"""
scripts/make_test_video.py — Generate a synthetic test video with a moving
stick figure so the pipeline can be tested without a real webcam or footage.

Usage:
    uv run python scripts/make_test_video.py
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

OUTPUT = Path(__file__).parent.parent / "assets" / "test_input.mp4"
WIDTH, HEIGHT = 512, 512
FPS = 30
DURATION_SEC = 10


def draw_stick_figure(canvas: np.ndarray, t: float) -> None:
    """Draw a simple animated stick figure at time t (seconds)."""
    cx, cy = WIDTH // 2, HEIGHT // 2

    # Gentle sway
    sway = int(40 * math.sin(t * 1.2))
    bob = int(10 * math.sin(t * 2.4))

    # Joint positions
    head = (cx + sway, cy - 120 + bob)
    neck = (cx + sway, cy - 85 + bob)
    lsho = (cx + sway - 45, cy - 70 + bob)
    rsho = (cx + sway + 45, cy - 70 + bob)
    lelbow = (cx + sway - 65 + int(20 * math.sin(t * 2)), cy - 20 + bob)
    relbow = (cx + sway + 65 - int(20 * math.sin(t * 2)), cy - 20 + bob)
    lwrist = (cx + sway - 55 + int(30 * math.sin(t * 2 + 0.5)), cy + 30 + bob)
    rwrist = (cx + sway + 55 - int(30 * math.sin(t * 2 + 0.5)), cy + 30 + bob)
    hip_c = (cx + sway, cy + 20 + bob)
    lhip = (cx + sway - 30, cy + 30 + bob)
    rhip = (cx + sway + 30, cy + 30 + bob)
    lknee = (cx + sway - 35 + int(15 * math.sin(t * 3)), cy + 100 + bob)
    rknee = (cx + sway + 35 - int(15 * math.sin(t * 3 + math.pi)), cy + 100 + bob)
    lankle = (cx + sway - 30 + int(10 * math.sin(t * 3)), cy + 170 + bob)
    rankle = (cx + sway + 30 - int(10 * math.sin(t * 3 + math.pi)), cy + 170 + bob)

    color = (220, 220, 220)
    thick = 4

    limbs = [
        (neck, lsho),
        (neck, rsho),
        (lsho, lelbow),
        (lelbow, lwrist),
        (rsho, relbow),
        (relbow, rwrist),
        (neck, hip_c),
        (hip_c, lhip),
        (hip_c, rhip),
        (lhip, lknee),
        (lknee, lankle),
        (rhip, rknee),
        (rknee, rankle),
    ]
    for a, b in limbs:
        cv2.line(canvas, a, b, color, thick, cv2.LINE_AA)

    # Head circle
    cv2.circle(canvas, head, 22, color, thick, cv2.LINE_AA)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = FPS * DURATION_SEC
    for i in range(total_frames):
        t = i / FPS
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Gradient background
        for row in range(HEIGHT):
            v = int(30 + 20 * (row / HEIGHT))
            canvas[row, :] = (v, v // 2, v)

        draw_stick_figure(canvas, t)

        # Frame counter
        cv2.putText(
            canvas,
            f"frame {i:04d}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 100),
            1,
        )

        writer.write(canvas)

    writer.release()
    print(f"Saved {total_frames} frames → {OUTPUT}")


if __name__ == "__main__":
    main()
