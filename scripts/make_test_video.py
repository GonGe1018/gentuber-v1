"""
scripts/make_test_video.py — Generate a synthetic test video with a moving
filled human silhouette so the pipeline can be tested without a real webcam.

Usage:
    uv run python scripts/make_test_video.py
"""

import math
from pathlib import Path

import cv2
import numpy as np

OUTPUT = Path(__file__).parent.parent / "assets" / "test_input.mp4"
WIDTH, HEIGHT = 512, 512
FPS = 30
DURATION_SEC = 10


def draw_person(canvas: np.ndarray, t: float) -> None:
    cx, cy = WIDTH // 2, HEIGHT // 2
    sway = int(35 * math.sin(t * 1.1))
    bob = int(8 * math.sin(t * 2.2))
    arm_s = math.sin(t * 2.5)
    leg_s = math.sin(t * 3.0)

    skin = (180, 140, 110)
    shirt = (60, 100, 180)
    pants = (40, 50, 90)
    shoe = (30, 30, 30)

    # ── Legs ──────────────────────────────────────────────────────────────
    lhip = (cx + sway - 22, cy + 30 + bob)
    rhip = (cx + sway + 22, cy + 30 + bob)
    lknee = (cx + sway - 28 + int(18 * leg_s), cy + 100 + bob)
    rknee = (cx + sway + 28 - int(18 * leg_s), cy + 100 + bob)
    lankle = (cx + sway - 24 + int(12 * leg_s), cy + 170 + bob)
    rankle = (cx + sway + 24 - int(12 * leg_s), cy + 170 + bob)

    for a, b, c in [(lhip, lknee, lankle), (rhip, rknee, rankle)]:
        pts = np.array([a, b, c], np.int32)
        cv2.polylines(canvas, [pts], False, pants, 18, cv2.LINE_AA)

    # Shoes
    for ankle in [lankle, rankle]:
        cv2.ellipse(canvas, ankle, (14, 7), 0, 0, 360, shoe, -1, cv2.LINE_AA)

    # ── Torso ─────────────────────────────────────────────────────────────
    neck = (cx + sway, cy - 80 + bob)
    waist = (cx + sway, cy + 30 + bob)
    torso_pts = np.array(
        [
            (neck[0] - 38, neck[1] + 10),
            (neck[0] + 38, neck[1] + 10),
            (waist[0] + 28, waist[1]),
            (waist[0] - 28, waist[1]),
        ],
        np.int32,
    )
    cv2.fillPoly(canvas, [torso_pts], shirt, cv2.LINE_AA)

    # ── Arms ──────────────────────────────────────────────────────────────
    lsho = (cx + sway - 42, cy - 68 + bob)
    rsho = (cx + sway + 42, cy - 68 + bob)
    lelbow = (cx + sway - 58 + int(22 * arm_s), cy - 15 + bob)
    relbow = (cx + sway + 58 - int(22 * arm_s), cy - 15 + bob)
    lwrist = (cx + sway - 50 + int(30 * arm_s), cy + 38 + bob)
    rwrist = (cx + sway + 50 - int(30 * arm_s), cy + 38 + bob)

    for a, b, c in [(lsho, lelbow, lwrist), (rsho, relbow, rwrist)]:
        pts = np.array([a, b, c], np.int32)
        cv2.polylines(canvas, [pts], False, skin, 16, cv2.LINE_AA)

    # ── Head ──────────────────────────────────────────────────────────────
    head = (cx + sway, cy - 115 + bob)
    cv2.ellipse(canvas, head, (28, 34), 0, 0, 360, skin, -1, cv2.LINE_AA)

    # Hair
    hair_pts = np.array(
        [
            (head[0] - 28, head[1]),
            (head[0] - 30, head[1] - 20),
            (head[0] - 10, head[1] - 38),
            (head[0] + 10, head[1] - 38),
            (head[0] + 30, head[1] - 20),
            (head[0] + 28, head[1]),
        ],
        np.int32,
    )
    cv2.fillPoly(canvas, [hair_pts], (60, 40, 20), cv2.LINE_AA)

    # Eyes
    for dx in [-10, 10]:
        cv2.circle(canvas, (head[0] + dx, head[1] - 5), 4, (255, 255, 255), -1)
        cv2.circle(canvas, (head[0] + dx, head[1] - 5), 2, (30, 30, 30), -1)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = FPS * DURATION_SEC
    for i in range(total_frames):
        t = i / FPS
        # Gradient background (light grey)
        canvas = np.full((HEIGHT, WIDTH, 3), 220, dtype=np.uint8)
        for row in range(HEIGHT):
            v = int(210 + 20 * (row / HEIGHT))
            canvas[row, :] = (v, v, v)

        draw_person(canvas, t)

        cv2.putText(
            canvas,
            f"frame {i:04d}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
        )
        writer.write(canvas)

    writer.release()
    print(f"Saved {total_frames} frames -> {OUTPUT}")


if __name__ == "__main__":
    main()
