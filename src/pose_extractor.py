"""
pose_extractor.py — MediaPipe Tasks API → OpenPose-style control map.

Uses the new mediapipe.tasks API (required for mediapipe >= 0.10 / Python 3.12).
Model .task files are downloaded automatically on first run.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model URLs (official Google storage) ────────────────────────────────────
_MODELS_DIR = Path(__file__).parent.parent / "assets" / "models"

# "lite" model: ~3ms/frame vs ~17ms for "full" — enough for real-time skeleton
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_POSE_MODEL_FILE = "pose_landmarker_lite.task"

_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_HAND_MODEL_FILE = "hand_landmarker.task"

# ── OpenPose colour palette ──────────────────────────────────────────────────
_POSE_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]

# MediaPipe 33-landmark index → OpenPose 18-joint index
_MP_TO_OP = {
    0: 0,  # nose
    11: 5,  # left shoulder
    12: 2,  # right shoulder
    13: 6,  # left elbow
    14: 3,  # right elbow
    15: 7,  # left wrist
    16: 4,  # right wrist
    23: 11,  # left hip
    24: 8,  # right hip
    25: 12,  # left knee
    26: 9,  # right knee
    27: 13,  # left ankle
    28: 10,  # right ankle
    5: 14,  # right eye (MP right eye outer → OP right eye)
    2: 15,  # left eye (MP left eye outer → OP left eye)
    8: 16,  # right ear
    7: 17,  # left ear
}

# OpenPose limb pairs (0-indexed, matching ControlNet's draw_bodypose)
_LIMBS = [
    (1, 2),  # neck → right shoulder
    (1, 5),  # neck → left shoulder
    (2, 3),  # right shoulder → right elbow
    (3, 4),  # right elbow → right wrist
    (5, 6),  # left shoulder → left elbow
    (6, 7),  # left elbow → left wrist
    (1, 8),  # neck → right hip
    (8, 9),  # right hip → right knee
    (9, 10),  # right knee → right ankle
    (1, 11),  # neck → left hip
    (11, 12),  # left hip → left knee
    (12, 13),  # left knee → left ankle
    (1, 0),  # neck → nose
    (0, 14),  # nose → right eye
    (14, 16),  # right eye → right ear
    (0, 15),  # nose → left eye
    (15, 17),  # left eye → left ear
    (2, 16),  # right shoulder → right ear
    (5, 17),  # left shoulder → left ear
]

# Hand connections (21 landmarks)
_HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def _download_model(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[PoseExtractor] Downloading {dest.name} …")
    urllib.request.urlretrieve(url, dest)
    print(f"[PoseExtractor] Saved → {dest}")


class PoseExtractor:
    """
    Full-body pose + hand landmark extractor using MediaPipe Tasks API.

    Runs in VIDEO mode: uses temporal tracking between frames (~30% faster
    than IMAGE mode which re-detects every frame from scratch).

    Usage
    -----
    extractor = PoseExtractor(width=512, height=512)
    control_map, keypoints = extractor.process(bgr_frame)
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        detect_hands: bool = True,
        half_body: bool = False,
    ):
        self.width = width
        self.height = height
        self._detect_hands = detect_hands
        self._half_body = half_body
        self._frame_idx = 0  # used as timestamp_ms for VIDEO mode

        # Lower body OpenPose joint indices to exclude in half_body mode
        self._lower_body_joints = {8, 9, 10, 11, 12, 13}  # hips, knees, ankles

        pose_path = _MODELS_DIR / _POSE_MODEL_FILE
        hand_path = _MODELS_DIR / _HAND_MODEL_FILE
        _download_model(_POSE_MODEL_URL, pose_path)
        if detect_hands:
            _download_model(_HAND_MODEL_URL, hand_path)

        # VIDEO mode: MediaPipe tracks across frames (faster than IMAGE mode)
        pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(pose_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(pose_opts)

        if detect_hands:
            hand_opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hands = mp_vision.HandLandmarker.create_from_options(hand_opts)
        else:
            self._hands = None

        # Pre-allocated float32 buffer for preprocess() — avoids per-call HWC alloc
        self._f32_buf = np.empty((height, width, 3), dtype=np.float32)
        # Pre-allocated float32 buffer for preprocess_source()
        self._src_f32_buf = np.empty((height, width, 3), dtype=np.float32)

    def process(self, bgr_frame: np.ndarray):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = self._frame_idx * 33  # ~30 FPS timestamp
        self._frame_idx += 1

        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        kp: dict = {}

        # ── Pose (VIDEO mode: uses tracking, faster than per-frame detection) ──
        pose_result = self._pose.detect_for_video(mp_image, ts_ms)
        if pose_result.pose_landmarks:
            lm = pose_result.pose_landmarks[0]  # first person
            joints: dict[int, tuple[int, int]] = {}

            for mp_idx, op_idx in _MP_TO_OP.items():
                pt = lm[mp_idx]
                # visibility is in pose_world_landmarks; normalised coords in pose_landmarks
                px = int(pt.x * self.width)
                py = int(pt.y * self.height)
                joints[op_idx] = (px, py)

            # Synthesise neck
            if 2 in joints and 5 in joints:
                joints[1] = (
                    (joints[2][0] + joints[5][0]) // 2,
                    (joints[2][1] + joints[5][1]) // 2,
                )

            kp["body"] = joints

            for a, b in _LIMBS:
                # Skip lower body limbs in half_body mode
                if self._half_body and (
                    a in self._lower_body_joints or b in self._lower_body_joints
                ):
                    continue
                if a in joints and b in joints:
                    color = _POSE_COLORS[a % len(_POSE_COLORS)]
                    cv2.line(canvas, joints[a], joints[b], color, 3, cv2.LINE_AA)

            for idx, (px, py) in joints.items():
                if self._half_body and idx in self._lower_body_joints:
                    continue
                color = _POSE_COLORS[idx % len(_POSE_COLORS)]
                cv2.circle(canvas, (px, py), 5, color, -1, cv2.LINE_AA)

        # ── Hands ─────────────────────────────────────────────────────────────
        if self._hands is not None:
            hand_result = self._hands.detect_for_video(mp_image, ts_ms)
            if hand_result.hand_landmarks:
                for hand_lm in hand_result.hand_landmarks:
                    pts = [
                        (int(p.x * self.width), int(p.y * self.height)) for p in hand_lm
                    ]
                    for a, b in _HAND_CONNECTIONS:
                        cv2.line(canvas, pts[a], pts[b], (0, 200, 200), 2, cv2.LINE_AA)
                    for p in pts:
                        cv2.circle(canvas, p, 3, (0, 255, 255), -1, cv2.LINE_AA)

        return canvas, kp

    def preprocess(self, canvas: np.ndarray) -> np.ndarray:
        """
        Pre-process a control map for the graph engine hot path.

        Converts HWC uint8 RGB -> CHW float16 [0,1] in the pose thread,
        eliminating this CPU op from the diffusion engine worker.

        Uses a pre-allocated float32 intermediate buffer to avoid one
        allocation per call. Returns a fresh float16 array each time
        (safe to queue without aliasing).

        Returns
        -------
        np.ndarray  shape (3, H, W)  dtype float16
        """
        np.multiply(canvas, 1.0 / 255.0, out=self._f32_buf, casting="unsafe")
        return self._f32_buf.transpose(2, 0, 1).astype(np.float16)

    def preprocess_source(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Pre-process a camera frame for VAE encode.

        Converts BGR uint8 (H,W,3) -> CHW float16 [-1, 1].
        Resizes to (self.width, self.height) if needed.

        Returns
        -------
        np.ndarray  shape (3, H, W)  dtype float16
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            rgb = cv2.resize(rgb, (self.width, self.height))
        np.multiply(rgb, 1.0 / 127.5, out=self._src_f32_buf, casting="unsafe")
        np.subtract(self._src_f32_buf, 1.0, out=self._src_f32_buf)
        return self._src_f32_buf.transpose(2, 0, 1).astype(np.float16)

    def close(self) -> None:
        self._pose.close()
        if self._hands is not None:
            self._hands.close()
