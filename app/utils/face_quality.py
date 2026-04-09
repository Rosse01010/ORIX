"""
face_quality.py
───────────────
Metrics to score a detected face before generating its embedding.

Quality score = sharpness * pose_penalty * size_penalty  ∈ [0, 1]

High quality  → use for recognition
Low quality   → skip or log as unreliable
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np


# ── Sharpness ──────────────────────────────────────────────────────────────────

def laplacian_sharpness(face_crop: np.ndarray) -> float:
    """
    Variance of Laplacian — higher = sharper.
    Returns a score in [0, 1] clamped from raw variance.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY) if face_crop.ndim == 3 else face_crop
    resized = cv2.resize(gray, (64, 64))
    variance = cv2.Laplacian(resized.astype(np.float32), cv2.CV_32F).var()
    # Variance > 500 → very sharp; < 20 → blurry
    score = min(variance / 500.0, 1.0)
    return float(score)


# ── Pose estimation ────────────────────────────────────────────────────────────

def pose_score_from_landmarks(kps: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
    """
    Estimate yaw, pitch, roll from 5-point facial landmarks.
    Returns (score, yaw_deg, pitch_deg, roll_deg).

    score = 1.0 when frontal, decreases with extreme angles.
    kps format: [[rx, ry], [lx, ly], [nx, ny], [rmx, rmy], [lmx, lmy]]
                  right_eye, left_eye, nose, right_mouth, left_mouth
    """
    if kps is None or len(kps) < 5:
        return 1.0, 0.0, 0.0, 0.0

    right_eye, left_eye, nose, right_mouth, left_mouth = kps[:5]

    # Yaw: horizontal asymmetry between eye-nose distances
    eye_center = (right_eye + left_eye) / 2.0
    eye_width = np.linalg.norm(left_eye - right_eye) + 1e-6

    nose_offset_x = (nose[0] - eye_center[0]) / eye_width
    yaw_deg = float(nose_offset_x * 90.0)   # rough estimate

    # Roll: tilt of eye line
    delta = left_eye - right_eye
    roll_deg = float(math.degrees(math.atan2(delta[1], delta[0])))

    # Pitch: vertical nose position relative to eye–mouth midpoint
    mouth_center = (right_mouth + left_mouth) / 2.0
    face_height = np.linalg.norm(mouth_center - eye_center) + 1e-6
    nose_vertical = (nose[1] - eye_center[1]) / face_height
    pitch_deg = float((nose_vertical - 0.5) * 90.0)

    # Score penalises extreme angles
    yaw_penalty = max(0.0, 1.0 - abs(yaw_deg) / 90.0)
    pitch_penalty = max(0.0, 1.0 - abs(pitch_deg) / 60.0)
    roll_penalty = max(0.0, 1.0 - abs(roll_deg) / 45.0)
    score = yaw_penalty * pitch_penalty * roll_penalty

    return float(score), yaw_deg, pitch_deg, roll_deg


def angle_hint_from_yaw(yaw_deg: float) -> str:
    """Convert yaw angle to a human-readable hint."""
    if yaw_deg < -40:
        return "right"      # camera sees left profile
    elif yaw_deg > 40:
        return "left"
    elif -15 <= yaw_deg <= 15:
        return "frontal"
    elif yaw_deg < 0:
        return "slight_right"
    else:
        return "slight_left"


# ── Size penalty ───────────────────────────────────────────────────────────────

def size_score(face_w: int, face_h: int, min_size: int = 40) -> float:
    """Penalise very small faces."""
    area = face_w * face_h
    min_area = min_size ** 2
    if area < min_area:
        return 0.0
    return min(1.0, area / (200 ** 2))


# ── Composite ─────────────────────────────────────────────────────────────────

def composite_quality(
    face_crop: np.ndarray,
    kps: Optional[np.ndarray],
    face_w: int,
    face_h: int,
    det_score: float,
) -> Tuple[float, float, float, float, float]:
    """
    Returns (quality, yaw, pitch, roll, pose_score).
    quality ∈ [0, 1] — overall usability of this face for recognition.
    """
    sharpness = laplacian_sharpness(face_crop)
    pose_sc, yaw, pitch, roll = pose_score_from_landmarks(kps)
    size_sc = size_score(face_w, face_h)

    quality = sharpness * pose_sc * size_sc * det_score
    return float(quality), yaw, pitch, roll, pose_sc
