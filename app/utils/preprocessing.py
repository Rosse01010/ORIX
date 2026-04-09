"""
preprocessing.py
────────────────
Frame pre-processing utilities shared by camera_worker and gpu_worker.
"""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame(
    frame: np.ndarray,
    target_width: int = 640,
    target_height: int = 480,
) -> np.ndarray:
    """
    Resize frame while preserving aspect ratio (letterbox).
    Returns a frame of exactly (target_height, target_width, 3).
    """
    h, w = frame.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to target size
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded


def normalize_face_chip(
    face: np.ndarray,
    size: int = 112,
) -> np.ndarray:
    """Resize and normalize a face crop for embedding models."""
    resized = cv2.resize(face, (size, size), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    # ImageNet-style normalization
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    normalized = (normalized - mean) / std
    return normalized


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
