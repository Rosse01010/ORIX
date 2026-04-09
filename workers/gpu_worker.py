"""
gpu_worker.py
─────────────
Consumes frames from stream:frames, runs:
  1. SCRFD-10G face detection (handles 100+ faces in crowds)
  2. 5-point landmark alignment
  3. Face quality scoring (sharpness + pose + size + det_score)
  4. ArcFace R100 embedding generation
  5. Publishes results to stream:vectors

Only faces above MIN_QUALITY_SCORE are embedded to avoid wasting DB lookups
on blurry, occluded, or extreme-angle faces.
"""
from __future__ import annotations

import base64
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import redis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.utils.face_quality import composite_quality, angle_hint_from_yaw
from app.utils.gpu_utils import build_detector, build_embedder
from app.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

CONSUMER_GROUP = "gpu_workers"
CONSUMER_NAME = "gpu_worker_0"

# Minimum composite quality to bother embedding a face
MIN_QUALITY_SCORE = 0.15

_running = True


def _shutdown(sig, frame):
    global _running
    _running = False
    log.info("gpu_worker_shutdown")


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


def _b64_to_frame(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _ensure_group(rc: redis.Redis, stream: str) -> None:
    try:
        rc.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def run() -> None:
    configure_logging(settings.worker_log_level)
    log.info("gpu_worker_start", backend=settings.detector_backend)

    rc = redis.from_url(settings.redis_url, decode_responses=True)
    in_stream = settings.stream_frames
    out_stream = settings.stream_vectors

    _ensure_group(rc, in_stream)

    detector = build_detector()
    embedder = build_embedder(detector)
    log.info("models_ready")

    batch_size = settings.gpu_worker_batch_size
    timeout_ms = settings.gpu_worker_timeout_ms

    while _running:
        try:
            results = rc.xreadgroup(
                CONSUMER_GROUP, CONSUMER_NAME,
                {in_stream: ">"},
                count=batch_size,
                block=timeout_ms,
            )
            if not results:
                continue

            for _stream_name, messages in results:
                for msg_id, fields in messages:
                    _process(rc, out_stream, in_stream, msg_id, fields, detector, embedder)

        except redis.ConnectionError:
            log.warning("gpu_worker_redis_reconnect")
            time.sleep(2)
        except Exception as exc:
            log.exception("gpu_worker_error", error=str(exc))
            time.sleep(0.5)

    log.info("gpu_worker_stopped")


def _process(
    rc: redis.Redis,
    out_stream: str,
    in_stream: str,
    msg_id: str,
    fields: Dict[str, Any],
    detector,
    embedder,
) -> None:
    camera_id = fields.get("camera_id", "unknown")
    timestamp = fields.get("timestamp", "")
    frame_b64 = fields.get("frame_b64", "")

    try:
        frame = _b64_to_frame(frame_b64)
        faces = detector.detect(frame)

        faces_data: List[Dict[str, Any]] = []

        for face in faces:
            # ── Quality gate ────────────────────────────────────────
            w, h = face.bbox[2], face.bbox[3]
            if w < settings.min_face_size or h < settings.min_face_size:
                continue
            if face.det_score < settings.detection_confidence:
                continue

            quality, yaw, pitch, roll, pose_sc = composite_quality(
                face.crop, face.kps, w, h, face.det_score
            )

            if quality < MIN_QUALITY_SCORE:
                log.debug(
                    "face_skipped_low_quality",
                    camera_id=camera_id,
                    quality=round(quality, 3),
                    yaw=round(yaw, 1),
                )
                continue

            # ── Embed ───────────────────────────────────────────────
            embedding = embedder.embed(face.crop)
            angle_hint = angle_hint_from_yaw(yaw)

            faces_data.append({
                "bbox": face.bbox,
                "embedding": embedding,
                "det_score": float(face.det_score),
                "quality": float(quality),
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "angle_hint": angle_hint,
            })

        if faces_data:
            rc.xadd(
                out_stream,
                {
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "faces_json": json.dumps(faces_data),
                },
                maxlen=settings.stream_max_len,
                approximate=True,
            )
            log.debug(
                "gpu_worker_processed",
                camera_id=camera_id,
                faces=len(faces_data),
            )

    except Exception as exc:
        log.warning("gpu_worker_frame_error", msg_id=msg_id, error=str(exc))
    finally:
        rc.xack(in_stream, CONSUMER_GROUP, msg_id)


if __name__ == "__main__":
    run()
