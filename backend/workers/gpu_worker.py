"""
gpu_worker.py
─────────────
Consumes frames from stream:frames, runs:
  1. SCRFD-10G face detection (handles 100+ faces in crowds)
  2. 5-point landmark alignment
  3. Face quality scoring (sharpness + pose + size + det_score)
  4. ArcFace R100 embedding generation
  5. Publishes results to stream:vectors

Production hardening:
  - Frame skipping (configurable N)
  - Resize before detection (max 640x640)
  - Face limit per frame (max_faces_per_frame)
  - Latency control: skip frame if processing exceeds max_processing_time_ms
  - Fault tolerance: try/catch around inference, GPU->CPU fallback
  - Embedding versioning tag on every output
  - Per-frame timing logs (detection_ms, embedding_ms, total_ms)
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

# Embedding version tag — must match services/insightface_service.py
EMBEDDING_VERSION = "arcface_r100_v1"

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


def _resize_for_detection(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Resize frame to fit within max_w x max_h, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_w and h <= max_h:
        return frame
    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _ensure_group(rc: redis.Redis, stream: str) -> None:
    try:
        rc.xgroup_create(stream, CONSUMER_GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def _build_models_with_fallback():
    """Build detector + embedder. Falls back to CPU if GPU init fails."""
    try:
        detector = build_detector()
        embedder = build_embedder(detector)
        return detector, embedder
    except Exception as exc:
        log.warning("gpu_init_failed_falling_back_to_cpu", error=str(exc))
        # Force CPU providers and retry
        original_gpu = settings.use_gpu
        settings.__dict__["use_gpu"] = False
        try:
            detector = build_detector()
            embedder = build_embedder(detector)
            log.info("cpu_fallback_models_ready")
            return detector, embedder
        finally:
            settings.__dict__["use_gpu"] = original_gpu


def run() -> None:
    configure_logging(settings.worker_log_level)
    log.info("gpu_worker_start", backend=settings.detector_backend)

    rc = redis.from_url(settings.redis_url, decode_responses=True)
    in_stream = settings.stream_frames
    out_stream = settings.stream_vectors

    _ensure_group(rc, in_stream)

    detector, embedder = _build_models_with_fallback()
    log.info("models_ready")

    batch_size = settings.gpu_worker_batch_size
    timeout_ms = settings.gpu_worker_timeout_ms
    frame_counter = 0
    redis_retry_delay = 1

    while _running:
        try:
            results = rc.xreadgroup(
                CONSUMER_GROUP, CONSUMER_NAME,
                {in_stream: ">"},
                count=batch_size,
                block=timeout_ms,
            )
            redis_retry_delay = 1  # reset on success

            if not results:
                continue

            for _stream_name, messages in results:
                for msg_id, fields in messages:
                    # Frame skipping: only process every Nth frame
                    frame_counter += 1
                    if frame_counter % settings.frame_skip_interval != 0:
                        rc.xack(in_stream, CONSUMER_GROUP, msg_id)
                        continue

                    _process(rc, out_stream, in_stream, msg_id, fields,
                             detector, embedder)

        except redis.ConnectionError:
            log.warning("gpu_worker_redis_reconnect", retry_in=redis_retry_delay)
            time.sleep(redis_retry_delay)
            redis_retry_delay = min(redis_retry_delay * 2, 30)
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

    t_start = time.monotonic()

    try:
        frame = _b64_to_frame(frame_b64)

        # Resize before detection for performance
        frame = _resize_for_detection(
            frame,
            settings.max_detection_width,
            settings.max_detection_height,
        )

        # ── Detection with fault tolerance ────────────────────────
        t_det = time.monotonic()
        try:
            faces = detector.detect(frame)
        except Exception as exc:
            log.warning("detection_error", camera_id=camera_id, error=str(exc))
            return
        detection_ms = (time.monotonic() - t_det) * 1000

        # Limit faces per frame (sorted by area, largest first)
        if len(faces) > settings.max_faces_per_frame:
            faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
            faces = faces[:settings.max_faces_per_frame]

        faces_data: List[Dict[str, Any]] = []
        embedding_ms_total = 0.0

        for face in faces:
            # Latency check: skip remaining faces if budget exceeded
            elapsed_ms = (time.monotonic() - t_start) * 1000
            if elapsed_ms > settings.max_processing_time_ms:
                log.debug(
                    "frame_budget_exceeded",
                    camera_id=camera_id,
                    elapsed_ms=round(elapsed_ms, 1),
                    faces_processed=len(faces_data),
                )
                break

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

            # ── Embed with fault tolerance ──────────────────────────
            t_emb = time.monotonic()
            try:
                embedding = embedder.embed(face.crop)
            except Exception as exc:
                log.warning("embedding_error", camera_id=camera_id, error=str(exc))
                continue
            emb_ms = (time.monotonic() - t_emb) * 1000
            embedding_ms_total += emb_ms

            angle_hint = angle_hint_from_yaw(yaw)

            faces_data.append({
                "bbox": face.bbox,
                "embedding": embedding,
                "embedding_version": EMBEDDING_VERSION,
                "det_score": float(face.det_score),
                "quality": float(quality),
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "angle_hint": angle_hint,
            })

        total_ms = (time.monotonic() - t_start) * 1000

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
                detection_ms=round(detection_ms, 1),
                embedding_ms=round(embedding_ms_total, 1),
                total_ms=round(total_ms, 1),
            )

    except Exception as exc:
        log.warning("gpu_worker_frame_error", msg_id=msg_id, error=str(exc))
    finally:
        rc.xack(in_stream, CONSUMER_GROUP, msg_id)


if __name__ == "__main__":
    run()
