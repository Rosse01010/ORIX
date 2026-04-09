"""
camera_worker.py
────────────────
Reads frames from a single camera source (RTSP / USB) and publishes them
to the Redis Stream `stream:frames`.

Each message contains:
  camera_id  – unique camera identifier
  frame_b64  – base64-encoded JPEG frame
  timestamp  – ISO-8601 UTC string

Design notes:
  • One process per camera (spawned by main_worker.py).
  • Back-pressure: if the stream grows beyond STREAM_MAX_LEN, oldest entries
    are trimmed automatically (MAXLEN ~ option).
  • Retries on connection failures with exponential back-off.
"""
from __future__ import annotations

import base64
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import redis
from tenacity import retry, stop_after_attempt, wait_exponential

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.utils.logging_utils import configure_logging, get_logger
from app.utils.preprocessing import preprocess_frame

log = get_logger(__name__)

_running = True


def _shutdown(sig, frame):
    global _running
    log.info("camera_worker_shutdown_requested")
    _running = False


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def _open_camera(source: str) -> cv2.VideoCapture:
    """Open camera with retry on failure."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_resize_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_resize_height)
    cap.set(cv2.CAP_PROP_FPS, settings.camera_frame_rate)
    log.info("camera_opened", source=source)
    return cap


def _frame_to_jpeg_b64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def run(camera_id: str, source: str) -> None:
    configure_logging(settings.worker_log_level)
    log.info("camera_worker_start", camera_id=camera_id, source=source)

    rc = redis.from_url(settings.redis_url, decode_responses=True)
    stream = settings.stream_frames
    frame_interval = 1.0 / settings.camera_frame_rate

    cap = _open_camera(source)
    last_frame_time = 0.0

    while _running:
        now = time.monotonic()
        if now - last_frame_time < frame_interval:
            time.sleep(0.001)
            continue

        ok, frame = cap.read()
        if not ok:
            log.warning("camera_read_failed", camera_id=camera_id)
            cap.release()
            time.sleep(2)
            cap = _open_camera(source)
            continue

        frame = preprocess_frame(
            frame,
            settings.camera_resize_width,
            settings.camera_resize_height,
        )
        try:
            frame_b64 = _frame_to_jpeg_b64(frame)
            rc.xadd(
                stream,
                {
                    "camera_id": camera_id,
                    "frame_b64": frame_b64,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                maxlen=settings.stream_max_len,
                approximate=True,
            )
            last_frame_time = now
        except redis.RedisError as e:
            log.warning("camera_redis_error", error=str(e))
            time.sleep(1)
        except Exception as e:
            log.exception("camera_unexpected_error", error=str(e))

    cap.release()
    log.info("camera_worker_stopped", camera_id=camera_id)


if __name__ == "__main__":
    # Allow running directly: python camera_worker.py cam_01 rtsp://...
    _, cam_id, cam_src = sys.argv
    run(cam_id, cam_src)
