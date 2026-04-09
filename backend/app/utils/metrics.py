"""
metrics.py
──────────
Prometheus metrics for the ORIX facial recognition pipeline.

Tracks:
  - FPS per camera
  - Match confidence distribution
  - Detection/embedding/total latency histograms
  - System load (face count, false match tracking)
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

# ── Model info ────────────────────────────────────────────────────────────────
model_info = Info("orix_model", "Active face recognition model")
model_info.info({
    "detector": "scrfd_10g",
    "embedder": "arcface_r100",
    "embedding_dim": "512",
    "version": "arcface_r100_v1",
})

# ── Counters ──────────────────────────────────────────────────────────────────
faces_detected_total = Counter(
    "orix_faces_detected_total",
    "Total faces detected",
    ["camera_id"],
)

matches_total = Counter(
    "orix_matches_total",
    "Total face matches (above threshold)",
    ["camera_id", "confidence_tier"],
)

unknown_faces_total = Counter(
    "orix_unknown_faces_total",
    "Total unrecognised faces",
    ["camera_id"],
)

frames_processed_total = Counter(
    "orix_frames_processed_total",
    "Total frames processed by gpu_worker",
    ["camera_id"],
)

frames_skipped_total = Counter(
    "orix_frames_skipped_total",
    "Frames skipped (frame_skip_interval or latency budget)",
    ["camera_id", "reason"],
)

# ── Gauges ────────────────────────────────────────────────────────────────────
camera_fps = Gauge(
    "orix_camera_fps",
    "Estimated FPS per camera",
    ["camera_id"],
)

# ── Histograms ────────────────────────────────────────────────────────────────
detection_latency = Histogram(
    "orix_detection_latency_ms",
    "Face detection latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 200, 500, 1000],
)

embedding_latency = Histogram(
    "orix_embedding_latency_ms",
    "Embedding extraction latency in milliseconds",
    buckets=[2, 5, 10, 25, 50, 100, 200],
)

total_frame_latency = Histogram(
    "orix_total_frame_latency_ms",
    "Total per-frame processing latency in milliseconds",
    buckets=[10, 25, 50, 100, 200, 500, 1000],
)

match_confidence = Histogram(
    "orix_match_confidence",
    "Distribution of match cosine similarities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

db_search_latency = Histogram(
    "orix_db_search_latency_ms",
    "Database search latency in milliseconds",
    buckets=[1, 5, 10, 25, 50, 100, 200],
)
