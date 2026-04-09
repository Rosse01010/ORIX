"""
Centralised settings loaded from environment / .env file.
All workers and the API import from here.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = "insecure_default_key"

    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://faceuser:facepassword@localhost:5432/facedb"

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Redis Streams ──────────────────────────────────────────────────────────
    stream_frames: str = "stream:frames"
    stream_vectors: str = "stream:vectors"
    stream_events: str = "stream:events"
    stream_max_len: int = 1000

    # ── Cameras ────────────────────────────────────────────────────────────────
    camera_sources: str = "0"          # comma-separated: "0,rtsp://..."
    camera_frame_rate: int = 15
    camera_resize_width: int = 640
    camera_resize_height: int = 480

    @property
    def camera_source_list(self) -> List[str]:
        return [s.strip() for s in self.camera_sources.split(",") if s.strip()]

    # ── AI Models ──────────────────────────────────────────────────────────────
    detector_backend: str = "insightface"   # insightface | mediapipe
    embedder_backend: str = "arcface"       # arcface | facenet
    model_dir: str = "/app/models"
    # buffalo_l = SCRFD-10G detector + ArcFace R100 (best accuracy)
    # buffalo_m = SCRFD-5G + ArcFace R50 (balanced)
    # buffalo_s = SCRFD-500M + MBF (fastest)
    insightface_model: str = "buffalo_l"

    # ── Recognition ────────────────────────────────────────────────────────────
    # Lower threshold = more permissive (catches more angles but more false positives)
    # 0.35 works well for multi-angle; 0.50 for high-security frontal-only
    similarity_threshold: float = 0.35
    min_face_size: int = 20          # pixels — smaller = more crowd detections
    detection_confidence: float = 0.50  # lower for crowds / partial occlusion

    # ── GPU ────────────────────────────────────────────────────────────────────
    gpu_device_id: int = 0
    use_gpu: bool = True
    onnx_providers: str = "CUDAExecutionProvider,CPUExecutionProvider"

    @property
    def onnx_provider_list(self) -> List[str]:
        return [p.strip() for p in self.onnx_providers.split(",") if p.strip()]

    # ── Workers ────────────────────────────────────────────────────────────────
    gpu_worker_batch_size: int = 4
    gpu_worker_timeout_ms: int = 100
    db_worker_batch_size: int = 8
    worker_log_level: str = "INFO"

    @field_validator("similarity_threshold")
    @classmethod
    def _threshold_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("similarity_threshold must be between 0 and 1")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
