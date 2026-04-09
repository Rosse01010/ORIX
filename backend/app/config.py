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
    camera_sources: str = "0"
    camera_frame_rate: int = 15
    camera_resize_width: int = 640
    camera_resize_height: int = 480

    @property
    def camera_source_list(self) -> List[str]:
        return [s.strip() for s in self.camera_sources.split(",") if s.strip()]

    # ── AI Models ──────────────────────────────────────────────────────────────
    detector_backend: str = "insightface"
    embedder_backend: str = "arcface"
    model_dir: str = "/app/models"
    insightface_model: str = "buffalo_l"

    # ── Recognition thresholds ─────────────────────────────────────────────────
    # ArcFace (Deng et al., CVPR 2019): 512-dim L2-normalised embeddings on the
    # unit hypersphere. Similarity is cosine similarity ∈ [-1, 1].
    #
    # Recommended ranges (derived from ArcFace angle-distribution analysis and
    # VGGFace2 cross-pose similarity matrices):
    #   ≥ 0.55          → confident match (frontal, good lighting)
    #   0.40 – 0.54     → probable match  (off-axis, partial occlusion, age gap)
    #   0.30 – 0.39     → uncertain       → candidate panel only
    #   < 0.30          → different identity
    #
    # Default 0.40 suits CCTV / crowd scenarios where profile views are common.
    # Use 0.55 for access-control with strictly frontal enrollment images.
    similarity_threshold: float = 0.40

    # Minimum cosine similarity for a person to appear in the candidate panel.
    # ArcFace paper (Figure 7) shows negative pairs rarely exceed 0.35 even
    # under large pose variation, so 0.30 is a safe noise-filtering floor.
    candidate_min_sim: float = 0.30

    # |yaw| degrees above which candidates are surfaced even for recognised faces
    # (guards against off-axis false accepts).
    candidate_yaw_threshold: float = 30.0

    min_face_size: int = 20
    detection_confidence: float = 0.50
    max_faces_per_frame: int = 3

    # ── Performance ───────────────────────────────────────────────────────────
    frame_skip_interval: int = 2          # process every Nth frame (1 = every frame)
    max_detection_width: int = 640        # resize frames before detection
    max_detection_height: int = 640

    # ── Latency control ──────────────────────────────────────────────────────
    max_processing_time_ms: int = 200     # skip frame if exceeded

    # ── Rate limiting ────────────────────────────────────────────────────────
    api_rate_limit: int = 30              # max requests per minute per endpoint

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

    @field_validator("candidate_min_sim")
    @classmethod
    def _candidate_sim_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("candidate_min_sim must be between 0 and 1")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
