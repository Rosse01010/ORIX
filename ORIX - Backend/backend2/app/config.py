"""
Configuración central del sistema.
Todas las variables se leen desde el entorno o un archivo .env
"""
from __future__ import annotations
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── PostgreSQL ──────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://surv:surv_pass@postgres:5432/surveillance"
    DATABASE_SYNC_URL: str = "postgresql://surv:surv_pass@postgres:5432/surveillance"

    # ── Redis ───────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379"
    REDIS_STREAM_KEY: str = "face:embeddings"       # Redis Stream (GPU → DB worker)
    REDIS_ALERT_CHANNEL: str = "face:alerts"         # Pub/Sub  (DB worker → API)
    REDIS_CONSUMER_GROUP: str = "db-workers"

    # ── Cámaras RTSP (separadas por coma) ──────────────────────────────
    RTSP_URLS: str = "rtsp://camera1/stream,rtsp://camera2/stream"

    # ── FaceNet ─────────────────────────────────────────────────────────
    FACENET_MODEL_PATH: str = "/app/models/facenet_model.h5"
    EMBEDDING_DIM: int = 512

    # ── Reconocimiento ──────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.6   # distancia coseno máxima para match
    GPU_BATCH_SIZE: int = 16
    GPU_BATCH_TIMEOUT_MS: int = 50      # ms máximo esperando llenar batch

    # ── Frames ──────────────────────────────────────────────────────────
    FRAME_SKIP: int = 5                 # procesar 1 de cada N frames
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480

    # ── API ─────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"

    def get_rtsp_list(self) -> List[str]:
        return [u.strip() for u in self.RTSP_URLS.split(",") if u.strip()]


settings = Settings()
