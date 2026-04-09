"""
Modelos ORM.
  Person          → personas registradas con su embedding facial.
  DetectionEvent  → log de cada detección: cámara, persona, score, timestamp.
"""
import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, Float, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.database import Base
from app.config import settings


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding: Mapped[list] = mapped_column(
        Vector(settings.EMBEDDING_DIM), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    events: Mapped[list["DetectionEvent"]] = relationship(back_populates="person")

    def __repr__(self) -> str:
        return f"<Person id={self.id} name={self.name}>"


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    person_id: Mapped[str | None] = mapped_column(
        ForeignKey("persons.id"), nullable=True
    )
    camera_id: Mapped[str] = mapped_column(String(100), nullable=False)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="unknown"
    )  # "matched" | "unknown"
    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    person: Mapped["Person | None"] = relationship(back_populates="events")
