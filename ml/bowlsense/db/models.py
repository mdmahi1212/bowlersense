"""SQLAlchemy ORM tables (spec Section 8). Local single-user app: no auth, no RLS.

The Alembic migrations in ``db/migrations`` must stay in sync with these models; a test
compares them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

STATUSES: tuple[str, ...] = ("queued", "processing", "done", "failed")
_STATUS_CHECK = "status IN ('queued', 'processing', 'done', 'failed')"


def new_id() -> str:
    """Return a new UUID4 string primary key."""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Return the current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base: every table has a UUID string ``id`` and a UTC ``created_at``."""

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )


class Bowler(Base):
    """A bowler profile (a local record, no account)."""

    __tablename__ = "bowlers"
    __table_args__ = (CheckConstraint("bowling_arm IN ('right', 'left')", name="ck_bowlers_arm"),)

    name: Mapped[str] = mapped_column(String(200), nullable=False)
    height_cm: Mapped[float | None] = mapped_column(Float)
    bowling_arm: Mapped[str | None] = mapped_column(String(5))
    bowling_type: Mapped[str | None] = mapped_column(String(50))
    dob: Mapped[str | None] = mapped_column(String(10))

    sessions: Mapped[list[Session]] = relationship(
        back_populates="bowler", cascade="all, delete-orphan", passive_deletes=True
    )


class Session(Base):
    """One uploaded video (or image) and its processing status."""

    __tablename__ = "sessions"
    __table_args__ = (CheckConstraint(_STATUS_CHECK, name="ck_sessions_status"),)

    bowler_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("bowlers.id", ondelete="CASCADE"), index=True
    )
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    video_path: Mapped[str] = mapped_column(Text, nullable=False)  # relative to storage root
    fps: Mapped[float | None] = mapped_column(Float)
    view: Mapped[str | None] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="queued")
    error: Mapped[str | None] = mapped_column(Text)
    calibration: Mapped[list[Any] | None] = mapped_column(JSON)

    bowler: Mapped[Bowler | None] = relationship(back_populates="sessions")
    jobs: Mapped[list[Job]] = relationship(cascade="all, delete-orphan", passive_deletes=True)
    deliveries: Mapped[list[Delivery]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )


class Job(Base):
    """A queued pipeline run for a session, claimed by the worker."""

    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(_STATUS_CHECK, name="ck_jobs_status"),
        Index("ix_jobs_status_created", "status", "created_at"),
    )

    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(12), nullable=False, default="queued")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stage: Mapped[str | None] = mapped_column(String(50))


class Delivery(Base):
    """Per-delivery summary metrics."""

    __tablename__ = "deliveries"
    __table_args__ = (UniqueConstraint("session_id", "idx", name="uq_deliveries_session_idx"),)

    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    phases: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    action_type: Mapped[str | None] = mapped_column(String(20))
    speed_kmh: Mapped[float | None] = mapped_column(Float)
    length: Mapped[str | None] = mapped_column(String(20))
    line: Mapped[str | None] = mapped_column(String(20))
    pitch_x: Mapped[float | None] = mapped_column(Float)
    pitch_y: Mapped[float | None] = mapped_column(Float)
    technique_score: Mapped[float | None] = mapped_column(Float)
    injury_overall: Mapped[float | None] = mapped_column(Float)
    injury_band: Mapped[str | None] = mapped_column(String(10))
    legality_flagged: Mapped[bool | None] = mapped_column(Boolean)
    elbow_extension_deg: Mapped[float | None] = mapped_column(Float)

    metrics: Mapped[list[DeliveryMetric]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )
    faults: Mapped[list[FaultRow]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )


class DeliveryMetric(Base):
    """An angle series for a delivery."""

    __tablename__ = "delivery_metrics"

    delivery_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("deliveries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    values: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class FaultRow(Base):
    """A technique fault detected in a delivery."""

    __tablename__ = "faults"

    delivery_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("deliveries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    code: Mapped[str] = mapped_column(String(30), nullable=False)
    severity: Mapped[float] = mapped_column(Float, nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    frames: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)


class SessionSummary(Base):
    """Session-level results and output file paths (relative to storage root)."""

    __tablename__ = "session_summaries"

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    consistency: Mapped[float | None] = mapped_column(Float)
    fatigue: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    workload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    similar: Mapped[list[Any] | None] = mapped_column(JSON)
    coach_report: Mapped[str | None] = mapped_column(Text)
    pdf_path: Mapped[str | None] = mapped_column(Text)
    overlay_path: Mapped[str | None] = mapped_column(Text)


class CoachMessage(Base):
    """One message in a session's coach chat."""

    __tablename__ = "coach_messages"
    __table_args__ = (CheckConstraint("role IN ('user', 'assistant')", name="ck_coach_role"),)

    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)


TABLE_NAMES: tuple[str, ...] = tuple(sorted(Base.metadata.tables))
