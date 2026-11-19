"""Database operations used by the API, worker and CLI. Returns pydantic records, not ORM rows."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict
from sqlalchemy import Engine, delete, literal_column, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as OrmSession

from bowlsense.db import models as m
from bowlsense.db.session import immediate, make_sessionmaker
from bowlsense.errors import DatabaseError, NotFoundError
from bowlsense.storage import LocalStorage

Status = Literal["queued", "processing", "done", "failed"]


def _as_utc(value: datetime) -> datetime:
    """SQLite drops tzinfo; every stored time is UTC, so reattach it."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class _Record(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)


class BowlerRecord(_Record):
    """A bowler profile."""

    id: str
    created_at: UtcDatetime
    name: str
    height_cm: float | None
    bowling_arm: Literal["right", "left"] | None
    bowling_type: str | None
    dob: str | None


class SessionRecord(_Record):
    """A session (``video_path`` is relative to the storage root)."""

    id: str
    created_at: UtcDatetime
    bowler_id: str | None
    recorded_at: UtcDatetime | None
    video_path: str
    fps: float | None
    view: str | None
    status: Status
    error: str | None
    calibration: list[list[float]] | None


class JobRecord(_Record):
    """A pipeline job."""

    id: str
    created_at: UtcDatetime
    session_id: str
    status: Status
    attempts: int
    locked_at: UtcDatetime | None
    last_error: str | None
    progress: float
    stage: str | None


class CoachMessageRecord(_Record):
    """A coach chat message."""

    id: str
    created_at: UtcDatetime
    session_id: str
    role: Literal["user", "assistant"]
    content: str


class Repository:
    """All database access goes through here."""

    def __init__(self, engine: Engine) -> None:
        """Bind to a (migrated) engine."""
        self.engine = engine
        self._sessions = make_sessionmaker(engine)
        self._immediate_sessions = make_sessionmaker(immediate(engine))

    @contextmanager
    def _tx(self, *, write: bool = False) -> Iterator[OrmSession]:
        """Open a session in one transaction; writes take the lock up front."""
        factory = self._immediate_sessions if write else self._sessions
        try:
            with factory() as s, s.begin():
                yield s
        except IntegrityError as exc:
            raise DatabaseError(f"Constraint violated: {exc.orig}") from exc
        except SQLAlchemyError as exc:
            raise DatabaseError(f"Database operation failed: {exc}") from exc

    @staticmethod
    def _get(s: OrmSession, model: type[Any], table: str, record_id: str) -> Any:
        row = s.get(model, record_id)
        if row is None:
            raise NotFoundError(table, record_id)
        return row

    # --- bowlers ---------------------------------------------------------------------------

    def create_bowler(
        self,
        name: str,
        *,
        height_cm: float | None = None,
        bowling_arm: Literal["right", "left"] | None = None,
        bowling_type: str | None = None,
        dob: str | None = None,
    ) -> BowlerRecord:
        """Create a bowler profile."""
        if not name.strip():
            raise DatabaseError("Bowler name must not be empty.")
        with self._tx(write=True) as s:
            row = m.Bowler(
                name=name.strip(),
                height_cm=height_cm,
                bowling_arm=bowling_arm,
                bowling_type=bowling_type,
                dob=dob,
            )
            s.add(row)
            s.flush()
            return BowlerRecord.model_validate(row)

    def get_bowler(self, bowler_id: str) -> BowlerRecord:
        """Return a bowler or raise ``NotFoundError``."""
        with self._tx() as s:
            return BowlerRecord.model_validate(self._get(s, m.Bowler, "bowler", bowler_id))

    def list_bowlers(self) -> list[BowlerRecord]:
        """Return all bowlers by name."""
        with self._tx() as s:
            rows = s.scalars(select(m.Bowler).order_by(m.Bowler.name)).all()
            return [BowlerRecord.model_validate(r) for r in rows]

    # --- sessions --------------------------------------------------------------------------

    def create_session(
        self,
        video_path: str,
        *,
        bowler_id: str | None = None,
        fps: float | None = None,
        view: str | None = None,
        recorded_at: datetime | None = None,
    ) -> SessionRecord:
        """Create a queued session for a stored file (``video_path`` relative to storage)."""
        with self._tx(write=True) as s:
            if bowler_id is not None:
                self._get(s, m.Bowler, "bowler", bowler_id)
            row = m.Session(
                video_path=video_path,
                bowler_id=bowler_id,
                fps=fps,
                view=view,
                recorded_at=recorded_at,
                status="queued",
            )
            s.add(row)
            s.flush()
            return SessionRecord.model_validate(row)

    def get_session(self, session_id: str) -> SessionRecord:
        """Return a session or raise ``NotFoundError``."""
        with self._tx() as s:
            return SessionRecord.model_validate(self._get(s, m.Session, "session", session_id))

    def list_sessions(self, bowler_id: str | None = None) -> list[SessionRecord]:
        """Return sessions, newest first, optionally for one bowler."""
        with self._tx() as s:
            q = select(m.Session).order_by(m.Session.created_at.desc())
            if bowler_id is not None:
                q = q.where(m.Session.bowler_id == bowler_id)
            return [SessionRecord.model_validate(r) for r in s.scalars(q).all()]

    def set_session_status(
        self, session_id: str, status: Status, error: str | None = None
    ) -> SessionRecord:
        """Set a session's status and error text."""
        if status not in m.STATUSES:
            raise DatabaseError(f"Invalid session status '{status}'.")
        with self._tx(write=True) as s:
            row = self._get(s, m.Session, "session", session_id)
            row.status, row.error = status, error
            s.flush()
            return SessionRecord.model_validate(row)

    def save_calibration(self, session_id: str, points: list[list[float]]) -> SessionRecord:
        """Store four pitch calibration points ``[[x, y], ...]`` in image pixels."""
        if len(points) != 4 or any(len(p) != 2 for p in points):
            raise DatabaseError("Calibration needs exactly 4 points, each [x, y].")
        with self._tx(write=True) as s:
            row = self._get(s, m.Session, "session", session_id)
            row.calibration = [[float(x), float(y)] for x, y in points]
            s.flush()
            return SessionRecord.model_validate(row)

    def delete_session(self, session_id: str, storage: LocalStorage) -> None:
        """Delete a session, all its rows (cascade) and its stored files."""
        with self._tx(write=True) as s:
            row = self._get(s, m.Session, "session", session_id)
            video_path = row.video_path
            s.execute(delete(m.Session).where(m.Session.id == session_id))
        storage.delete_session_files(session_id, video_path)

    def delete_all_data(self, storage: LocalStorage) -> None:
        """Delete every row in every table and every stored file."""
        with self._tx(write=True) as s:
            for table in reversed(m.Base.metadata.sorted_tables):
                s.execute(table.delete())
        storage.wipe()

    # --- jobs ------------------------------------------------------------------------------

    def create_job(self, session_id: str) -> JobRecord:
        """Queue a pipeline job for a session."""
        with self._tx(write=True) as s:
            self._get(s, m.Session, "session", session_id)
            row = m.Job(session_id=session_id, status="queued", attempts=0, progress=0.0)
            s.add(row)
            s.flush()
            return JobRecord.model_validate(row)

    def get_job(self, job_id: str) -> JobRecord:
        """Return a job or raise ``NotFoundError``."""
        with self._tx() as s:
            return JobRecord.model_validate(self._get(s, m.Job, "job", job_id))

    def latest_job_for_session(self, session_id: str) -> JobRecord | None:
        """Return the newest job of a session, if any."""
        with self._tx() as s:
            row = s.scalars(
                select(m.Job)
                .where(m.Job.session_id == session_id)
                .order_by(m.Job.created_at.desc())
                .limit(1)
            ).first()
            return JobRecord.model_validate(row) if row else None

    def claim_next_job(self) -> JobRecord | None:
        """Atomically claim the oldest queued job (``BEGIN IMMEDIATE`` + ``UPDATE…RETURNING``)."""
        with self._tx(write=True) as s:
            job_id = s.execute(
                text(
                    "UPDATE jobs SET status = 'processing', locked_at = :now,"
                    " attempts = attempts + 1, progress = 0, stage = 'starting'"
                    " WHERE id = (SELECT id FROM jobs WHERE status = 'queued'"
                    " ORDER BY created_at, rowid LIMIT 1) RETURNING id"
                ),
                {"now": m.utcnow()},
            ).scalar_one_or_none()
            if job_id is None:
                return None
            row = self._get(s, m.Job, "job", job_id)
            s.refresh(row)
            session = self._get(s, m.Session, "session", row.session_id)
            session.status, session.error = "processing", None
            s.flush()
            return JobRecord.model_validate(row)

    def update_progress(self, job_id: str, progress: float, stage: str) -> JobRecord:
        """Record pipeline progress (0-100) and the current stage name for live status."""
        with self._tx(write=True) as s:
            row = self._get(s, m.Job, "job", job_id)
            row.progress, row.stage = max(0.0, min(100.0, progress)), stage
            s.flush()
            return JobRecord.model_validate(row)

    def complete_job(self, job_id: str) -> JobRecord:
        """Mark a job and its session done."""
        with self._tx(write=True) as s:
            row = self._get(s, m.Job, "job", job_id)
            row.status, row.last_error, row.progress, row.stage = "done", None, 100.0, "done"
            session = self._get(s, m.Session, "session", row.session_id)
            session.status, session.error = "done", None
            s.flush()
            return JobRecord.model_validate(row)

    def fail_job(self, job_id: str, error: str, max_attempts: int) -> JobRecord:
        """Record a failure; requeue while attempts remain, else mark job and session failed."""
        with self._tx(write=True) as s:
            row = self._get(s, m.Job, "job", job_id)
            final = row.attempts >= max_attempts
            row.status = "failed" if final else "queued"
            row.last_error, row.locked_at = error, None
            session = self._get(s, m.Session, "session", row.session_id)
            session.status = "failed" if final else "queued"
            session.error = error if final else None
            s.flush()
            return JobRecord.model_validate(row)

    # --- coach messages --------------------------------------------------------------------

    def add_coach_message(
        self, session_id: str, role: Literal["user", "assistant"], content: str
    ) -> CoachMessageRecord:
        """Append a coach chat message."""
        with self._tx(write=True) as s:
            self._get(s, m.Session, "session", session_id)
            row = m.CoachMessage(session_id=session_id, role=role, content=content)
            s.add(row)
            s.flush()
            return CoachMessageRecord.model_validate(row)

    def list_coach_messages(self, session_id: str) -> list[CoachMessageRecord]:
        """Return a session's chat, oldest first."""
        with self._tx() as s:
            rows = s.scalars(
                select(m.CoachMessage)
                .where(m.CoachMessage.session_id == session_id)
                .order_by(m.CoachMessage.created_at, literal_column("coach_messages.rowid"))
            ).all()
            return [CoachMessageRecord.model_validate(r) for r in rows]

    def count_rows(self, table: str) -> int:
        """Return the number of rows in ``table`` (used by tests and the settings page)."""
        if table not in m.TABLE_NAMES:
            raise DatabaseError(f"Unknown table '{table}'.")
        with self._tx() as s:
            return int(s.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one())
