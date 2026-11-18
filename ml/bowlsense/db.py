"""Persistence layer.

``SQLiteDatabase`` mirrors the Supabase tables in ``supabase/migrations`` (without RLS) so the
ML service, worker and tests run fully offline. Select the backend with ``BOWLSENSE_DB``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from bowlsense.config import EnvSettings, load_env
from bowlsense.errors import ConfigError, DatabaseError, NotFoundError

SessionStatus = Literal["queued", "processing", "done", "failed"]
JobStatus = Literal["queued", "processing", "done", "failed"]
SESSION_STATUSES: tuple[str, ...] = ("queued", "processing", "done", "failed")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    user_id TEXT NOT NULL UNIQUE,
    display_name TEXT,
    role TEXT NOT NULL CHECK (role IN ('player', 'coach'))
);
CREATE TABLE IF NOT EXISTS bowlers (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    owner_id TEXT,
    name TEXT NOT NULL,
    height_cm REAL,
    bowling_arm TEXT CHECK (bowling_arm IN ('right', 'left')),
    bowling_type TEXT,
    dob TEXT
);
CREATE TABLE IF NOT EXISTS bowler_shares (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    bowler_id TEXT NOT NULL REFERENCES bowlers(id) ON DELETE CASCADE,
    coach_id TEXT NOT NULL,
    UNIQUE (bowler_id, coach_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    bowler_id TEXT REFERENCES bowlers(id) ON DELETE CASCADE,
    recorded_at TEXT,
    video_path TEXT NOT NULL,
    fps REAL,
    view TEXT,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    error TEXT,
    calibration TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'processing', 'done', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    locked_at TEXT,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (status, created_at);
CREATE TABLE IF NOT EXISTS deliveries (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    phases TEXT,
    action_type TEXT,
    speed_kmh REAL,
    length TEXT,
    line TEXT,
    pitch_x REAL,
    pitch_y REAL,
    technique_score REAL,
    injury_overall REAL,
    injury_band TEXT,
    legality_flagged INTEGER,
    elbow_extension_deg REAL,
    UNIQUE (session_id, idx)
);
CREATE TABLE IF NOT EXISTS delivery_metrics (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    delivery_id TEXT NOT NULL REFERENCES deliveries(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    "values" TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS faults (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    delivery_id TEXT NOT NULL REFERENCES deliveries(id) ON DELETE CASCADE,
    code TEXT NOT NULL,
    severity REAL NOT NULL,
    value REAL,
    threshold REAL,
    frames TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_summaries (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    consistency REAL,
    fatigue TEXT,
    workload TEXT,
    similar TEXT,
    coach_report TEXT,
    pdf_path TEXT,
    overlay_path TEXT
);
CREATE TABLE IF NOT EXISTS coach_messages (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL
);
"""

TABLES: tuple[str, ...] = (
    "profiles",
    "bowlers",
    "bowler_shares",
    "sessions",
    "jobs",
    "deliveries",
    "delivery_metrics",
    "faults",
    "session_summaries",
    "coach_messages",
)


class _Record(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class BowlerRecord(_Record):
    """A row of ``bowlers``."""

    id: str
    created_at: str
    owner_id: str | None
    name: str
    height_cm: float | None
    bowling_arm: Literal["right", "left"] | None
    bowling_type: str | None
    dob: str | None


class SessionRecord(_Record):
    """A row of ``sessions`` (``calibration`` decoded from JSON)."""

    id: str
    created_at: str
    bowler_id: str | None
    recorded_at: str | None
    video_path: str
    fps: float | None
    view: str | None
    status: SessionStatus
    error: str | None
    calibration: list[list[float]] | None


class JobRecord(_Record):
    """A row of ``jobs``."""

    id: str
    created_at: str
    session_id: str
    status: JobStatus
    attempts: int
    locked_at: str | None
    last_error: str | None


class CoachMessageRecord(_Record):
    """A row of ``coach_messages``."""

    id: str
    created_at: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _new_id() -> str:
    return str(uuid.uuid4())


class SQLiteDatabase:
    """SQLite implementation of the BowlSense store. Thread-safe via a single lock."""

    def __init__(self, path: Path | str) -> None:
        """Open (and create if needed) the database at ``path``; ``":memory:"`` is allowed."""
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not open SQLite database at {self.path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Run statements in one IMMEDIATE transaction; roll back on error."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                self._conn.execute("ROLLBACK")
                raise DatabaseError(f"Database operation failed: {exc}") from exc
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def _one(self, sql: str, params: tuple[Any, ...]) -> sqlite3.Row | None:
        with self._lock:
            row: sqlite3.Row | None = self._conn.execute(sql, params).fetchone()
            return row

    def table_names(self) -> list[str]:
        """Return the user tables present in the database."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [r["name"] for r in rows]

    # --- bowlers ---------------------------------------------------------------------------

    def create_bowler(
        self,
        name: str,
        *,
        owner_id: str | None = None,
        height_cm: float | None = None,
        bowling_arm: Literal["right", "left"] | None = None,
        bowling_type: str | None = None,
        dob: str | None = None,
    ) -> BowlerRecord:
        """Insert a bowler and return it."""
        if not name.strip():
            raise DatabaseError("Bowler name must not be empty.")
        bid = _new_id()
        with self._tx() as c:
            c.execute(
                "INSERT INTO bowlers (id, created_at, owner_id, name, height_cm, bowling_arm,"
                " bowling_type, dob) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    bid,
                    _now(),
                    owner_id,
                    name.strip(),
                    height_cm,
                    bowling_arm,
                    bowling_type,
                    dob,
                ),
            )
        return self.get_bowler(bid)

    def get_bowler(self, bowler_id: str) -> BowlerRecord:
        """Return a bowler or raise ``NotFoundError``."""
        row = self._one("SELECT * FROM bowlers WHERE id = ?", (bowler_id,))
        if row is None:
            raise NotFoundError("bowler", bowler_id)
        return BowlerRecord.model_validate(dict(row))

    # --- sessions --------------------------------------------------------------------------

    def create_session(
        self,
        video_path: str,
        *,
        bowler_id: str | None = None,
        fps: float | None = None,
        view: str | None = None,
        recorded_at: str | None = None,
    ) -> SessionRecord:
        """Insert a session in ``queued`` status and return it."""
        sid = _new_id()
        with self._tx() as c:
            c.execute(
                "INSERT INTO sessions (id, created_at, bowler_id, recorded_at, video_path, fps,"
                " view, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued')",
                (sid, _now(), bowler_id, recorded_at, video_path, fps, view),
            )
        return self.get_session(sid)

    def get_session(self, session_id: str) -> SessionRecord:
        """Return a session or raise ``NotFoundError``."""
        row = self._one("SELECT * FROM sessions WHERE id = ?", (session_id,))
        if row is None:
            raise NotFoundError("session", session_id)
        data = dict(row)
        data["calibration"] = json.loads(data["calibration"]) if data["calibration"] else None
        return SessionRecord.model_validate(data)

    def set_session_status(
        self, session_id: str, status: SessionStatus, error: str | None = None
    ) -> SessionRecord:
        """Update a session's status (and error text) and return it."""
        if status not in SESSION_STATUSES:
            raise DatabaseError(f"Invalid session status '{status}'.")
        with self._tx() as c:
            cur = c.execute(
                "UPDATE sessions SET status = ?, error = ? WHERE id = ?",
                (status, error, session_id),
            )
            if cur.rowcount == 0:
                raise NotFoundError("session", session_id)
        return self.get_session(session_id)

    def save_calibration(self, session_id: str, points: list[list[float]]) -> SessionRecord:
        """Store four pitch calibration points ``[[x, y], ...]`` in image pixels."""
        if len(points) != 4 or any(len(p) != 2 for p in points):
            raise DatabaseError("Calibration needs exactly 4 points, each [x, y].")
        with self._tx() as c:
            cur = c.execute(
                "UPDATE sessions SET calibration = ? WHERE id = ?",
                (json.dumps(points), session_id),
            )
            if cur.rowcount == 0:
                raise NotFoundError("session", session_id)
        return self.get_session(session_id)

    # --- jobs ------------------------------------------------------------------------------

    def create_job(self, session_id: str) -> JobRecord:
        """Queue a processing job for a session."""
        self.get_session(session_id)
        jid = _new_id()
        with self._tx() as c:
            c.execute(
                "INSERT INTO jobs (id, created_at, session_id, status, attempts)"
                " VALUES (?, ?, ?, 'queued', 0)",
                (jid, _now(), session_id),
            )
        return self.get_job(jid)

    def get_job(self, job_id: str) -> JobRecord:
        """Return a job or raise ``NotFoundError``."""
        row = self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise NotFoundError("job", job_id)
        return JobRecord.model_validate(dict(row))

    def claim_next_job(self) -> JobRecord | None:
        """Atomically lock the oldest queued job, mark it processing and return it."""
        with self._tx() as c:
            row = c.execute(
                "UPDATE jobs SET status = 'processing', locked_at = ?, attempts = attempts + 1"
                " WHERE id = (SELECT id FROM jobs WHERE status = 'queued'"
                " ORDER BY created_at LIMIT 1) RETURNING *",
                (_now(),),
            ).fetchone()
            if row is None:
                return None
            c.execute(
                "UPDATE sessions SET status = 'processing', error = NULL WHERE id = ?",
                (row["session_id"],),
            )
        return JobRecord.model_validate(dict(row))

    def complete_job(self, job_id: str) -> JobRecord:
        """Mark a job and its session done."""
        job = self.get_job(job_id)
        with self._tx() as c:
            c.execute(
                "UPDATE jobs SET status = 'done', last_error = NULL WHERE id = ?",
                (job_id,),
            )
            c.execute(
                "UPDATE sessions SET status = 'done', error = NULL WHERE id = ?",
                (job.session_id,),
            )
        return self.get_job(job_id)

    def fail_job(self, job_id: str, error: str, max_attempts: int) -> JobRecord:
        """Record a failure. Requeue if attempts remain, otherwise mark job and session failed."""
        job = self.get_job(job_id)
        final = job.attempts >= max_attempts
        status = "failed" if final else "queued"
        with self._tx() as c:
            c.execute(
                "UPDATE jobs SET status = ?, last_error = ?, locked_at = NULL WHERE id = ?",
                (status, error, job_id),
            )
            c.execute(
                "UPDATE sessions SET status = ?, error = ? WHERE id = ?",
                (
                    "failed" if final else "queued",
                    error if final else None,
                    job.session_id,
                ),
            )
        return self.get_job(job_id)

    # --- coach messages --------------------------------------------------------------------

    def add_coach_message(
        self, session_id: str, role: Literal["user", "assistant"], content: str
    ) -> CoachMessageRecord:
        """Append a chat message to a session."""
        self.get_session(session_id)
        mid = _new_id()
        with self._tx() as c:
            c.execute(
                "INSERT INTO coach_messages (id, created_at, session_id, role, content)"
                " VALUES (?, ?, ?, ?, ?)",
                (mid, _now(), session_id, role, content),
            )
        row = self._one("SELECT * FROM coach_messages WHERE id = ?", (mid,))
        return CoachMessageRecord.model_validate(dict(row) if row else {})

    def list_coach_messages(self, session_id: str) -> list[CoachMessageRecord]:
        """Return a session's chat messages, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM coach_messages WHERE session_id = ? ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
        return [CoachMessageRecord.model_validate(dict(r)) for r in rows]


def get_database(env: EnvSettings | None = None) -> SQLiteDatabase:
    """Return the database selected by ``BOWLSENSE_DB``.

    Raises:
        ConfigError: if the Supabase backend is selected; it is not available until the
            Supabase integration phase lands. Use ``BOWLSENSE_DB=sqlite`` meanwhile.
    """
    env = env or load_env()
    if env.db_backend == "supabase":
        raise ConfigError(
            "BOWLSENSE_DB=supabase is not available yet in this build. "
            "Set BOWLSENSE_DB=sqlite to use the local database."
        )
    return SQLiteDatabase(env.sqlite_path)
