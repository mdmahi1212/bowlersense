"""Tests for the SQLite fallback database."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from bowlsense.config import load_env
from bowlsense.db import TABLES, SQLiteDatabase, get_database
from bowlsense.errors import ConfigError, DatabaseError, NotFoundError


def test_all_tables_created(db: SQLiteDatabase) -> None:
    assert set(TABLES) <= set(db.table_names())


def test_memory_database() -> None:
    mem = SQLiteDatabase(":memory:")
    assert "jobs" in mem.table_names()
    mem.close()


def test_bowler_crud(db: SQLiteDatabase) -> None:
    b = db.create_bowler(" Test Bowler ", height_cm=180, bowling_arm="left")
    assert b.name == "Test Bowler"
    assert db.get_bowler(b.id) == b
    with pytest.raises(NotFoundError):
        db.get_bowler("missing")
    with pytest.raises(DatabaseError):
        db.create_bowler("  ")


def test_session_lifecycle(db: SQLiteDatabase) -> None:
    b = db.create_bowler("A")
    s = db.create_session("videos/a.mp4", bowler_id=b.id, fps=240)
    assert s.status == "queued" and s.calibration is None
    s = db.save_calibration(s.id, [[0, 0], [1, 0], [1, 1], [0, 1]])
    assert s.calibration == [[0, 0], [1, 0], [1, 1], [0, 1]]
    with pytest.raises(DatabaseError, match="4 points"):
        db.save_calibration(s.id, [[0, 0]])
    s = db.set_session_status(s.id, "failed", "boom")
    assert s.status == "failed" and s.error == "boom"
    with pytest.raises(NotFoundError):
        db.set_session_status("missing", "done")
    with pytest.raises(NotFoundError):
        db.save_calibration("missing", [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_foreign_key_enforced(db: SQLiteDatabase) -> None:
    with pytest.raises(DatabaseError):
        db.create_session("v.mp4", bowler_id="no-such-bowler")


def test_job_queue_success(db: SQLiteDatabase) -> None:
    s1 = db.create_session("1.mp4")
    s2 = db.create_session("2.mp4")
    j1 = db.create_job(s1.id)
    db.create_job(s2.id)
    claimed = db.claim_next_job()
    assert claimed is not None and claimed.id == j1.id
    assert claimed.status == "processing" and claimed.attempts == 1
    assert db.get_session(s1.id).status == "processing"
    done = db.complete_job(claimed.id)
    assert done.status == "done"
    assert db.get_session(s1.id).status == "done"
    second = db.claim_next_job()
    assert second is not None and second.session_id == s2.id
    assert db.claim_next_job() is None


def test_job_retry_then_fail(db: SQLiteDatabase) -> None:
    s = db.create_session("x.mp4")
    db.create_job(s.id)
    for attempt in (1, 2):
        job = db.claim_next_job()
        assert job is not None and job.attempts == attempt
        job = db.fail_job(job.id, f"err {attempt}", max_attempts=3)
        assert job.status == "queued" and job.last_error == f"err {attempt}"
        assert db.get_session(s.id).status == "queued"
    job = db.claim_next_job()
    assert job is not None and job.attempts == 3
    job = db.fail_job(job.id, "final", max_attempts=3)
    assert job.status == "failed"
    sess = db.get_session(s.id)
    assert sess.status == "failed" and sess.error == "final"
    assert db.claim_next_job() is None


def test_create_job_needs_session(db: SQLiteDatabase) -> None:
    with pytest.raises(NotFoundError):
        db.create_job("missing")


def test_concurrent_claims_never_duplicate(tmp_path: Path) -> None:
    db = SQLiteDatabase(tmp_path / "c.db")
    for i in range(20):
        db.create_job(db.create_session(f"{i}.mp4").id)
    claimed: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        while (job := db.claim_next_job()) is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == 20 == len(set(claimed))
    db.close()


def test_coach_messages(db: SQLiteDatabase) -> None:
    s = db.create_session("v.mp4")
    db.add_coach_message(s.id, "user", "How was my action?")
    db.add_coach_message(s.id, "assistant", "Solid.")
    msgs = db.list_coach_messages(s.id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    with pytest.raises(NotFoundError):
        db.add_coach_message("missing", "user", "hi")


def test_get_database_sqlite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOWLSENSE_DB", "sqlite")
    monkeypatch.setenv("BOWLSENSE_SQLITE_PATH", str(tmp_path / "sub" / "x.db"))
    database = get_database(load_env())
    assert (tmp_path / "sub" / "x.db").is_file()
    database.close()


def test_get_database_supabase_not_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOWLSENSE_DB", "supabase")
    with pytest.raises(ConfigError, match="sqlite"):
        get_database(load_env())
