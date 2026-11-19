"""Tests for the SQLAlchemy/Alembic SQLite database and repository."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
from datetime import UTC
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect, text

from bowlsense.config import Settings
from bowlsense.db import open_repository
from bowlsense.db.models import TABLE_NAMES, Base
from bowlsense.db.repo import Repository
from bowlsense.db.session import get_engine, make_engine, upgrade
from bowlsense.errors import DatabaseError, NotFoundError
from bowlsense.storage import LocalStorage

ML_DIR = Path(__file__).resolve().parents[1]


def test_upgrade_head_on_empty_db(tmp_path: Path) -> None:
    engine = make_engine(tmp_path / "sub" / "fresh.db")
    upgrade(engine)
    tables = set(inspect(engine).get_table_names())
    assert set(TABLE_NAMES) <= tables
    assert "alembic_version" in tables
    upgrade(engine)  # idempotent
    engine.dispose()


def test_migrations_match_models(repo: Repository) -> None:
    with repo.engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []


def test_downgrade_to_base(tmp_path: Path) -> None:
    from alembic import command

    from bowlsense.db.session import alembic_config

    engine = make_engine(tmp_path / "d.db")
    upgrade(engine)
    command.downgrade(alembic_config(engine), "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    engine.dispose()


def test_pragmas(repo: Repository) -> None:
    with repo.engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"


def test_no_auth_tables() -> None:
    assert "profiles" not in TABLE_NAMES
    assert "bowler_shares" not in TABLE_NAMES


def test_bowlers(repo: Repository) -> None:
    b = repo.create_bowler(" Zed ", height_cm=180, bowling_arm="left")
    assert b.name == "Zed" and len(b.id) == 36
    assert b.created_at.tzinfo is not None
    repo.create_bowler("Amy")
    assert [x.name for x in repo.list_bowlers()] == ["Amy", "Zed"]
    assert repo.get_bowler(b.id).created_at.tzinfo == UTC
    with pytest.raises(NotFoundError):
        repo.get_bowler("missing")
    with pytest.raises(DatabaseError):
        repo.create_bowler(" ")
    with pytest.raises(DatabaseError, match="Constraint"):
        repo.create_bowler("Bad", bowling_arm="both")  # type: ignore[arg-type]


def test_sessions(repo: Repository) -> None:
    b = repo.create_bowler("A")
    s = repo.create_session("raw/a.mp4", bowler_id=b.id, fps=240)
    assert s.status == "queued" and s.calibration is None
    s = repo.save_calibration(s.id, [[0, 0], [1, 0], [1, 1], [0, 1]])
    assert s.calibration == [[0, 0], [1, 0], [1, 1], [0, 1]]
    with pytest.raises(DatabaseError, match="4 points"):
        repo.save_calibration(s.id, [[0, 0]])
    s = repo.set_session_status(s.id, "failed", "boom")
    assert (s.status, s.error) == ("failed", "boom")
    with pytest.raises(DatabaseError, match="Invalid"):
        repo.set_session_status(s.id, "nope")  # type: ignore[arg-type]
    other = repo.create_session("raw/b.mp4")
    assert [x.id for x in repo.list_sessions(bowler_id=b.id)] == [s.id]
    assert {x.id for x in repo.list_sessions()} == {s.id, other.id}
    with pytest.raises(NotFoundError):
        repo.create_session("raw/c.mp4", bowler_id="missing")


def test_job_queue_success(repo: Repository) -> None:
    s1, s2 = repo.create_session("raw/1.mp4"), repo.create_session("raw/2.mp4")
    j1 = repo.create_job(s1.id)
    repo.create_job(s2.id)
    claimed = repo.claim_next_job()
    assert claimed is not None and claimed.id == j1.id
    assert (claimed.status, claimed.attempts, claimed.stage) == ("processing", 1, "starting")
    assert repo.get_session(s1.id).status == "processing"
    assert repo.update_progress(claimed.id, 150, "pose").progress == 100.0
    assert repo.update_progress(claimed.id, 42.5, "pose").stage == "pose"
    assert repo.latest_job_for_session(s1.id) is not None
    done = repo.complete_job(claimed.id)
    assert (done.status, done.progress) == ("done", 100.0)
    assert repo.get_session(s1.id).status == "done"
    second = repo.claim_next_job()
    assert second is not None and second.session_id == s2.id
    assert repo.claim_next_job() is None
    assert repo.latest_job_for_session(repo.create_session("raw/3.mp4").id) is None


def test_job_retry_then_fail(repo: Repository, settings: Settings) -> None:
    max_attempts = settings.jobs.max_attempts
    s = repo.create_session("raw/x.mp4")
    repo.create_job(s.id)
    for attempt in range(1, max_attempts):
        job = repo.claim_next_job()
        assert job is not None and job.attempts == attempt
        job = repo.fail_job(job.id, f"err {attempt}", max_attempts)
        assert (job.status, job.last_error) == ("queued", f"err {attempt}")
        assert repo.get_session(s.id).status == "queued"
    job = repo.claim_next_job()
    assert job is not None
    job = repo.fail_job(job.id, "final", max_attempts)
    assert job.status == "failed"
    assert (repo.get_session(s.id).status, repo.get_session(s.id).error) == ("failed", "final")
    assert repo.claim_next_job() is None


def test_create_job_needs_session(repo: Repository) -> None:
    with pytest.raises(NotFoundError):
        repo.create_job("missing")


def test_threads_never_claim_twice(repo: Repository) -> None:
    for i in range(20):
        repo.create_job(repo.create_session(f"raw/{i}.mp4").id)
    claimed: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        while (job := repo.claim_next_job()) is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == 20 == len(set(claimed))


def test_two_processes_write_concurrently(tmp_path: Path) -> None:
    """An 'API' process inserts sessions+jobs while a 'worker' process claims them."""
    db_path = tmp_path / "shared.db"
    engine = make_engine(db_path)
    upgrade(engine)
    engine.dispose()
    n = 40
    api = textwrap.dedent(f"""
        from bowlsense.db.session import make_engine
        from bowlsense.db.repo import Repository
        r = Repository(make_engine({str(db_path)!r}))
        for i in range({n}):
            r.create_job(r.create_session(f"raw/{{i}}.mp4").id)
    """)
    worker = textwrap.dedent(f"""
        import time
        from bowlsense.db.session import make_engine
        from bowlsense.db.repo import Repository
        r = Repository(make_engine({str(db_path)!r}))
        done, deadline = 0, time.monotonic() + 60
        while done < {n} and time.monotonic() < deadline:
            job = r.claim_next_job()
            if job is None:
                time.sleep(0.01)
                continue
            r.update_progress(job.id, 50, "pose")
            r.complete_job(job.id)
            done += 1
        print(done)
    """)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=ML_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for code in (worker, api)
    ]
    outs = [p.communicate(timeout=120) for p in procs]
    for p, (_out, err) in zip(procs, outs, strict=True):
        assert p.returncode == 0, err
        assert "database is locked" not in err
    assert outs[0][0].strip() == str(n)
    check = Repository(make_engine(db_path))
    assert all(s.status == "done" for s in check.list_sessions())
    assert check.count_rows("jobs") == n


def test_coach_messages(repo: Repository) -> None:
    s = repo.create_session("raw/v.mp4")
    repo.add_coach_message(s.id, "user", "How was my action?")
    repo.add_coach_message(s.id, "assistant", "Solid.")
    assert [m.role for m in repo.list_coach_messages(s.id)] == ["user", "assistant"]
    with pytest.raises(NotFoundError):
        repo.add_coach_message("missing", "user", "hi")


def _seed_session(repo: Repository, storage: LocalStorage, bowler_id: str | None = None) -> str:
    raw = storage.root / "raw" / "vid.mp4"
    raw.write_bytes(b"video")
    s = repo.create_session("raw/vid.mp4", bowler_id=bowler_id)
    repo.create_job(s.id)
    repo.add_coach_message(s.id, "user", "hi")
    (storage.session_output_dir(s.id) / "overlay.mp4").write_bytes(b"x")
    return s.id


def test_delete_session_cascades_rows_and_files(repo: Repository, storage: LocalStorage) -> None:
    sid = _seed_session(repo, storage)
    repo.delete_session(sid, storage)
    assert repo.count_rows("sessions") == 0
    assert repo.count_rows("jobs") == 0
    assert repo.count_rows("coach_messages") == 0
    assert not (storage.root / "raw" / "vid.mp4").exists()
    assert not (storage.root / "outputs" / sid).exists()
    with pytest.raises(NotFoundError):
        repo.delete_session(sid, storage)


def test_delete_bowler_cascades_in_sql(repo: Repository, storage: LocalStorage) -> None:
    b = repo.create_bowler("A")
    _seed_session(repo, storage, bowler_id=b.id)
    with repo.engine.begin() as conn:
        conn.execute(text("DELETE FROM bowlers WHERE id = :id"), {"id": b.id})
    assert repo.count_rows("sessions") == 0 and repo.count_rows("jobs") == 0


def test_delete_all_data(repo: Repository, storage: LocalStorage) -> None:
    b = repo.create_bowler("A")
    _seed_session(repo, storage, bowler_id=b.id)
    repo.delete_all_data(storage)
    assert all(repo.count_rows(t) == 0 for t in TABLE_NAMES)
    assert storage.usage_bytes() == 0
    assert (storage.root / "raw").is_dir() and (storage.root / "outputs").is_dir()


def test_count_rows_rejects_unknown_table(repo: Repository) -> None:
    with pytest.raises(DatabaseError):
        repo.count_rows("sqlite_master; DROP TABLE jobs")


def test_open_repository_uses_settings(settings: Settings, tmp_path: Path) -> None:
    custom = settings.model_copy(
        update={"storage": settings.storage.model_copy(update={"db_path": tmp_path / "o.db"})}
    )
    repo = open_repository(custom)
    assert (tmp_path / "o.db").is_file()
    assert repo.engine is get_engine(str(tmp_path / "o.db"))
    repo.engine.dispose()


def test_migration_failure_wrapped(tmp_path: Path) -> None:
    engine = make_engine(tmp_path / "m.db")
    with pytest.raises(DatabaseError, match="no-such-rev"):
        upgrade(engine, "no-such-rev")
    engine.dispose()


def test_coach_messages_keep_insert_order_on_equal_timestamps(repo: Repository) -> None:
    s = repo.create_session("raw/v.mp4")
    for i in range(10):
        repo.add_coach_message(s.id, "user" if i % 2 == 0 else "assistant", str(i))
    with repo.engine.begin() as conn:
        conn.execute(text("UPDATE coach_messages SET created_at = '2026-01-01 00:00:00.000000'"))
    msgs = repo.list_coach_messages(s.id)
    assert len({m.created_at for m in msgs}) == 1
    assert [m.content for m in msgs] == [str(i) for i in range(10)]
