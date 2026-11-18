# BowlSense — Progress

## Phase 1: Scaffold ✅ (2026-11-19, reworked for the 100%-local spec)

The spec was revised to be **100% local** (no Supabase, no hosted APIs, Ollama for the coach). Phase 1 was redone to match it. `CLAUDE.md` holds the current spec.

**Done**
- Layout: `config/`, `data/{storage/raw,storage/outputs,reference}/`, `models/`, `scripts/`, and the `ml/bowlsense/` subpackages (`io, detect, pose, geometry, phases, analysis, coach, render, db`) plus `ml/synth/`.
- Config: `config/default.yaml` holds the Section 14 defaults, plus video-quality thresholds and `jobs`/`api` sections, so no limits are hard-coded. `bowlsense/config.py` loads it into strict pydantic models. Unknown keys are rejected, and these are validated: fault/injury weight keys, the weight sum, band order, length-zone order and fps order. **The Ollama URL must be localhost, `server.host` must be loopback**, and storage paths resolve against the repo root. The only env vars are `BOWLSENSE_CONFIG` and `BOWLSENSE_OFFLINE`; there are no secrets.
- Schemas: `bowlsense/schemas.py` has every Section 4 contract, plus `FatigueDrift`, `FatigueResult` and `WorkloadResult` (named but not defined by the spec). JSON Schema export: `python -m bowlsense.schemas --out schema/bowlsense.schema.json`.
- Errors: `bowlsense/errors.py` defines exceptions with actionable messages.
- Database (`bowlsense/db/`): SQLAlchemy 2.x ORM (`models.py`) for the Section 8 tables. There's no auth: no `profiles`/`bowler_shares`/`owner_id`. Jobs also carry `progress` and `stage` for the SSE stream. `session.py` sets `foreign_keys=ON`, WAL, `busy_timeout`, and `BEGIN IMMEDIATE` for writes. The Alembic migration `0001` is applied automatically by `get_engine`; `alembic upgrade head` also works from `ml/`. `repo.py` handles bowlers, sessions, the atomic job claim (`UPDATE … RETURNING`), progress, retry→fail, coach messages, delete-session (rows + files) and delete-all-data.
- Storage: `bowlsense/storage.py` streams uploads in chunks to `raw/<uuid>.<ext>` (the original name is never used as a path) and enforces the size limit. It rejects path traversal, and handles session output folders, wipe and usage.
- Offline: `bowlsense/offline.py`. With `BOWLSENSE_OFFLINE=1`, importing `bowlsense` patches `socket` so any non-loopback connect/DNS lookup raises `OfflineViolationError`. Library telemetry env vars are set to off on import.
- `make check` → `scripts/check.py`: the **local-only URL check** (`scripts/check_urls.py`), ruff, black, mypy --strict, pytest+coverage, and the web checks once `web/` exists.

**Tested**: 103 tests, 98% coverage. Includes `alembic upgrade head` on an empty DB, migrations ≡ models (autogenerate diff is empty), downgrade, pragmas, the job queue with retries, 4 threads + **2 OS processes writing concurrently with no `database is locked`**, cascade delete of rows *and* files, delete-all, upload limits/traversal, the offline guard (blocks remote, allows loopback, installs on import), and the URL scanner.

**Known issues / notes**
- `make` isn't installed on this Windows machine; run `ml/.venv/Scripts/python scripts/check.py` instead.
- `ffmpeg` isn't installed yet; Phase 8 (overlay re-mux) needs it. Ollama isn't installed yet; Phase 11 needs it, and its tests will mock it.
- Python venv: `ml/.venv` (Python 3.11.9, created with `uv`). `scripts/setup.py` comes in Phase 16.

**Next**: Phase 2: the synthetic skeleton generator + `render_to_video`.
