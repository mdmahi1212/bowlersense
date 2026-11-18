# BowlSense — Progress

## Phase 1 — Scaffold ✅ (2026-11-19)

**Done**
- Repo layout started: `config/`, `ml/bowlsense/`, `ml/tests/`, `scripts/`. The build spec now lives in `CLAUDE.md`.
- `config/default.yaml` (spec Section 14 defaults, plus `video` quality thresholds, `jobs` and `api` sections so no limits are hard-coded) and `bowlsense/config.py`: strict pydantic settings (unknown keys rejected; fault/injury weight keys, weight sum, band order, length-zone order and fps order validated), env settings (`BOWLSENSE_CONFIG`, `BOWLSENSE_DB`, `BOWLSENSE_SQLITE_PATH`, `BOWLSENSE_CACHE_DIR`, `ANTHROPIC_API_KEY`, Supabase vars, `BOWLSENSE_WEB_ORIGIN`).
- `bowlsense/schemas.py`: every Section 4 contract, plus `FatigueDrift`, `FatigueResult` and `WorkloadResult` (the spec names these but doesn't define them). JSON Schema export via `python -m bowlsense.schemas --out schema/bowlsense.schema.json` → `ml/schema/bowlsense.schema.json`.
- `bowlsense/errors.py`: an exception hierarchy with actionable messages (`LowFpsError`, `NoBowlerDetectedError`, `PoseQualityError`, `UnsupportedMediaError`, `ModelDownloadError`, ...).
- `bowlsense/db.py`: an SQLite store mirroring the Section 8 tables (including `bowler_shares`), FK-enforced, with an atomic job claim (`UPDATE … RETURNING`), retry/fail logic and coach messages.
- `make check` → `scripts/check.py` (ruff, black, mypy --strict, pytest+coverage; it also runs the web checks once `web/` exists).

**Tested**: 47 tests, 98% coverage of `bowlsense`. Covers config validation, env defaults, schema validation/round-trip/export, error messages, the DB CRUD, the job retry → fail path, and concurrent job claims with no duplicates.

**Known issues / notes**
- `make` isn't installed on this Windows machine; run `ml/.venv/Scripts/python scripts/check.py` instead (it's what `make check` calls).
- `ffmpeg` isn't installed yet; Phase 8 (overlay re-mux) needs it.
- `BOWLSENSE_DB=supabase` raises a clear `ConfigError` until Phase 13.
- The Python venv is at `ml/.venv` (Python 3.11.9, created with `uv`).

**Next**: Phase 2: the synthetic skeleton generator + `render_to_video`.
