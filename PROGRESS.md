# BowlSense — Progress

## Phase 2: Synthetic generator + render_to_video ✅ (2026-11-19)

**Done** (`ml/synth/skeleton_gen.py`)
- A 3D stick figure with anthropometric segment lengths (Winter / Drillis-Contini ratios × height), driven through a keyframed action: standing → run-up (8 steps, legs cycle, arm swing) → bound → BFC → FFC → arm circle (~360°) → release → follow-through. Joint and body angles are interpolated with monotone cubic splines (PCHIP). Legs use 2-link IK towards scripted foot plants, so **segment lengths are exactly constant**.
- Orthographic projection to `side`, `front` or `behind` views. Left-arm bowlers are an exact mirror (Z→−Z, l/r names swapped).
- Controllable parameters: `fps, height_px, bowling_arm, view, action_type (side_on/semi_open/front_on/mixed), elbow_extension_deg, counter_rotation_deg, knee_flex_change_deg, stride_pct, lateral_flexion_deg, release_height_pct, runup_decel_pct, follow_through_pct, front_arm_ffc_deg, noise_px, dropout_rate, seed, include_z`.
- Outputs a `SynthResult`: the `PoseSequence`, ground-truth `PhaseEvents`, and a `SynthTruth` with every parameter **re-measured on the clean motion**. Ground truth uses the spec's own event definitions: contacts = the first frame with the foot down, release = wrist-height peak, arm-horizontal = the upper arm crossing horizontal going up, bound = the highest hip point, run-up start / follow-through end = hip-speed thresholds.
- **Fails loudly**: a parameter combination the body can't physically produce raises `ValueError` naming the parameter and the achievable range. It never silently returns a different value.
- `render_to_video(seq, path)` draws the stick figure to MP4 (mp4v; the browser-ready H.264 re-mux is Phase 8). Frames are drawn one at a time, and the writer is released in `finally`. `SKELETON_EDGES` was added to `schemas.py` for all renderers.
- ~0.15 s per 120-fps sequence.

**Tested**: 34 generator tests, 138 total, all green. Covers: constant segment lengths for side/front/behind × right/left; height scaling; exact left/right mirroring; the bowling wrist is fastest on the correct side; truth ≈ request across 5 parameter sets; elbow extension visible in the **2D** side view (the arm moves in the camera plane); release height from keypoints; mixed-action detection (hip–shoulder separation > 30° only for `mixed`); phase times independent of fps; feet on the ground at their contact frames; noise std and dropout rate; seed determinism; unreachable parameters raise; JSON round-trip; MP4 read back with the right frame count, size and fps.

Also fixed a real Phase 1 bug the full run caught: coach messages written within the same clock tick could come back out of order (the tie-break was a random UUID). They're now ordered by insertion, with a regression test that forces equal timestamps.

**Definitions later phases must match** (deliberate, documented choices)
- **Stride** = back ankle at BFC → front ankle at FFC (the standard delivery-stride definition). A literal same-frame ankle-to-ankle stride of 90%+ of height is anatomically out of reach (leg ≈ 49% of height). Phase 5's OVERSTRIDE/UNDERSTRIDE must use this definition.
- **Release height** = estimated ball height = wrist height + hand length (0.108 H). Wrist height alone can't reach the spec's 115% LOW_RELEASE threshold for a normal action. Phase 5 must use the same definition.
- Reachable ranges: stride 60–105%, release height ≈104–125% (depends on other params), run-up decel ≤ 50%, follow-through ≤ 120% of stride.
- Below 100 fps, only stride is checked against the request: the other quantities are measured at specific frames, and a coarse frame grid shifts them.

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

---

**Next**: Phase 3: geometry + smoothing + body (angles, velocities, height, bowling-arm detection, mirroring).
