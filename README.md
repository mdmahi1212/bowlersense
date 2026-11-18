# BowlSense

AI analysis of cricket bowling from video, running **100% locally**: pose, phases, legality
screening, technique faults, injury-risk screening, ball speed/length, fatigue and coaching
via a local LLM (Ollama). No cloud services, no accounts, no telemetry. See `CLAUDE.md` for
the full spec and `PROGRESS.md` for status. (The full README comes in Phase 16.)

## Dev setup (Phase 1)

```bash
cd ml
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt   # Linux: .venv/bin/python
```

Run all checks from the repo root:

```bash
ml/.venv/Scripts/python scripts/check.py      # or: make check
```

The SQLite database (`data/bowlsense.db`) is created and migrated automatically on first use;
to migrate by hand: `cd ml && .venv/Scripts/python -m alembic upgrade head`.
