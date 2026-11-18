# BowlSense

AI analysis of cricket bowling from video: pose, phases, legality screening, technique faults,
injury-risk screening, ball speed/length, fatigue and coaching. See `CLAUDE.md` for the full
spec and `PROGRESS.md` for status. (The full README with setup, architecture and disclaimers
comes in Phase 16.)

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
