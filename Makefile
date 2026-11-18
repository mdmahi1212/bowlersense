# `make check` runs lint, type checks and tests. On Windows without make, run:
#   ml/.venv/Scripts/python scripts/check.py
PY ?= ml/.venv/bin/python
ifeq ($(OS),Windows_NT)
PY := ml/.venv/Scripts/python.exe
endif

.PHONY: check schema
check:
	$(PY) scripts/check.py

schema:
	cd ml && ../$(PY) -m bowlsense.schemas --out schema/bowlsense.schema.json
