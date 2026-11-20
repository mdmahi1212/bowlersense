"""Phase 16 gate: with BOWLSENSE_OFFLINE=1 (any non-localhost connection raises), the full
synthetic pipeline and the API tests pass."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]


def test_gate_pipeline_and_api_pass_offline() -> None:
    env = {**os.environ, "BOWLSENSE_OFFLINE": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_pipeline_synthetic.py", "tests/test_api.py", "tests/test_offline.py"],
        cwd=ML_DIR, env=env, capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    tail = (proc.stdout + proc.stderr)[-3000:]
    assert proc.returncode == 0, tail
    assert "[100%]" in proc.stdout and "F" not in proc.stdout.split("[100%]")[0], tail
