"""One-time setup: Python virtualenv + pinned packages, web app packages, database.

Run with any Python 3.11:  ``python scripts/setup.py``   (``--dev`` adds test/lint tools,
``--no-web`` skips the Node.js part). Needs the internet once; afterwards run
``python scripts/download_models.py`` and then ``python run.py``.

Uses ``uv`` when it is installed (much faster), otherwise ``python -m venv`` + pip.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "ml"
WEB = ROOT / "web"
VENV = ML / ".venv"


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    """Run a command, echoing it; exit with a clear message if it fails."""
    print(f"$ {' '.join(cmd)}", flush=True)
    if subprocess.call(cmd, cwd=cwd) != 0:
        sys.exit(f"Setup step failed: {' '.join(cmd)}")


def venv_python() -> Path:
    """The virtualenv's interpreter path for this OS."""
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def setup_python(dev: bool) -> Path:
    """Create ``ml/.venv`` (Python 3.11) and install the pinned requirements."""
    reqs = ML / ("requirements-dev.txt" if dev else "requirements.txt")
    uv = shutil.which("uv")
    py = venv_python()
    if uv:
        if not py.is_file():
            run([uv, "venv", "--python", "3.11", str(VENV)])
        run([uv, "pip", "install", "--python", str(py), "-r", str(reqs)])
    else:
        if sys.version_info[:2] != (3, 11):
            print(f"Note: BowlSense is tested on Python 3.11; this is {sys.version.split()[0]}.")
        if not py.is_file():
            run([sys.executable, "-m", "venv", str(VENV)])
        run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(py), "-m", "pip", "install", "-r", str(reqs)])
    run(
        [str(py), "-c", "import bowlsense, cv2, torch, onnxruntime, fastapi; print('imports OK')"],
        ML,
    )
    return py


def setup_web() -> None:
    """Install the exact web packages from package-lock.json and build the app."""
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm") or shutil.which("npm")
    if npm is None:
        sys.exit("Node.js 20+ (with npm) is needed for the web app: https://nodejs.org")
    os.environ["NEXT_TELEMETRY_DISABLED"] = "1"  # no Next.js telemetry
    run([npm, "ci", "--no-audit", "--no-fund"], WEB)
    run([npm, "run", "build"], WEB)


def main(argv: list[str] | None = None) -> int:
    """Set everything up and print the next steps."""
    parser = argparse.ArgumentParser(description="Set up BowlSense")
    parser.add_argument("--dev", action="store_true", help="also install test/lint tools")
    parser.add_argument("--no-web", action="store_true", help="skip the Node.js web app")
    args = parser.parse_args(argv)
    py = setup_python(args.dev)
    run(
        [
            str(py),
            "-c",
            "from bowlsense.config import load_settings; "
            "from bowlsense.db.session import get_engine; "
            "s = load_settings(); get_engine(str(s.storage.db_file)); "
            "print('database ready:', s.storage.db_file)",
        ],
        ML,
    )
    if not args.no_web:
        setup_web()
    rel = venv_python().relative_to(ROOT)
    print("\nSetup complete. Next:")
    print(f"  1. {rel} scripts/download_models.py   (one-time model download)")
    print("  2. python run.py                        (starts everything, opens the browser)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
