"""Cross-platform `make check`: local-only URL check, ruff, black, mypy, pytest, web checks
(lint + format, build, Vitest, Playwright smoke test).

Run from anywhere: ``ml/.venv/Scripts/python scripts/check.py`` (Windows) or
``ml/.venv/bin/python scripts/check.py`` (Linux/macOS). Exits non-zero on the first failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "ml"
WEB_DIR = ROOT / "web"


def run(name: str, cmd: list[str], cwd: Path) -> None:
    """Run one check, streaming output, and exit if it fails."""
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        print(f"\nFAIL: {name} failed (exit {result.returncode})", flush=True)
        sys.exit(result.returncode)
    print(f"OK: {name}", flush=True)


def main() -> None:
    """Run every check in order."""
    py = sys.executable
    run("local-only URLs", [py, str(ROOT / "scripts" / "check_urls.py")], ROOT)
    run("ruff", [py, "-m", "ruff", "check", "."], ML_DIR)
    run("black", [py, "-m", "black", "--check", "--quiet", "."], ML_DIR)
    run("mypy", [py, "-m", "mypy"], ML_DIR)
    run(
        "pytest",
        [
            py,
            "-m",
            "pytest",
            "--cov=bowlsense",
            "--cov-report=term-missing:skip-covered",
        ],
        ML_DIR,
    )
    if (WEB_DIR / "package.json").exists():
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        run("web lint", [npm, "run", "lint"], WEB_DIR)
        run("web build", [npm, "run", "build"], WEB_DIR)
        run("web test", [npm, "test"], WEB_DIR)
        run("web e2e (Playwright)", [npm, "run", "test:e2e"], WEB_DIR)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
