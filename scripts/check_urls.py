"""Fail if runtime code contains any URL that isn't localhost / 127.0.0.1 / [::1].

BowlSense is 100% local. Setup-only scripts (``scripts/``) and tests are excluded because the
one-time model download legitimately needs remote URLs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_GLOBS: tuple[str, ...] = (
    "ml/bowlsense/**/*.py",
    "config/*.yaml",
    "run.py",
    "web/app/**/*.ts",
    "web/app/**/*.tsx",
    "web/components/**/*.ts",
    "web/components/**/*.tsx",
    "web/lib/**/*.ts",
)
URL_RE = re.compile(r"\b(?:https?|wss?)://[^\s\"'`<>()\]\[{}]+(?:\[[^\]]*\][^\s\"'`<>()]*)?")
LOCAL_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


def offending_urls(text: str) -> list[str]:
    """Return every URL in ``text`` whose host is not a loopback host."""
    bad: list[str] = []
    for match in URL_RE.finditer(text):
        url = match.group(0)
        try:
            host = urlsplit(url).hostname or ""
        except ValueError:
            host = ""
        if host not in LOCAL_HOSTS:
            bad.append(url)
    return bad


def scan(root: Path = ROOT) -> list[str]:
    """Return ``path:line: url`` for every non-local URL in runtime files under ``root``."""
    problems: list[str] = []
    seen: set[Path] = set()
    for pattern in RUNTIME_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file() or "node_modules" in path.parts:
                continue
            seen.add(path)
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for lineno, line in enumerate(lines, 1):
                for url in offending_urls(line):
                    problems.append(f"{path.relative_to(root).as_posix()}:{lineno}: {url}")
    return problems


def main() -> None:
    """Print offending URLs and exit 1 if there are any."""
    problems = scan()
    if problems:
        print("Non-local URLs found in runtime code (BowlSense must stay 100% local):")
        print("\n".join(f"  {p}" for p in problems))
        sys.exit(1)
    print("No non-local URLs in runtime code.")


if __name__ == "__main__":
    main()
