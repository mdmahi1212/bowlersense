"""One-time download of the model weights BowlSense needs into ``models/``.

After this has run once, BowlSense works with the network unplugged
(``BOWLSENSE_OFFLINE=1``). Files that already exist are skipped (``--force`` re-downloads).
SHA-256 hashes are written to ``models/MANIFEST.json``.

Run: ml/.venv/Scripts/python scripts/download_models.py     (Linux: ml/.venv/bin/python)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Asset:
    """One file to fetch. ``member`` = file to extract when the download is a zip."""

    name: str
    url: str
    approx_mb: float
    member: str | None = None


ASSETS: tuple[Asset, ...] = (
    Asset(
        "rtmpose-m_halpe26_256x192.onnx",
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
        "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        52.1,
        member="end2end.onnx",
    ),
    Asset(
        "yolo11n.pt",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        5.6,
    ),
    Asset(
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        9.4,
    ),
)


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "bowlsense-model-download"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def fetch(asset: Asset, force: bool) -> Path:
    """Download one asset into ``models/`` (atomically) and return its path."""
    target = MODELS / asset.name
    if target.exists() and not force:
        print(f"  {asset.name}: already present, skipping")
        return target
    print(f"  {asset.name} (~{asset.approx_mb:.1f} MB) from {asset.url}")
    with tempfile.TemporaryDirectory(dir=MODELS) as tmp:
        tmp_dir = Path(tmp)
        raw = tmp_dir / "download"
        _download(asset.url, raw)
        if asset.member is not None:
            with zipfile.ZipFile(raw) as zf:
                matches = [n for n in zf.namelist() if Path(n).name == asset.member]
                if not matches:
                    raise RuntimeError(f"{asset.member} not found inside {asset.url}")
                with zf.open(matches[0]) as src, (tmp_dir / "file").open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            produced = tmp_dir / "file"
        else:
            produced = raw
        if produced.stat().st_size == 0:
            raise RuntimeError(f"{asset.name} downloaded empty")
        produced.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    """Download every asset and write the manifest."""
    parser = argparse.ArgumentParser(description="Download BowlSense model weights")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model weights into {MODELS}")
    manifest = {}
    failed = []
    for asset in ASSETS:
        try:
            path = fetch(asset, args.force)
        except Exception as exc:  # report every failure, then exit non-zero
            failed.append(asset.name)
            print(
                f"  FAILED {asset.name}: {exc}\n    Download it manually from {asset.url}"
                + (f" (extract {asset.member} and rename it)" if asset.member else "")
                + f" and save it as {MODELS / asset.name}",
                file=sys.stderr,
            )
            continue
        manifest[asset.name] = {
            "url": asset.url,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (MODELS / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if failed:
        print(f"{len(failed)} download(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("All model weights present. BowlSense can now run offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
