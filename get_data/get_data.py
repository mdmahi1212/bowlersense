"""
Download only the CricketBowling clips from guyuchao/UCF101 and split them
into close shots vs wide shots.

A clip counts as "close" when the main person fills a large share of the
frame (median person-box height / frame height across sampled frames).

Usage (Windows PowerShell):
    pip install -U huggingface_hub ultralytics opencv-python
    python get_close_bowling.py --out "E:\\datasets\\cricket_bowling"

Options:
    --threshold 0.45   higher = stricter "close" (0.35 loose, 0.55 strict)
    --frames 10        frames sampled per clip
    --skip-download    reuse clips already in <out>\\raw

Output in --out:
    raw\\      all CricketBowling clips as downloaded
    close\\    clips judged close shots
    wide\\     the rest
    results.csv   score per clip
    review.html   thumbnail grid, sorted by score, to check by eye
"""
import argparse
import base64
import csv
import importlib
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def ensure(pkg_import, pkg_pip):
    try:
        importlib.import_module(pkg_import)
    except ImportError:
        print(f"Installing {pkg_pip} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg_pip])


# Installs anything missing so the script works straight from VS Code's Run button
ensure("huggingface_hub", "huggingface_hub")
ensure("cv2", "opencv-python")
ensure("ultralytics", "ultralytics")

import cv2  # noqa: E402

CLASS = "CricketBowling"
# Default save location: E:\datasets\cricket_bowling if drive E: exists, else next to this script
DEFAULT_OUT = (
    r"E:\datasets\cricket_bowling"
    if os.name == "nt" and os.path.exists("E:\\")
    else str(Path(__file__).resolve().parent / "cricket_bowling")
)


def download(raw_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {CLASS} clips from guyuchao/UCF101 ...")
    snapshot_download(
        repo_id="guyuchao/UCF101",
        repo_type="dataset",
        local_dir=str(raw_dir),
        allow_patterns=[f"*{CLASS}*", f"*{CLASS.lower()}*"],
    )


def find_clips(raw_dir: Path):
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in exts and CLASS.lower() in p.name.lower())


def sample_frames(path: Path, n: int):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    if total <= 0:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        step = max(1, len(frames) // n)
        return frames[::step][:n]
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def score_clip(model, frames):
    """Median over frames of (tallest person box height / frame height)."""
    ratios = []
    for f in frames:
        h = f.shape[0]
        res = model(f, classes=[0], conf=0.35, verbose=False)[0]
        best = 0.0
        for box in res.boxes.xyxy.tolist():
            best = max(best, (box[3] - box[1]) / h)
        ratios.append(best)
    return statistics.median(ratios) if ratios else 0.0


def thumb_b64(frame, width=220):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode() if ok else ""


def write_review(rows, path: Path, threshold: float):
    cards = []
    for r in sorted(rows, key=lambda r: -r["score"]):
        tag = "close" if r["close"] else "wide"
        color = "#2e7d32" if r["close"] else "#9e9e9e"
        cards.append(
            f'<figure><img src="data:image/jpeg;base64,{r["thumb"]}">'
            f'<figcaption><b style="color:{color}">{tag} · {r["score"]:.2f}</b><br>{r["name"]}</figcaption></figure>'
        )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>CricketBowling review</title>
<style>body{{font:14px system-ui;margin:20px;background:#fafafa}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;padding:6px}}
img{{width:100%;border-radius:4px}}figcaption{{font-size:12px;word-break:break-all;margin-top:4px}}</style></head>
<body><h2>CricketBowling clips by closeness (threshold {threshold})</h2>
<p>Score = how much of the frame height the bowler fills. Sorted closest first.</p>
<div class="g">{''.join(cards)}</div></body></html>"""
    path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    print(f"Saving to: {out.resolve()}")
    raw, close_dir, wide_dir = out / "raw", out / "close", out / "wide"
    for d in (raw, close_dir, wide_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not a.skip_download:
        download(raw)

    clips = find_clips(raw)
    if not clips:
        sys.exit(
            "No CricketBowling video files found after download.\n"
            "The repo may store videos inside an archive. Check its 'Files and versions' tab on Hugging Face,\n"
            "download the archive, extract the CricketBowling folder into " + str(raw) + " and rerun with --skip-download."
        )
    print(f"Found {len(clips)} clips. Scoring with YOLOv8n (first run downloads the 6 MB model) ...")

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")

    rows = []
    for i, clip in enumerate(clips, 1):
        frames = sample_frames(clip, a.frames)
        if not frames:
            print(f"  [{i}/{len(clips)}] unreadable: {clip.name}")
            continue
        s = score_clip(model, frames)
        is_close = s >= a.threshold
        shutil.copy2(clip, (close_dir if is_close else wide_dir) / clip.name)
        rows.append({"name": clip.name, "score": s, "close": is_close, "thumb": thumb_b64(frames[len(frames) // 2])})
        print(f"  [{i}/{len(clips)}] {s:.2f} {'CLOSE' if is_close else 'wide '} {clip.name}")

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "closeness_score", "close_shot"])
        for r in sorted(rows, key=lambda r: -r["score"]):
            w.writerow([r["name"], f"{r['score']:.3f}", r["close"]])
    write_review(rows, out / "review.html", a.threshold)

    n_close = sum(r["close"] for r in rows)
    print(f"\nDone. {n_close} close / {len(rows) - n_close} wide.")
    print(f"Open {out / 'review.html'} to check, then adjust --threshold and rerun with --skip-download if needed.")


if __name__ == "__main__":
    main()