"""Train the BiLSTM phase/release model and compare it with the rule-based release detector.

Protocol (no leakage):
- Train/test come from the dataset's own split. Validation for early stopping holds out
  whole *videos* of the train split (deliveries in one video share bowler and camera).
- Every extracted window ends 1.5 s after release, so training uses random crops with the
  release at a random position, and evaluation uses fixed seeded crops (3 s, release placed
  uniformly). The rule-based detector is scored on exactly the same crops.

Run: ml/.venv/Scripts/python -m train.train_release
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.config import REPO_ROOT, load_settings
from bowlsense.errors import BowlSenseError
from bowlsense.geometry.angles import keypoint_track
from bowlsense.geometry.body import detect_bowling_arm, joint
from bowlsense.phases.features import FEATURE_VERSION, frame_features
from bowlsense.phases.heuristic import find_release
from bowlsense.phases.model import build_network, meta_path
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import PoseSequence
from train.extract_poses import DEFAULT_OUT, ExtractedDelivery, load_extracted

log = logging.getLogger("train.release")
EVAL_CROP_S = 3.0
MIN_MARGIN = 5  # frames kept on each side of the release in any crop
REPORTS = REPO_ROOT / "data" / "training" / "reports"


@dataclass
class Sample:
    """One delivery ready for training: features, per-frame phase labels, release index."""

    key: str
    video: str
    seq: PoseSequence  # cleaned, bowling arm detected
    x: np.ndarray
    phase: np.ndarray
    release: int


def prepare(items: list[ExtractedDelivery]) -> list[Sample]:
    """Clean, detect the bowling arm (as the live pipeline would) and label every frame."""
    settings = load_settings()
    out = []
    for it in items:
        seq = clean_sequence(it.sequence, settings.pose)
        try:
            arm = detect_bowling_arm(seq).arm
        except BowlSenseError:
            continue  # wrists never seen
        seq = seq.model_copy(update={"bowling_arm": arm})
        n = len(seq.frames)
        phase = np.zeros(n, dtype=np.int64)
        phase[max(0, it.span_start_idx) : it.release_idx] = 1
        phase[it.release_idx :] = 2
        feats = frame_features(seq)
        out.append(Sample(it.delivery.key, it.delivery.video, seq, feats, phase, it.release_idx))
    return out


def crop_bounds(n: int, release: int, length: int, rng: np.random.Generator) -> tuple[int, int]:
    """A ``[start, stop)`` crop of ``length`` frames with the release uniformly placed."""
    length = min(length, n)
    lo = max(0, release - (length - 1 - MIN_MARGIN))
    hi = min(release - MIN_MARGIN, n - length)
    start = (
        int(rng.integers(lo, hi + 1))
        if hi >= lo
        else max(0, min(release - length // 2, n - length))
    )
    return start, start + length


def eval_crops(samples: list[Sample], fps: float, seed: int = 0) -> list[tuple[Sample, int, int]]:
    """Three fixed crops per sample (deterministic)."""
    rng = np.random.default_rng(seed)
    length = round(EVAL_CROP_S * fps)
    return [(s, *crop_bounds(len(s.x), s.release, length, rng)) for s in samples for _ in range(3)]


def heuristic_release(seq: PoseSequence, start: int, stop: int) -> int | None:
    """The rule-based detector on a crop (auto-detected arm, as in the pipeline)."""
    settings = load_settings()
    crop = seq.model_copy(update={"frames": seq.frames[start:stop]})
    try:
        arm = detect_bowling_arm(crop).arm
        wrist = keypoint_track(crop, joint(arm, "bowling", "wrist"))
        shoulder = keypoint_track(crop, joint(arm, "bowling", "shoulder"))
        return find_release(wrist, shoulder, crop.fps, settings.phases.release_refine_frames)
    except BowlSenseError:
        return None


def summarise(errors: list[int | None], fps: float) -> dict[str, float]:
    """Hit rates within +-1/2/3 frames, and mean absolute error (failures count as misses)."""
    n = len(errors)
    found = [abs(e) for e in errors if e is not None]
    return {
        "n": n,
        "within_1": sum(e <= 1 for e in found) / n,
        "within_2": sum(e <= 2 for e in found) / n,
        "within_3": sum(e <= 3 for e in found) / n,
        "within_5": sum(e <= 5 for e in found) / n,
        "mae_frames": float(np.mean(found)) if found else float("nan"),
        "mae_ms": float(np.mean(found)) * 1000.0 / fps if found else float("nan"),
        "no_answer": (n - len(found)) / n,
    }


def train(
    samples: list[Sample], val: list[Sample], args: argparse.Namespace
) -> tuple[Any, dict[str, Any]]:
    """Train with early stopping on validation release error; return the best network."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = build_network(hidden=args.hidden, layers=args.layers, dropout=args.dropout)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    fps = samples[0].seq.fps
    val_crops = eval_crops(val, fps, seed=123)
    best: tuple[float, dict[str, Any] | None, int] = (float("inf"), None, -1)
    history = []
    for epoch in range(args.epochs):
        net.train()
        order = rng.permutation(len(samples))
        losses = []
        for b in range(0, len(order), args.batch):
            batch = [samples[i] for i in order[b : b + args.batch]]
            crops = []
            for s in batch:
                length = int(rng.integers(round(2.0 * fps), len(s.x) + 1))
                a, z = crop_bounds(len(s.x), s.release, length, rng)
                crops.append((s, a, z))
            t_max = max(z - a for _, a, z in crops)
            x = torch.zeros(len(crops), t_max, samples[0].x.shape[1])
            ph = torch.full((len(crops), t_max), -100, dtype=torch.long)
            mask = torch.zeros(len(crops), t_max, dtype=torch.bool)
            rel = torch.zeros(len(crops), dtype=torch.long)
            for i, (s, a, z) in enumerate(crops):
                x[i, : z - a] = torch.tensor(s.x[a:z], dtype=torch.float32)
                ph[i, : z - a] = torch.tensor(s.phase[a:z])
                mask[i, : z - a] = True
                rel[i] = s.release - a
            x = x + args.noise * torch.randn_like(x) * mask[..., None]
            phase_logits, release_logits = net(x)
            loss_phase = F.cross_entropy(
                phase_logits.reshape(-1, 3), ph.reshape(-1), ignore_index=-100
            )
            release_logits = release_logits.masked_fill(~mask, -1e4)
            positions = torch.arange(t_max)[None, :].float()
            target = torch.exp(-0.5 * ((positions - rel[:, None].float()) / args.sigma) ** 2)
            target = target * mask
            target = target / target.sum(dim=1, keepdim=True)
            loss_release = -(target * F.log_softmax(release_logits, dim=1)).sum(dim=1).mean()
            loss = loss_phase + args.release_weight * loss_release
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        val_errors = predict_errors(net, val_crops)
        val_mae = float(np.mean(np.abs(val_errors)))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mae_frames": val_mae})
        if val_mae < best[0]:
            best = (val_mae, {k: v.detach().clone() for k, v in net.state_dict().items()}, epoch)
        if epoch - best[2] >= args.patience:
            break
        if epoch % 5 == 0:
            log.info("epoch %3d loss %.3f val MAE %.2f frames (best %.2f @ %d)", epoch,
                     np.mean(losses), val_mae, best[0], best[2])  # fmt: skip
    assert best[1] is not None
    net.load_state_dict(best[1])
    return net, {"best_epoch": best[2], "best_val_mae_frames": best[0], "history": history}


def predict_errors(net: Any, crops: list[tuple[Sample, int, int]]) -> list[int]:
    """Signed release errors (frames) of the network on crops."""
    import torch

    net.eval()
    errs = []
    with torch.no_grad():
        for s, a, z in crops:
            x = torch.tensor(s.x[a:z], dtype=torch.float32)[None]
            _, logits = net(x)
            errs.append(int(torch.argmax(logits[0])) - (s.release - a))
    return errs


def main(argv: list[str] | None = None) -> int:
    """Train, evaluate on the test split, save weights + report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--sigma", type=float, default=1.0, help="release target width (frames)")
    p.add_argument("--release-weight", type=float, default=2.0)
    p.add_argument("--noise", type=float, default=0.02, help="feature jitter for augmentation")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=load_settings().storage.models_path)
    p.add_argument("--name", default="phase_model.pt", help="weights file name in --out")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    items = load_extracted(args.data)
    train_items = [i for i in items if i.delivery.split == "train"]
    test_items = [i for i in items if i.delivery.split == "test"]
    t0 = time.time()
    train_all, test = prepare(train_items), prepare(test_items)
    videos = sorted({s.video for s in train_all})
    rng = np.random.default_rng(args.seed)
    val_videos = set(
        rng.choice(videos, size=max(1, round(args.val_frac * len(videos))), replace=False)
    )
    train_s = [s for s in train_all if s.video not in val_videos]
    val_s = [s for s in train_all if s.video in val_videos]
    log.info("train %d deliveries (%d videos), val %d (%d videos), test %d; prep %.0f s",
             len(train_s), len(videos) - len(val_videos), len(val_s), len(val_videos),
             len(test), time.time() - t0)  # fmt: skip

    net, info = train(train_s, val_s, args)
    fps = test[0].seq.fps
    crops = eval_crops(test, fps, seed=args.seed)
    model_err: list[int | None] = list(predict_errors(net, crops))
    heur_err: list[int | None] = []
    for s, a, z in crops:
        h = heuristic_release(s.seq, a, z)
        heur_err.append(None if h is None else h - (s.release - a))
    report: dict[str, Any] = {
        "feature_version": FEATURE_VERSION,
        "protocol": f"{len(crops)} fixed {EVAL_CROP_S:.0f} s crops of {len(test)} test deliveries",
        "learned_model": summarise(model_err, fps),
        "rule_based": summarise(heur_err, fps),
        "training": {k: v for k, v in info.items() if k != "history"},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    weights = out / args.name
    torch.save(net.state_dict(), weights)
    meta = {
        "feature_version": FEATURE_VERSION,
        "network": {"hidden": args.hidden, "layers": args.layers, "dropout": args.dropout},
        "fps_trained": fps,
        "test_metrics": report["learned_model"],
    }
    meta_path(weights).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_name = (
        "release_model.json"
        if args.name == "phase_model.pt"
        else (f"release_model_{Path(args.name).stem}.json")
    )
    (REPORTS / report_name).write_text(
        json.dumps({**report, "history": info["history"]}, indent=2), encoding="utf-8"
    )
    for name in ("learned_model", "rule_based"):
        m = report[name]
        hits = "  ".join(f"±{k}: {100 * m[f'within_{k}']:4.0f}%" for k in (1, 2, 3, 5))
        log.info(
            "%-13s  %s  MAE %.2f frames (%.0f ms)  no answer %.0f%%",
            name, hits, m["mae_frames"], m["mae_ms"], 100 * m["no_answer"],
        )  # fmt: skip
    log.info("saved %s (+ %s); report in %s", weights.name, meta_path(weights).name, REPORTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
