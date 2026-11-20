"""Train the sequence classifiers on the dataset and score them on its test split.

- ``delivery_type`` (pace vs spin): role-ordered features over the run-up and action.

Validation (for early stopping) holds out whole videos of the train split. Test crops use the
arm and release the live pipeline would have (learned arm, learned release). A model is saved
only if it beats its baseline on validation; the report says which.

Run: ml/.venv/Scripts/python -m train.train_classifiers
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.config import load_settings
from bowlsense.geometry.body import detect_bowling_arm_at_release
from bowlsense.phases.features import FEATURE_VERSION
from bowlsense.phases.model import load_phase_model
from bowlsense.phases.seqclf import Layout, build_classifier, sequence_features, stack_padded
from bowlsense.pose.smoothing import clean_sequence
from bowlsense.schemas import PoseSequence
from train.extract_poses import DEFAULT_OUT, ExtractedDelivery, load_extracted
from train.train_release import REPORTS

log = logging.getLogger("train.classifiers")


@dataclass
class Item:
    """One training example: cleaned sequence, its release index, label, video."""

    seq: PoseSequence
    release: int
    label: int
    video: str


def crop(item: Item, rng: np.random.Generator | None, fps: float) -> tuple[int, int]:
    """[start, stop) around release: 2.5 s before to 0.5 s after (random jitter in training)."""
    before, after = round(2.5 * fps), round(0.5 * fps)
    if rng is not None:
        before = int(rng.integers(round(1.5 * fps), round(3.0 * fps) + 1))
        after = int(rng.integers(round(0.2 * fps), round(1.0 * fps) + 1))
    return max(0, item.release - before), min(len(item.seq.frames), item.release + after + 1)


def features(
    items: list[Item], layout: Layout, rng: np.random.Generator | None
) -> list[np.ndarray]:
    """Cropped feature arrays."""
    out = []
    for it in items:
        a, z = crop(it, rng, it.seq.fps)
        out.append(
            sequence_features(it.seq.model_copy(update={"frames": it.seq.frames[a:z]}), layout)
        )
    return out


def fit(
    train: list[Item], val: list[Item], layout: Layout, n_classes: int, args: argparse.Namespace
) -> tuple[Any, float, list[float]]:
    """Train with early stopping on validation balanced accuracy."""
    import torch
    import torch.nn.functional as F  # noqa: N812

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = build_classifier(n_classes, hidden=args.hidden, dropout=args.dropout)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    y_tr = np.array([it.label for it in train])
    weights = torch.tensor(len(y_tr) / (n_classes * np.bincount(y_tr, minlength=n_classes)),
                           dtype=torch.float32)  # fmt: skip
    best: tuple[float, dict[str, Any] | None, int] = (-1.0, None, -1)
    history = []
    for epoch in range(args.epochs):
        net.train()
        for _ in range(0, len(train), args.batch):
            batch = [train[i] for i in rng.permutation(len(train))[: args.batch]]
            x, mask = stack_padded(features(batch, layout, rng))
            xt = torch.tensor(x) + args.noise * torch.randn(x.shape)
            labels = torch.tensor([it.label for it in batch])
            loss = F.cross_entropy(net(xt, torch.tensor(mask)), labels, weight=weights)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        score = balanced_accuracy(net, val, layout, n_classes)
        history.append(score)
        if score > best[0]:
            best = (score, {k: v.detach().clone() for k, v in net.state_dict().items()}, epoch)
        if epoch - best[2] >= args.patience:
            break
    assert best[1] is not None
    net.load_state_dict(best[1])
    return net, best[0], history


def predict(net: Any, items: list[Item], layout: Layout) -> np.ndarray:
    """Predicted class per item (fixed crops)."""
    import torch

    net.eval()
    with torch.no_grad():
        return np.array(
            [
                int(net(torch.tensor(f[None], dtype=torch.float32)).argmax())
                for f in features(items, layout, None)
            ]
        )


def balanced_accuracy(net: Any, items: list[Item], layout: Layout, n_classes: int) -> float:
    """Mean per-class recall."""
    y = np.array([it.label for it in items])
    p = predict(net, items, layout)
    recalls = [float(np.mean(p[y == c] == c)) for c in range(n_classes) if np.any(y == c)]
    return float(np.mean(recalls))


def scores(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict[str, Any]:
    """Accuracy, balanced accuracy and per-class recall."""
    recall = {c: float(np.mean(p[y == i] == i)) for i, c in enumerate(classes) if np.any(y == i)}
    return {
        "accuracy": float(np.mean(y == p)),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "recall": recall,
        "n": len(y),
    }


def split_val(items: list[Item], frac: float, seed: int) -> tuple[list[Item], list[Item]]:
    """Hold out whole videos."""
    videos = sorted({it.video for it in items})
    rng = np.random.default_rng(seed)
    val_v = set(rng.choice(videos, size=max(1, round(frac * len(videos))), replace=False))
    return [i for i in items if i.video not in val_v], [i for i in items if i.video in val_v]


def save(net: Any, out: Path, name: str, classes: list[str], layout: Layout,
         args: argparse.Namespace, metrics: dict[str, Any]) -> None:  # fmt: skip
    """Weights + JSON metadata into ``models/``."""
    import torch

    torch.save(net.state_dict(), out / f"{name}.pt")
    meta = {
        "feature_version": FEATURE_VERSION,
        "classes": classes,
        "layout": layout,
        "network": {"hidden": args.hidden, "dropout": args.dropout},
        "test_metrics": metrics,
    }
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Train both classifiers; write models and a report."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=load_settings().storage.models_path)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    settings = load_settings()
    raw_items: list[ExtractedDelivery] = load_extracted(args.data)
    cleaned = {it.delivery.key: clean_sequence(it.sequence, settings.pose) for it in raw_items}
    report: dict[str, Any] = {}

    # Bowling arm: a learned arm classifier (raw L/R features + mirror augmentation) was
    # tried and was no better than chance on left-armers, so the pipeline uses the
    # elbow-at-learned-release rule (bowlsense.pipeline.choose_bowling_arm) instead.
    arms = ["right", "left"]
    phase_model = load_phase_model(args.out / "phase_model.pt")

    # --- delivery type (role features, true arm in training; learned arm + release at test) ---
    types = ["spin", "fast"]

    def type_items(split: str) -> list[Item]:
        out = []
        for it in raw_items:
            d = it.delivery
            if d.split != split or d.bowling_type is None or d.bowler_hand is None:
                continue
            seq = cleaned[d.key].model_copy(update={"bowling_arm": d.bowler_hand})
            out.append(Item(seq, it.release_idx, types.index(d.bowling_type), d.video))
        return out

    tr, va = split_val(type_items("train"), 0.2, args.seed)
    net_t, val_t, _ = fit(tr, va, "role", 2, args)
    live = []
    for it in type_items("test"):
        best = max(
            (phase_model.predict(it.seq.model_copy(update={"bowling_arm": a})) for a in arms),
            key=lambda pr: pr.release_confidence,
        )
        arm = detect_bowling_arm_at_release(it.seq, best.release).arm
        seq = it.seq.model_copy(update={"bowling_arm": arm})
        live.append(Item(seq, phase_model.predict(seq).release, it.label, it.video))
    y_t = np.array([it.label for it in live])
    type_scores = scores(y_t, predict(net_t, live, "role"), types)
    majority = int(np.bincount([i.label for i in tr]).argmax())
    baseline = scores(y_t, np.full_like(y_t, majority), types)
    beats = val_t > 0.6
    report["delivery_type"] = {"learned_end_to_end": type_scores, "majority_baseline": baseline,
                               "val_balanced_accuracy": val_t, "saved": beats}  # fmt: skip
    if beats:
        save(net_t, args.out, "delivery_type_seq", types, "role", args, type_scores)
    log.info(
        "pace/spin    learned %s | majority %s | saved=%s", _fmt(type_scores), _fmt(baseline), beats
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "classifiers.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def _fmt(m: dict[str, Any]) -> str:
    return f"acc {m['accuracy']:.1%} balanced {m['balanced_accuracy']:.1%} recall {m['recall']}"


if __name__ == "__main__":
    raise SystemExit(main())
