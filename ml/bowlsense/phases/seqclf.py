"""Small sequence classifier (BiGRU + attention pooling) over per-frame pose features.

Used for the bowling arm (features in raw left/right order) and delivery type (features in
bowling/front role order). Weights load with ``torch.load(weights_only=True)``; a JSON file
alongside pins the feature version, network size, class names and input layout.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from bowlsense.errors import ModelNotFoundError
from bowlsense.geometry.angles import FloatArray
from bowlsense.phases.features import FEATURE_VERSION, N_FEATURES, frame_features
from bowlsense.schemas import PoseSequence

Layout = Literal["raw", "role"]


def build_classifier(n_classes: int, hidden: int = 48, dropout: float = 0.2) -> Any:
    """The network (torch imported lazily)."""
    import torch
    from torch import nn

    class SeqClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(N_FEATURES, hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
            self.att = nn.Linear(2 * hidden, 1)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(2 * hidden, n_classes)

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            h, _ = self.gru(torch.relu(self.inp(x)))
            score = self.att(h).squeeze(-1)
            if mask is not None:
                score = score.masked_fill(~mask, -1e4)
            w = torch.softmax(score, dim=1)
            pooled = (h * w[..., None]).sum(dim=1)
            logits: torch.Tensor = self.out(self.drop(pooled))
            return logits

    return SeqClassifier()


def sequence_features(seq: PoseSequence, layout: Layout) -> FloatArray:
    """Per-frame features; ``raw`` keeps left/right order (right side first)."""
    if layout == "raw":
        seq = seq.model_copy(update={"bowling_arm": "right"})
    return frame_features(seq)


class SequenceClassifier:
    """Loaded classifier; ``predict_proba`` on a cleaned sequence."""

    def __init__(self, weights: Path, meta: dict[str, Any]) -> None:
        """Load once; the feature version must match."""
        import torch

        if meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"{weights.name} uses an old feature version; retrain it.")
        self.meta = meta
        self.classes: list[str] = list(meta["classes"])
        self.layout: Layout = meta["layout"]
        self.net = build_classifier(len(self.classes), **meta["network"])
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net.eval()

    def predict_proba(self, seq: PoseSequence) -> dict[str, float]:
        """Class name -> probability."""
        import torch

        x = torch.tensor(sequence_features(seq, self.layout), dtype=torch.float32)[None]
        with torch.no_grad():
            p = torch.softmax(self.net(x)[0], dim=0).numpy()
        return {c: float(v) for c, v in zip(self.classes, p, strict=True)}

    def predict(self, seq: PoseSequence) -> tuple[str, float]:
        """Most likely class and its probability."""
        probs = self.predict_proba(seq)
        best = max(probs, key=probs.__getitem__)
        return best, probs[best]


@lru_cache(maxsize=4)
def load_classifier(weights: Path) -> SequenceClassifier:
    """Load (and cache) a trained sequence classifier from ``models/``.

    Raises:
        ModelNotFoundError: if the weights or metadata are missing.
    """
    meta_file = weights.with_suffix(".json")
    if not weights.is_file() or not meta_file.is_file():
        raise ModelNotFoundError(
            weights.name, weights.parent, "Train it with `python -m train.train_classifiers`."
        )
    return SequenceClassifier(weights, json.loads(meta_file.read_text(encoding="utf-8")))


def stack_padded(chunks: list[FloatArray]) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad variable-length (T, F) arrays into (B, T, F) plus a boolean mask (B, T)."""
    t_max = max(len(c) for c in chunks)
    x = np.zeros((len(chunks), t_max, N_FEATURES), dtype=np.float32)
    mask = np.zeros((len(chunks), t_max), dtype=bool)
    for i, c in enumerate(chunks):
        x[i, : len(c)] = c
        mask[i, : len(c)] = True
    return x, mask
