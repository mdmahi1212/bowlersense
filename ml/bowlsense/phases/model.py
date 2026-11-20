"""Optional learned phase/release model (spec Section 6): a BiLSTM over normalised keypoint
sequences.

Two heads per frame: phase (0 = before the bowling action, 1 = bowling, 2 = after release)
and a release score; the release frame is the argmax of the release score over time. The
pipeline uses it only when ``phases.use_model`` is true and the weights file exists in
``models/``; the rule-based detector stays the default.

Weights are loaded with ``torch.load(weights_only=True)`` (no pickled code), together with a
JSON metadata file that pins the feature version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from bowlsense.errors import ModelNotFoundError
from bowlsense.phases.features import FEATURE_VERSION, N_FEATURES, frame_features
from bowlsense.schemas import PoseSequence

PHASE_NAMES: tuple[str, ...] = ("before", "bowling", "after")


def build_network(hidden: int = 64, layers: int = 2, dropout: float = 0.2) -> Any:
    """The BiLSTM (imports torch lazily so the rest of BowlSense doesn't need it)."""
    import torch
    from torch import nn

    class PhaseNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(N_FEATURES, hidden)
            self.lstm = nn.LSTM(
                hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                dropout=dropout if layers > 1 else 0.0,
            )  # fmt: skip
            self.phase = nn.Linear(2 * hidden, len(PHASE_NAMES))
            self.release = nn.Linear(2 * hidden, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h, _ = self.lstm(torch.relu(self.inp(x)))
            return self.phase(h), self.release(h).squeeze(-1)

    return PhaseNet()


@dataclass(frozen=True)
class PhasePrediction:
    """Model output for one sequence."""

    release: int
    release_confidence: float  # softmax probability mass within +-1 frame of the argmax
    phase: np.ndarray  # per-frame phase index


class PhaseModel:
    """Loaded weights + metadata; call ``predict`` on a cleaned sequence."""

    def __init__(self, weights: Path, meta: dict[str, Any]) -> None:
        """Load once; metadata must match the current feature version."""
        import torch

        if meta.get("feature_version") != FEATURE_VERSION:
            raise ValueError(
                f"{weights.name} was trained on feature version {meta.get('feature_version')}"
                f", this BowlSense uses {FEATURE_VERSION}; retrain the model."
            )
        self.meta = meta
        self.net = build_network(**meta["network"])
        self.net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net.eval()

    def predict(self, seq: PoseSequence) -> PhasePrediction:
        """Release frame and per-frame phases for one sequence."""
        import torch

        x = torch.tensor(frame_features(seq), dtype=torch.float32)[None]
        with torch.no_grad():
            phase_logits, release_logits = self.net(x)
        probs = torch.softmax(release_logits[0], dim=0).numpy()
        release = int(np.argmax(probs))
        lo, hi = max(0, release - 1), min(len(probs), release + 2)
        return PhasePrediction(
            release=release,
            release_confidence=float(probs[lo:hi].sum()),
            phase=phase_logits[0].argmax(dim=1).numpy(),
        )


def meta_path(weights: Path) -> Path:
    """The metadata JSON next to a weights file."""
    return weights.with_suffix(".json")


@lru_cache(maxsize=2)
def load_phase_model(weights: Path) -> PhaseModel:
    """Load (and cache) a trained phase model from ``models/``.

    Raises:
        ModelNotFoundError: if the weights or metadata are missing.
    """
    if not weights.is_file() or not meta_path(weights).is_file():
        raise ModelNotFoundError(
            weights.name, weights.parent, "Train it with `python -m train.train_release`."
        )
    meta = json.loads(meta_path(weights).read_text(encoding="utf-8"))
    return PhaseModel(weights, meta)
