"""Module G: identify which known bowler an action belongs to (GRU over angle curves).

Trains only with enough data: at least ``min_bowlers`` bowlers with
``min_deliveries_per_bowler`` deliveries each. Otherwise every call returns
``insufficient data`` with how much more is needed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from bowlsense.analysis.compare import JOINTS
from bowlsense.config import BowlerIdSettings
from bowlsense.geometry.angles import FloatArray


class BowlerIdResult(BaseModel):
    """A prediction, or the reason there isn't one."""

    model_config = ConfigDict(frozen=True)

    status: str  # "ok" or "insufficient data"
    bowler_id: str | None = None
    probability: float | None = None
    detail: str = ""


def data_status(labels: Sequence[str], s: BowlerIdSettings) -> tuple[bool, str]:
    """Whether there's enough data to train, and a human-readable reason."""
    counts = Counter(labels)
    enough = [b for b, n in counts.items() if n >= s.min_deliveries_per_bowler]
    if len(enough) >= s.min_bowlers:
        return True, f"{len(enough)} bowlers with >= {s.min_deliveries_per_bowler} deliveries"
    return False, (
        f"needs {s.min_bowlers} bowlers with at least {s.min_deliveries_per_bowler} deliveries "
        f"each; have {len(enough)}"
    )


def _network(n_classes: int, hidden: int) -> Any:
    import torch
    from torch import nn

    class BowlerNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(len(JOINTS), hidden, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * hidden, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, _ = self.gru(x)
            logits: torch.Tensor = self.out(h.mean(dim=1))
            return logits

    return BowlerNet()


class BowlerIdentifier:
    """Train on (bowler, action curve) pairs; predict the bowler of a new curve."""

    def __init__(self, settings: BowlerIdSettings) -> None:
        """An untrained identifier."""
        self.settings = settings
        self.classes: list[str] = []
        self.net: Any = None
        self.status_detail = "not trained"

    def fit(self, curves: Sequence[FloatArray], labels: Sequence[str], seed: int = 0) -> bool:
        """Train if there is enough data (only bowlers that meet the minimum are used).

        Returns whether a model was trained.
        """
        import torch
        import torch.nn.functional as F  # noqa: N812

        ok, detail = data_status(labels, self.settings)
        self.status_detail = detail
        if not ok:
            self.net = None
            return False
        counts = Counter(labels)
        keep = [
            i for i, b in enumerate(labels) if counts[b] >= self.settings.min_deliveries_per_bowler
        ]
        self.classes = sorted({labels[i] for i in keep})
        x = torch.tensor(np.stack([curves[i] for i in keep]), dtype=torch.float32)
        y = torch.tensor([self.classes.index(labels[i]) for i in keep])
        torch.manual_seed(seed)
        self.net = _network(len(self.classes), self.settings.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        self.net.train()
        for _ in range(self.settings.epochs):
            loss = F.cross_entropy(self.net(x), y)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        self.net.eval()
        return True

    def predict(self, curve: FloatArray) -> BowlerIdResult:
        """The most likely known bowler, or ``insufficient data``."""
        import torch

        if self.net is None:
            return BowlerIdResult(status="insufficient data", detail=self.status_detail)
        with torch.no_grad():
            p = torch.softmax(self.net(torch.tensor(curve[None], dtype=torch.float32))[0], dim=0)
        k = int(p.argmax())
        return BowlerIdResult(
            status="ok",
            bowler_id=self.classes[k],
            probability=float(p[k]),
            detail=self.status_detail,
        )
