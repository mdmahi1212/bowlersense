"""Export MotionBERT (3D pose lifting) to ONNX for the optional ``pose.lift_3d`` feature.

BowlSense works fully without 3D lifting. To enable it:

1. Get the MotionBERT code and a 3D-pose checkpoint (one-time, needs the internet):
   ``git clone https://github.com/Walter0807/MotionBERT`` and download the "MotionBERT-Lite
   (3D pose)" checkpoint linked from its README (``best_epoch.bin``).
2. Export it (offline from here on):
   ``ml/.venv/Scripts/python scripts/export_motionbert.py --repo path/to/MotionBERT
   --checkpoint path/to/best_epoch.bin``
   This writes ``models/<pose.lift_model>`` (default ``motionbert_lite.onnx``).
3. Set ``pose.lift_3d: true`` in ``config/default.yaml``.

Needs the ``onnx`` package (in requirements-dev.txt).

The network settings below are MotionBERT-Lite's (``configs/pose3d/MB_ft_h36m_global_lite``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

LITE = {"dim_in": 3, "dim_out": 3, "dim_feat": 256, "dim_rep": 512, "depth": 5, "num_heads": 8,
        "mlp_ratio": 4, "num_joints": 17, "maxlen": 243}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    """Build MotionBERT-Lite from a local checkout, load the checkpoint, export ONNX."""
    from bowlsense.config import load_settings

    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo", type=Path, required=True, help="MotionBERT git checkout")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best_epoch.bin")
    parser.add_argument(
        "--out", type=Path, default=settings.storage.models_path / settings.pose.lift_model
    )
    args = parser.parse_args(argv)
    if not (args.repo / "lib" / "model" / "DSTformer.py").is_file():
        print(f"{args.repo} is not a MotionBERT checkout (lib/model/DSTformer.py missing).")
        return 2
    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}")
        return 2
    import torch

    sys.path.insert(0, str(args.repo))
    from lib.model.DSTformer import DSTformer  # type: ignore[import-not-found]

    model = DSTformer(**LITE)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_pos"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    model.eval()
    window = settings.pose.lift_window
    dummy = torch.zeros(1, window, 17, 3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (dummy,), str(args.out), input_names=["keypoints_2d"],
                      output_names=["keypoints_3d"], opset_version=17, dynamo=False)  # fmt: skip
    print(f"wrote {args.out}; set pose.lift_3d: true to use it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
