"""GPU use: optional, auto-detected, and switchable off (settings page or env)."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)
DISABLE_ENV = "BOWLSENSE_DISABLE_GPU"


def gpu_allowed() -> bool:
    """False when ``BOWLSENSE_DISABLE_GPU=1`` (the settings page's "GPU off")."""
    return os.environ.get(DISABLE_ENV, "").strip() not in {"1", "true", "yes"}


def set_gpu_allowed(allowed: bool) -> None:
    """Switch GPU use on or off for models loaded after this call."""
    if allowed:
        os.environ.pop(DISABLE_ENV, None)
    else:
        os.environ[DISABLE_ENV] = "1"


def gpu_status() -> dict[str, bool]:
    """Which GPU runtimes are available (torch CUDA, onnxruntime CUDA) and whether use is on."""
    cuda_torch = cuda_ort = False
    try:
        import torch

        cuda_torch = bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover - torch is a dependency
        pass
    try:
        import onnxruntime as ort

        cuda_ort = "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:  # pragma: no cover - onnxruntime is a dependency
        pass
    return {"torch_cuda": cuda_torch, "onnxruntime_cuda": cuda_ort, "enabled": gpu_allowed()}
