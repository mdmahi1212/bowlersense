"""GPU use: optional, auto-detected, and switchable off (settings page or env)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

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
    cuda_ort = _onnx_cuda_works()
    return {"torch_cuda": cuda_torch, "onnxruntime_cuda": cuda_ort, "enabled": gpu_allowed()}


def onnx_cuda_ready() -> bool:
    """True if ONNX Runtime can run on the GPU (and GPU use is on).

    onnxruntime-gpu needs the CUDA and cuDNN DLLs; the CUDA build of PyTorch ships them, and
    importing torch first puts them on the DLL search path. Then a tiny CUDA session is
    created to confirm the provider really loads (listing it isn't enough).
    """
    if not gpu_allowed():
        return False
    return _onnx_cuda_works()


@lru_cache(maxsize=1)
def _onnx_cuda_works() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        ort.preload_dlls()
    except (ImportError, OSError, AttributeError) as exc:
        log.info("ONNX Runtime GPU unavailable: %s", exc)
        return False
    return True
