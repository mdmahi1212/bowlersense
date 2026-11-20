"""Multi-delivery synthetic sessions: several generated deliveries joined by pauses."""

from __future__ import annotations

from collections.abc import Sequence

from bowlsense.schemas import FramePose, PoseSequence
from synth.skeleton_gen import SynthParams, SynthResult, generate


def multi_ball_session(
    params: Sequence[SynthParams], pause_s: float = 2.0
) -> tuple[PoseSequence, list[SynthResult], list[int]]:
    """Join deliveries (same fps) with ``pause_s`` of standing still between them.

    Returns the joined sequence, each delivery's generator result, and each release frame in
    the joined numbering.
    """
    if not params:
        raise ValueError("At least one delivery is needed.")
    fps = params[0].fps
    if any(p.fps != fps for p in params):
        raise ValueError("All deliveries must share one fps.")
    parts = [generate(p) for p in params]
    frames: list[FramePose] = []
    releases = []
    for j, r in enumerate(parts):
        src = list(r.sequence.frames)
        if j < len(parts) - 1:
            src += [src[-1]] * round(pause_s * fps)
        off = len(frames)
        releases.append(off + r.phases.release)
        frames += [
            FramePose(frame_idx=off + i, t=(off + i) / fps, keypoints=f.keypoints)
            for i, f in enumerate(src)
        ]
    return parts[0].sequence.model_copy(update={"frames": frames}), parts, releases
