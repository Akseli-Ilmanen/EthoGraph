"""Plan an S3D extraction: a configuration in **seconds** becomes frame counts
for the video at hand.

The rate is never a setting — it is read from the video (``probe_video``)
and the plan derives everything else from it, so the same configuration
means the same thing on a 30 fps webcam and a 200 fps high-speed camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Fewest frames one S3D window may hold. The trunk pools time 8× (a
#: stride-2 conv, a stride-2 pool and a final *unpadded* 2-pool) and the
#: window head then averages *pairs* of positions, so it needs two of them:
#: 13 → 7 → 4 → 2. Below this the temporal axis collapses.
MIN_STACK = 13

S3DMode = Literal["windows", "dense"]
S3DPrecision = Literal["fp16", "fp32"]


@dataclass(frozen=True)
class S3DConfig:
    """What the user decides, in physical units.

    analysis_fps
        The rate S3D should see. ``None`` = every frame. Otherwise frames are
        *skipped* (never interpolated up) to get as close as possible — a
        30 fps video with ``analysis_fps=25`` keeps every frame.
    stack_s
        Temporal extent of one S3D window, in seconds. Each frame's feature
        is the network's embedding of the window centred on it.
    mode
        ``"windows"`` — one embedding per window, exactly the sliding-stack
        scheme, batched. ``"dense"`` — the convolutional trunk runs once over
        the whole video (its full ~99-frame receptive field), spatially
        averaged, smoothed over ``stack_s`` and interpolated to every frame;
        far cheaper, different features — an ablation, not a default.
    truncate_at
        Dense mode only: cut the trunk at a named stage (see
        ``s3d.S3D_STAGES``) for shorter-context, lower-level features.
    batch / chunk
        Windows per forward pass; frames decoded per step (and the dense
        core length).
    precision
        ``"fp16"`` autocasts on CUDA (identical up to rounding); ``"fp32"``
        reproduces bit-exactly.
    """

    analysis_fps: float | None = None
    stack_s: float = 0.1
    mode: S3DMode = "windows"
    truncate_at: str | None = None
    batch: int = 16
    chunk: int = 128
    precision: S3DPrecision = "fp16"


@dataclass(frozen=True)
class S3DPlan:
    """The configuration resolved against one video's rate, in frames."""

    video_fps: float
    #: Keep every ``step``-th frame.
    step: int
    #: Frames per window — odd, so a window is centred on its frame.
    stack_frames: int

    @property
    def effective_fps(self) -> float:
        return self.video_fps / self.step

    @property
    def stack_s(self) -> float:
        return self.stack_frames / self.effective_fps

    def describe(self) -> str:
        return (
            f"stack = {self.stack_frames} frames = {self.stack_s:.3f} s at "
            f"{self.effective_fps:g} fps (video {self.video_fps:g} fps, step {self.step})"
        )


def plan_s3d(video_fps: float, cfg: S3DConfig) -> S3DPlan:
    """Resolve *cfg* for a video at *video_fps*.

    Raises ``ValueError`` when the window would be too short for the network
    at that rate, naming the shortest ``stack_s`` that works — a silent clamp
    would quietly change what the features mean.
    """
    if video_fps <= 0:
        raise ValueError("video_fps must be positive — read it from the video, never default it.")
    if cfg.analysis_fps is None:
        step = 1
    else:
        if cfg.analysis_fps <= 0:
            raise ValueError("analysis_fps must be positive (or None for every frame).")
        step = max(1, int(round(video_fps / cfg.analysis_fps)))
    effective_fps = video_fps / step
    stack = int(round(cfg.stack_s * effective_fps))
    if stack % 2 == 0:
        stack += 1
    if stack < MIN_STACK:
        raise ValueError(
            f"stack_s={cfg.stack_s:g} s is {stack} frame(s) at {effective_fps:g} fps; S3D needs at "
            f"least {MIN_STACK}. Use stack_s >= {MIN_STACK / effective_fps:.3f} s"
            + (" or a higher analysis_fps." if cfg.analysis_fps is not None else ".")
        )
    return S3DPlan(video_fps=float(video_fps), step=step, stack_frames=stack)
