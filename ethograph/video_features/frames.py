"""Streaming frames for feature extraction — a video is never read whole.

Decodes with PyAV in order, yielding RGB ``uint8`` frames (or chunks of
them) every ``step``-th frame, so memory is bounded by the chunk size
whatever the recording's length.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ethograph.io.video_decode import iter_rgb_frames
from ethograph.io.video_probe import VideoProbe, probe_video

__all__ = ["VideoProbe", "iter_frame_chunks", "iter_frames", "probe_video"]


def iter_frames(path: str | Path, *, step: int = 1) -> Iterator[np.ndarray]:
    """Yield every *step*-th frame as an ``(H, W, 3)`` RGB ``uint8`` array."""
    return iter_rgb_frames(path, step=step)


def iter_frame_chunks(path: str | Path, *, step: int = 1, chunk: int = 128) -> Iterator[np.ndarray]:
    """Yield ``(N, H, W, 3)`` RGB ``uint8`` batches, ``N <= chunk``, in order."""
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    buf: list[np.ndarray] = []
    for frame in iter_frames(path, step=step):
        buf.append(frame)
        if len(buf) == chunk:
            yield np.stack(buf)
            buf = []
    if buf:
        yield np.stack(buf)
