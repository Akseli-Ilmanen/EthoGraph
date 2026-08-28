"""Sequential decode of a video into RGB frames, Qt-free.

A codec reads front to back cheaply; what costs is turning each decoded frame
into RGB. ``VideoFrame.to_ndarray(format="rgb24")`` builds a fresh swscale
context per call, and for the full-range ``yuvj420p`` cameras write that
set-up is ~5 ms a frame — sixteen times the H.264 decode itself (a 512×562
stream: 180 fps converting per call, 2,980 with one context kept for the whole
video; the pixels are bit-identical). :class:`RGBConverter` keeps that context,
and :func:`iter_rgb_frames` converts only the frames its caller keeps.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
from av.video.reformatter import VideoReformatter


def decode_frames(path: str | Path, *, threads: int | None = None) -> Iterator[av.VideoFrame]:
    """The decoded frames of *path* in order, still in the codec's own pixel format.

    *threads* caps the codec's own threads (``None`` = the codec decides):
    one per container when several containers decode at once.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if threads is not None:
            stream.codec_context.thread_count = int(threads)
        yield from container.decode(stream)


class RGBConverter:
    """``VideoFrame`` → ``(H, W, 3)`` RGB ``uint8``, with one conversion context for every frame.

    One per decode, never shared between threads: the context is not.
    """

    def __init__(self) -> None:
        self._reformatter = VideoReformatter()

    def __call__(self, frame: av.VideoFrame) -> np.ndarray:
        return self._reformatter.reformat(frame, format="rgb24").to_ndarray()


def iter_rgb_frames(path: str | Path, *, step: int = 1, threads: int | None = None) -> Iterator[np.ndarray]:
    """Every *step*-th frame of *path* as RGB; the frames between are decoded and never converted."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    to_rgb = RGBConverter()
    for i, frame in enumerate(decode_frames(path, threads=threads)):
        if i % step == 0:
            yield to_rgb(frame)
