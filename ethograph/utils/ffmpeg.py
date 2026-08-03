"""Locate an ffmpeg executable for the one feature that shells out (proxy encode).

Resolution order: the ``ETHOGRAPH_FFMPEG`` environment override, then an ffmpeg
on ``PATH``, then the binary bundled with ``imageio-ffmpeg`` (the ``proxy``
extra). Proxy generation is the only caller — it is a performance optimisation,
so a missing binary degrades to full-resolution playback rather than an error.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache


class FfmpegNotFoundError(RuntimeError):
    """Raised when no usable ffmpeg executable can be located."""


@lru_cache(maxsize=1)
def ffmpeg_executable() -> str:
    """Locate an ffmpeg executable: env override, then PATH, then bundled wheel."""
    if override := os.environ.get("ETHOGRAPH_FFMPEG"):
        return override
    if found := shutil.which("ffmpeg"):
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        raise FfmpegNotFoundError(
            'No ffmpeg found. Install it with: uv pip install "ethograph[proxy]", '
            "conda install -c conda-forge ffmpeg, or see the installation docs."
        ) from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg_available() -> bool:
    """Return True if an ffmpeg executable can be located."""
    try:
        ffmpeg_executable()
    except FfmpegNotFoundError:
        return False
    return True
