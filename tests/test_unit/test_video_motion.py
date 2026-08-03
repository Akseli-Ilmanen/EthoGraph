"""Unit tests for PyAV-based video-motion extraction.

``extract_video_motion`` computes YDIF (mean absolute luma difference between
consecutive frames) entirely in-process via PyAV — no ``ffmpeg`` executable is
required. These tests synthesise a short clip with PyAV and check the shape,
first-frame convention, and that a moving segment reads as higher motion than a
static one.
"""

import numpy as np
import pytest

av = pytest.importorskip("av")

from ethograph.features.movement import extract_video_motion  # noqa: E402
from ethograph.io.validation import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS  # noqa: E402


def _make_clip(path, n_static, n_moving, width=64, height=48, fps=30):
    """Write a clip: ``n_static`` still frames then ``n_moving`` with motion."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for i in range(n_static + n_moving):
            img = np.full((height, width, 3), 40, dtype=np.uint8)
            if i >= n_static:
                # Bright square sweeping left→right = large luma difference.
                x = (i - n_static) * 6 % (width - 12)
                img[10:30, x : x + 12] = 230
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def test_extract_video_motion_shape_and_first_frame(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, n_static=15, n_moving=15)

    da = extract_video_motion(clip, fps=30.0, verbose=False)

    assert da.ndim == 1
    assert len(da) == 30
    assert da.values[0] == 0.0
    np.testing.assert_allclose(da["time"].values, np.arange(30) / 30.0)


def test_extract_video_motion_detects_movement(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, n_static=15, n_moving=15)

    motion = extract_video_motion(clip, fps=30.0, verbose=False).values

    # Skip the first frame of each segment (0.0 / the static→moving boundary).
    static = motion[1:15]
    moving = motion[16:]
    assert moving.mean() > static.mean()


def test_audio_extensions_exclude_video_containers():
    # Regression guard for the deleted MP4 audio branch: re-adding a video
    # container extension here would resurrect dead code paths.
    assert AUDIO_EXTENSIONS.isdisjoint(VIDEO_EXTENSIONS)
    assert ".mp4" not in AUDIO_EXTENSIONS
    assert ".mov" not in AUDIO_EXTENSIONS
    assert ".avi" not in AUDIO_EXTENSIONS
