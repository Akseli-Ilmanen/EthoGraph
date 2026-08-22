"""S3D extraction: streaming decode, batched windows, dense trunk, time axis.

The window/dense plumbing is checked with stub networks (exact, fast); the
real S3D runs once on a tiny synthetic clip so the end-to-end path — probe,
plan, preprocess, checkpoint, DataArray — is exercised too.
"""

import numpy as np
import pytest

av = pytest.importorskip("av")
torch = pytest.importorskip("torch")

from ethograph.video_features.extract import (  # noqa: E402
    CHECKPOINT,
    FEATURE_DIM,
    TIME_DIM,
    dense_positions,
    dense_to_frames,
    extract_s3d,
    preprocess,
    window_features,
)
from ethograph.video_features.frames import iter_frame_chunks, iter_frames, probe_video  # noqa: E402
from ethograph.video_features.plan import S3DConfig  # noqa: E402
from ethograph.video_features.s3d import S3D_STAGES, S3DStage  # noqa: E402


def _make_clip(path, n_frames, width=64, height=48, fps=30):
    """A clip whose frame *i* has mean brightness that identifies it."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            img = np.full((height, width, 3), 20 + 10 * i, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
        container.mux(stream.encode(None))


# ----------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------


class TestFrames:
    def test_step_only_skips(self, tmp_path):
        """``step=2`` yields exactly every other frame of ``step=1``."""
        clip = tmp_path / "clip.mp4"
        _make_clip(clip, 12)
        every = list(iter_frames(clip))
        half = list(iter_frames(clip, step=2))
        assert len(every) == 12 and len(half) == 6
        for a, b in zip(every[::2], half):
            assert np.array_equal(a, b)

    def test_chunks_tile_the_video(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip, 11)
        chunks = list(iter_frame_chunks(clip, chunk=4))
        assert [c.shape[0] for c in chunks] == [4, 4, 3]
        assert chunks[0].shape[1:] == (48, 64, 3)
        assert probe_video(str(clip)).fps == 30.0

    def test_preprocess_is_224_square_in_unit_range(self):
        frames = np.random.default_rng(0).integers(0, 255, (3, 48, 64, 3), dtype=np.uint8)
        x = preprocess(frames, torch.device("cpu"))
        assert x.shape == (3, 3, 224, 224)
        assert 0.0 <= x.min() and x.max() <= 1.0


# ----------------------------------------------------------------------
# Windows (stub embedder: the centre frame's per-channel mean)
# ----------------------------------------------------------------------


def _centre_mean(stack):
    def embed(x):  # (B, 3, stack, H, W) → (B, 3)
        assert x.shape[2] == stack
        return x[:, :, stack // 2].mean(dim=(2, 3))

    return embed


def _frames(n, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).random((n, 3, 8, 8), dtype=np.float32))


class TestWindows:
    def test_one_row_per_frame_centred_on_it(self):
        frames = _frames(10)
        out = torch.cat(list(window_features(_centre_mean(5), iter([frames]), stack=5, batch=3)))
        assert out.shape == (10, 3)
        assert torch.allclose(out, frames.mean(dim=(2, 3)))

    def test_chunking_and_batching_do_not_change_the_result(self):
        """The rolling carry joins chunks seamlessly, whatever their size."""
        frames = _frames(23)
        reference = torch.cat(list(window_features(_centre_mean(7), iter([frames]), stack=7, batch=64)))
        for chunk in (1, 4, 7, 10):
            pieces = [frames[i : i + chunk] for i in range(0, 23, chunk)]
            out = torch.cat(list(window_features(_centre_mean(7), iter(pieces), stack=7, batch=2)))
            assert torch.allclose(out, reference)

    def test_the_ends_see_zero_frames(self):
        """Beyond the video the stack is black — the first window's left half is zeros."""
        seen = []

        def embed(x):
            seen.append(x.clone())
            return x.mean(dim=(2, 3, 4))

        frames = _frames(4) + 1.0  # strictly positive, so zeros are unmistakable
        list(window_features(embed, iter([frames]), stack=5, batch=64))
        first = torch.cat(seen)[0]  # (3, 5, H, W)
        assert torch.all(first[:, :2] == 0) and torch.all(first[:, 2:] > 0)

    def test_even_stack_is_refused(self):
        with pytest.raises(ValueError, match="odd"):
            list(window_features(_centre_mean(4), iter([_frames(4)]), stack=4, batch=1))


# ----------------------------------------------------------------------
# Dense (stub trunk: stride-2 positions carrying their centre frame's mean)
# ----------------------------------------------------------------------


class TestDense:
    STAGE = S3DStage(modules=0, stride=2, offset=0, receptive_field=5, channels=3)

    @staticmethod
    def _trunk(x):  # (1, 3, L, H, W) → (1, 3, L // 2, 1, 1): position m = frame 2m
        return x[:, :, ::2].mean(dim=(3, 4), keepdim=True)

    def test_positions_are_centred_where_the_stage_says(self):
        frames = _frames(20)
        centres, feats = dense_positions(self._trunk, self.STAGE, iter([frames[:9], frames[9:]]), core=6)
        assert centres.tolist() == list(range(0, 20, 2))
        assert np.allclose(feats, frames[::2].mean(dim=(2, 3)).numpy())

    def test_interpolation_hits_every_frame_and_smooths(self):
        centres = np.array([0, 2, 4])
        feats = np.array([[0.0], [2.0], [4.0]])
        dense = dense_to_frames(centres, feats, n_frames=6, stack=1)
        assert dense[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 4.0]  # flat past the last position
        smoothed = dense_to_frames(centres, feats, n_frames=6, stack=3)
        assert smoothed.shape == (6, 1) and smoothed[2, 0] == pytest.approx(2.0)


# ----------------------------------------------------------------------
# End to end with the real network
# ----------------------------------------------------------------------


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="S3D checkpoint not present")
class TestExtract:
    def test_windows_dataarray_on_its_own_time_axis(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip, 12)
        da = extract_s3d(clip, S3DConfig(stack_s=0.45, precision="fp32"))  # 15 frames at 30 fps
        assert da.dims == (TIME_DIM, FEATURE_DIM) and da.shape == (12, 1024)
        assert np.allclose(da[TIME_DIM].values, np.arange(12) / 30.0)
        assert np.isfinite(da.values).all()
        assert da.attrs["stack_frames"] == 15 and da.attrs["step"] == 1 and da.attrs["mode"] == "windows"

    def test_subsampled_time_axis(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip, 12)
        da = extract_s3d(clip, S3DConfig(analysis_fps=15.0, stack_s=1.0, precision="fp32"))  # step 2 → 15 fps
        assert da.shape[0] == 6
        assert np.allclose(da[TIME_DIM].values, np.arange(6) * 2 / 30.0)
        assert da.attrs["effective_fps"] == 15.0

    def test_dense_and_truncated(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        _make_clip(clip, 12)
        da = extract_s3d(clip, S3DConfig(stack_s=0.45, mode="dense", precision="fp32"))
        assert da.shape == (12, 1024) and np.isfinite(da.values).all()
        da3 = extract_s3d(clip, S3DConfig(stack_s=0.45, mode="dense", truncate_at="Mixed_3c", precision="fp32"))
        assert da3.shape == (12, S3D_STAGES["Mixed_3c"].channels) and da3.attrs["stage"] == "Mixed_3c"

    def test_truncate_needs_dense(self, tmp_path):
        with pytest.raises(ValueError, match="dense"):
            extract_s3d(tmp_path / "x.mp4", S3DConfig(truncate_at="Mixed_3c"))
