"""The extractor registry and the frame-wise (timm) extractor's plumbing.

No weights are downloaded: the timm backbone is a stub module with a
``pretrained_cfg``, so what is pinned is the contract every extractor shares
— the sidecar's dims, clock and attrs — and timm's own eval transform
(resize to ``side / crop_pct``, centre crop, mean/std) reproduced on tensors.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from ethograph.io.schema import KIND, VIDEO_FEATURE  # noqa: E402
from ethograph.video_features import (  # noqa: E402
    EXTRACTORS,
    TIME_DIM,
    CropBox,
    build_extractor,
    feature_dim,
    feature_dim_of,
    plan_frames,
    time_dim_of,
)
from ethograph.video_features.base import to_dataarray  # noqa: E402
from ethograph.video_features.timm_extract import (  # noqa: E402
    DEFAULT_MODEL,
    DataConfig,
    TimmConfig,
    TimmExtractor,
    extract_timm,
    frame_features,
    preprocess,
)


class TestSidecarContract:
    def test_every_extractor_writes_the_same_shape_of_sidecar(self):
        da = to_dataarray(np.ones((5, 3), dtype=np.float32), name="x", video_fps=50.0, step=2, attrs={"a": 1})
        assert da.dims == (TIME_DIM, feature_dim("x")) and da.name == "x"
        np.testing.assert_allclose(da[TIME_DIM].values, np.arange(5) * 2 / 50.0)
        assert da.attrs[KIND] == VIDEO_FEATURE
        assert (da.attrs["extractor"], da.attrs["step"], da.attrs["effective_fps"], da.attrs["a"]) == ("x", 2, 25.0, 1)
        assert time_dim_of(da) == TIME_DIM and feature_dim_of(da) == "x_dims"

    def test_a_legacy_time_dim_is_still_found(self):
        da = to_dataarray(np.ones((2, 3), dtype=np.float32), name="s3d", video_fps=1.0, step=1, attrs={})
        assert time_dim_of(da.rename({TIME_DIM: "time_s3d"})) == "time_s3d"

    def test_registry_names_and_unknown_name(self):
        assert set(EXTRACTORS) == {"s3d", "timm"}
        with pytest.raises(ValueError, match="s3d.*timm"):
            build_extractor("vjepa")

    def test_plan_frames_never_upsamples(self):
        assert plan_frames(200.0, 25.0).step == 8
        assert plan_frames(30.0, 50.0).step == 1
        with pytest.raises(ValueError, match="read it from the video"):
            plan_frames(0.0, None)


class TestCrop:
    def test_box_is_cut_before_the_network(self):
        frames = np.arange(2 * 6 * 8 * 3, dtype=np.uint8).reshape(2, 6, 8, 3)
        cut = CropBox(2, 1, 5, 4).apply(frames)
        assert cut.shape == (2, 3, 3, 3)
        np.testing.assert_array_equal(cut, frames[:, 1:4, 2:5])

    def test_a_non_square_box_warns_with_what_is_lost(self, caplog):
        """The network centre-crops a square, so a 203x164 box loses 19 % of its width."""
        with caplog.at_level("WARNING", logger="ethograph.video_features.base"):
            CropBox(164, 0, 367, 164).validate(640, 480, "clip.mp4")
            assert "203x164" in caplog.text and "19% of the box's width" in caplog.text
            caplog.clear()
            CropBox(0, 0, 100, 100).validate(640, 480, "clip.mp4")
            assert caplog.text == ""

    def test_empty_or_outside_boxes_are_refused(self):
        with pytest.raises(ValueError, match="empty"):
            CropBox(5, 5, 5, 9)
        with pytest.raises(ValueError, match="outside"):
            CropBox(-1, 0, 4, 4)
        with pytest.raises(ValueError, match="reaches outside clip.mp4"):
            CropBox(0, 0, 100, 10).validate(64, 48, "clip.mp4")


class _Backbone(torch.nn.Module):
    """A timm-shaped stub: ``pretrained_cfg`` + ``num_features`` + a per-frame mean."""

    def __init__(self, side: int = 8, crop_pct: float = 1.0) -> None:
        super().__init__()
        self.pretrained_cfg = {
            "input_size": (3, side, side),
            "interpolation": "bilinear",
            "mean": (0.5, 0.5, 0.5),
            "std": (0.25, 0.25, 0.25),
            "crop_pct": crop_pct,
        }
        self.num_features = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, 3, H, W) → (N, 3)
        return x.mean(dim=(2, 3))


class TestTimmPreprocess:
    def test_data_config_is_read_off_the_model(self):
        data = DataConfig.of(_Backbone(side=8, crop_pct=0.875))
        assert (data.side, data.crop_pct, data.interpolation) == (8, 0.875, "bilinear")
        assert data.mean == (0.5, 0.5, 0.5) and data.std == (0.25, 0.25, 0.25)

    def test_resize_crop_normalise(self):
        data = DataConfig.of(_Backbone(side=8))
        frames = np.full((2, 12, 20, 3), 128, dtype=np.uint8)  # grey: 128/255 ≈ 0.502
        x = preprocess(frames, data, torch.device("cpu"))
        assert x.shape == (2, 3, 8, 8)
        expected = (128 / 255 - 0.5) / 0.25
        assert torch.allclose(x, torch.full_like(x, expected), atol=1e-6)

    def test_crop_pct_resizes_larger_then_crops_the_centre(self):
        data = DataConfig.of(_Backbone(side=8, crop_pct=0.5))
        x = preprocess(np.zeros((1, 32, 32, 3), dtype=np.uint8), data, torch.device("cpu"))
        assert x.shape == (1, 3, 8, 8)

    def test_frame_features_keep_one_row_per_frame_across_batches(self):
        chunks = [torch.rand(7, 3, 4, 4), torch.rand(5, 3, 4, 4)]
        out = torch.cat(list(frame_features(lambda x: x.mean(dim=(2, 3)), iter(chunks), batch=3)))
        assert out.shape == (12, 3)
        assert torch.allclose(out, torch.cat(chunks).mean(dim=(2, 3)))


class TestTimmExtractor:
    def test_name_and_plan(self):
        extractor = TimmExtractor(TimmConfig(analysis_fps=25.0))
        assert extractor.name == "timm"
        assert extractor.plan(200.0).step == 8
        assert TimmConfig().model_name == DEFAULT_MODEL

    def test_extract_streams_the_video_into_a_sidecar(self, tmp_path, monkeypatch):
        av = pytest.importorskip("av")
        clip = tmp_path / "clip.mp4"
        with av.open(str(clip), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for i in range(9):
                img = np.full((48, 64, 3), 20 + 20 * i, dtype=np.uint8)
                container.mux(stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
            container.mux(stream.encode(None))

        monkeypatch.setattr("ethograph.video_features.timm_extract.load_timm", lambda name, device: _Backbone())
        da = extract_timm(
            clip, TimmConfig(model_name="stub", analysis_fps=15.0, batch=2, chunk=4), crop=CropBox(0, 0, 32, 48)
        )
        assert da.dims == (TIME_DIM, "timm_dims") and da.shape == (5, 3)  # step 2 → frames 0,2,4,6,8
        np.testing.assert_allclose(da[TIME_DIM].values, np.arange(5) * 2 / 30.0)
        assert da.attrs["model_name"] == "stub" and da.attrs["crop"] == [0, 0, 32, 48]
        # brighter frames → larger normalised means, monotonically
        means = da.values.mean(axis=1)
        assert np.all(np.diff(means) > 0)
