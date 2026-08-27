"""The pose teacher: the shift reads past and future, the model keeps E2E-Spot's output contract, and the
strided clock is one clock.

What earns a test: a shift longer than the clip must zero the shifted
channels rather than wrap; the target must be dilated on the strided clock
exactly as the pixel dataset does it; the statistics must be the training
split's and a missing value must read as 0 after them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ethograph.spot.config import TeacherConfig, config_from_dict
from ethograph.spot.features import NAMES_FILE, Stats, write_trial_features
from ethograph.spot.pose_model import PoseSpotter, shift_channels
from ethograph.spot.teacher import load_trials, sample_clip


class TestShiftScales:
    def test_defaults_are_umeg_nets_at_25_fps(self):
        assert TeacherConfig().shift_samples(25.0) == [1, 2, 4]

    def test_same_durations_at_200_fps(self):
        assert TeacherConfig().shift_samples(200.0) == [8, 16, 32]

    def test_a_scale_below_one_sample_rounds_up_and_deduplicates(self):
        assert TeacherConfig(shift_scales_ms=[1.0, 2.0, 40.0]).shift_samples(25.0) == [1]


class TestShift:
    def test_reads_past_and_future_with_zero_padding(self):
        h = torch.zeros(1, 5, 4)
        h[0, :, 2] = torch.arange(5.0)  # the "past" channel
        h[0, :, 3] = torch.arange(5.0)  # the "future" channel
        out = shift_channels(h, offset=1, n_shift=1)
        assert out[0, :, 2].tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]
        assert out[0, :, 3].tolist() == [1.0, 2.0, 3.0, 4.0, 0.0]
        assert (out[0, :, :2] == 0).all()

    def test_offset_longer_than_the_clip_zeroes_the_shifted_channels(self):
        h = torch.ones(1, 3, 4)
        out = shift_channels(h, offset=10, n_shift=1)
        assert (out[0, :, 2:] == 0).all() and (out[0, :, :2] == 1).all()


class TestModel:
    def test_emits_e2e_spots_contract_and_an_embedding_to_distil_from(self):
        model = PoseSpotter(n_features=3, n_classes=2, scales=[1, 2], hidden=16, head_hidden=8)
        x = torch.rand(2, 20, 3)
        x[0, 4, 1] = float("nan")  # a missing value reads as 0
        out = model(x)
        assert out.shape == (2, 20, 3)
        assert torch.isfinite(out).all()
        assert model.features(x).shape == (2, 20, model.embed_dim) and model.embed_dim == 16

    def test_the_wrong_width_is_refused(self):
        model = PoseSpotter(n_features=3, n_classes=2, scales=[1], hidden=16)
        with pytest.raises(ValueError, match="expected"):
            model(torch.rand(1, 10, 4))
        with pytest.raises(ValueError, match="features: is empty"):
            PoseSpotter(n_features=0, n_classes=2, scales=[1])


class TestTrials:
    def _config(self, tmp_path):
        source = tmp_path / "ses-01.nc"
        source.touch()
        return config_from_dict(
            {
                "sessions": [str(source)],
                "labels": {"classes": [31, 32]},
                "root": str(tmp_path),
                "features": {"speed": {}},
            },
            tmp_path,
        )

    def test_targets_are_dilated_on_the_strided_clock(self, tmp_path):
        cfg = self._config(tmp_path)
        cfg.features_dir.mkdir(parents=True)
        (cfg.features_dir / NAMES_FILE).write_text('{"names": ["speed"]}')
        time = np.arange(400) / 200.0
        write_trial_features(cfg.features_dir / "v.npz", time, np.ones((400, 1), np.float32), {31: 0.5, 32: 1.2})
        clip = cfg.clip.resolve(200.0)  # stride 2, dilate 1 at the default 2 s / 10 ms / 10 ms
        (trial,) = load_trials(cfg, ["v"], clip)
        assert trial.x.shape == (200, 1) and trial.fps == 100.0
        centre = 100 // clip.stride  # 0.5 s at 200 fps is frame 100
        assert trial.target[centre - clip.dilate_len : centre + clip.dilate_len + 1].tolist() == [1] * (
            2 * clip.dilate_len + 1
        )
        assert trial.target[240 // clip.stride] == 2
        assert (trial.target == 0).sum() == 200 - 2 * (2 * clip.dilate_len + 1)

    def test_a_clip_past_the_end_is_zero_padded(self, tmp_path):
        cfg = self._config(tmp_path)
        cfg.features_dir.mkdir(parents=True)
        (cfg.features_dir / NAMES_FILE).write_text('{"names": ["speed"]}')
        write_trial_features(cfg.features_dir / "v.npz", np.arange(50) / 200.0, np.ones((50, 1), np.float32), {})
        (trial,) = load_trials(cfg, ["v"], cfg.clip.resolve(200.0))
        x, y = sample_clip(trial, clip_len=60, rng=np.random.default_rng(0))
        assert x.shape == (60, 1) and y.shape == (60,)
        assert (x[25:] == 0).all()

    def test_stats_are_per_column_and_a_missing_value_reads_as_zero(self):
        a = np.array([[1.0, 10.0], [3.0, 30.0], [np.nan, 50.0]], np.float32)
        stats = Stats([a])
        scaled = stats.apply(a)
        np.testing.assert_allclose(stats.mean, [2.0, 30.0])
        assert scaled[2, 0] == 0.0 and scaled[0, 1] < 0 < scaled[2, 1]
