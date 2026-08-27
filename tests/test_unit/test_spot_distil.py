"""Distillation plumbing, the VRAM preflight and the order flag — the parts
that fail silently when wrong: a student trained on the wrong teacher, a run
that pages instead of failing, a reordered event nobody reviews.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ethograph.spot.confidence import CurveStats
from ethograph.spot.config import config_from_dict
from ethograph.spot.inference import flag_out_of_order
from ethograph.spot.predict import SpottedEvent
from ethograph.spot.vendored import check_vram


def _config(tmp_path, **extra):
    source = tmp_path / "ses-01.nc"
    source.touch()
    data = {"sessions": [str(source)], "labels": {"classes": [31, 32]}, "root": str(tmp_path)}
    data.update(extra)
    return config_from_dict(data, tmp_path)


def _event(video, label, frame, focus=0.9, ratio=0.9):
    return SpottedEvent(
        video, label, float(frame), frame / 200.0, CurveStats(index=frame, peak=0.9, focus=focus, ratio=ratio)
    )


class TestOrderFlag:
    def test_in_order_trial_is_untouched(self, tmp_path):
        config = _config(tmp_path)
        events = [_event("v", 31, 100), _event("v", 32, 300)]
        out = flag_out_of_order(events, config)
        assert [e.confidence for e in out] == pytest.approx([0.81, 0.81])  # focus x ratio, untouched

    def test_out_of_order_trial_is_flagged_not_reordered(self, tmp_path):
        config = _config(tmp_path)
        events = [_event("v", 31, 300), _event("v", 32, 100)]
        out = flag_out_of_order(events, config)
        assert all(e.confidence == 0.0 for e in out)
        assert [(e.label, e.frame) for e in out] == [(31, 300.0), (32, 100.0)]  # kept as predicted

    def test_a_single_event_cannot_be_out_of_order(self, tmp_path):
        config = _config(tmp_path)
        out = flag_out_of_order([_event("v", 32, 100)], config)
        assert out[0].confidence == pytest.approx(0.81)

    def test_the_switch_is_config(self, tmp_path):
        assert _config(tmp_path).infer.flag_out_of_order is True
        assert _config(tmp_path, infer={"flag_out_of_order": False}).infer.flag_out_of_order is False


class TestPreflight:
    def test_no_cuda_means_nothing_to_check(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        check_vram(200)  # no raise

    def test_a_full_card_is_refused_naming_the_need(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2.0e9, 10.0e9))
        with pytest.raises(RuntimeError, match=r"2\.0 GB of 10\.0 GB free.*200 frames.*6\.0 GB"):
            check_vram(200)

    def test_a_free_card_passes(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (9.0e9, 10.0e9))
        check_vram(200)


class TestDistilConfig:
    def test_defaults_and_overrides(self, tmp_path):
        cfg = _config(tmp_path, distil={"epochs": 3, "head_epochs": 1, "teacher_run": "ctx2s_res10ms_gcn"})
        assert (cfg.distil.epochs, cfg.distil.head_epochs, cfg.distil.teacher_run) == (3, 1, "ctx2s_res10ms_gcn")
        assert cfg.distil.init_run is None

    def test_the_feature_list_is_fingerprinted_into_the_teacher_folder(self, tmp_path):
        from ethograph.spot.config import features_fingerprint
        from ethograph.spot.teacher import teacher_run_dir

        features = {"speed": {"keypoint": ["stickTip"]}, "pellet_stickClosest_dist": {}}
        cfg = _config(tmp_path, features=features)
        same = _config(tmp_path, features=dict(features))
        assert features_fingerprint(cfg) == features_fingerprint(same)
        edited = _config(tmp_path, features={**features, "velocity": {"space": ["x", "y"]}})
        assert features_fingerprint(edited) != features_fingerprint(cfg)
        clip = cfg.clip.resolve(200.0)
        assert teacher_run_dir(cfg, clip).name == f"ctx2s_res10ms_{features_fingerprint(cfg)}"
        assert teacher_run_dir(edited, clip).parent == teacher_run_dir(cfg, clip).parent

    def test_embeddings_live_beside_the_features(self, tmp_path):
        cfg = _config(tmp_path)
        assert cfg.embeddings_dir == cfg.features_dir / "embeddings"


class TestTeacherEmbeddingsFile:
    def test_round_trips_stride_and_dim(self, tmp_path):
        """What the student's loader reads back must be what the teacher wrote."""
        path = tmp_path / "v.npz"
        np.savez(path, embedding=np.zeros((50, 256), np.float32), stride=np.int64(2), fps=np.float64(200.0))
        with np.load(path) as npz:
            assert int(npz["stride"]) == 2 and npz["embedding"].shape == (50, 256)
        info = {"run": "teacher/x", "epoch": 1, "dim": 256, "n_clips": 1}
        (tmp_path / "teacher.json").write_text(json.dumps(info))
        assert json.loads((tmp_path / "teacher.json").read_text())["dim"] == 256
