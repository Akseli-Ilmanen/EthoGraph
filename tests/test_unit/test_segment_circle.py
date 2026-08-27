"""The circle term and the config reader's nested blocks, end to end through the real pipeline.

One synthetic session whose label bouts are where a keypoint's speed is high —
the same shape as ``test_segment_pipeline``'s fixture — with only the objective
changed between runs. That is the whole point of the design: an experiment is a
config, not a fork.

What is asserted here is the wiring, not the accuracy. Two epochs on six
synthetic trials say nothing about F1@90; they say that the term reaches the
loss and that what a run records is what it trained with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

torch = pytest.importorskip("torch")

import ethograph as eto  # noqa: E402
from ethograph.labels.intervals import LABELING_MANUAL  # noqa: E402
from ethograph.labels.tsv_store import save_labels_tsv  # noqa: E402
from ethograph.segment.config import load_config  # noqa: E402

FS = 50.0
DURATION = 8.0
TRIALS = [1, 2, 3, 4, 5, 6]
KEYPOINTS = ["beak", "tail"]


def _trial_ds(trial: int) -> tuple[xr.Dataset, list[tuple[float, float]]]:
    rng = np.random.default_rng(trial)
    t = np.arange(0.0, DURATION, 1.0 / FS)
    pos = rng.normal(0, 0.2, size=(t.size, 2, len(KEYPOINTS), 1))
    bouts = [(1.0 + 0.2 * (trial % 3), 2.0 + 0.2 * (trial % 3)), (4.5, 5.5)]
    for on, off in bouts:
        m = (t >= on) & (t <= off)
        pos[m] += np.sin(t[m, None, None, None] * 40) * 3.0
    speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * FS
    ds = xr.Dataset(
        {
            "position": (("time", "space", "keypoint", "individual"), pos),
            "speed": (("time", "keypoint", "individual"), speed),
        },
        coords={"time": t, "space": ["x", "y"], "keypoint": KEYPOINTS, "individual": ["A"]},
        attrs={"trial": trial, "fps": FS},
    )
    return ds, bouts


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A minimal one-session project, ready to train any architecture."""
    folder = tmp_path / "sessions" / "s1"
    folder.mkdir(parents=True)
    datasets, rows = [], []
    for trial in TRIALS:
        ds, bouts = _trial_ds(trial)
        datasets.append(ds)
        rows += [
            {
                "trial": trial,
                "individual": "A",
                "individual_rec": "",
                "labels": 3,
                "onset_s": on,
                "offset_s": off,
                "event_type": "state",
                "confidence": 1.0,
                "labeling_method": LABELING_MANUAL,
                "changepoint_corrected": 0,
                "prediction_source": "",
                "n_samples": int(DURATION * FS),
            }
            for on, off in bouts
        ]
    nc_path = folder / "s1.nc"
    eto.from_datasets(datasets).save(str(nc_path))
    save_labels_tsv(folder / "s1_labels.tsv", pd.DataFrame(rows))

    root = tmp_path / "project"
    root.mkdir()
    (root / "mapping.txt").write_text("0 background\n3 flap\n", encoding="utf-8")
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "sessions": [{"source": str(nc_path), "labels_path": str(folder / "s1_labels.tsv")}],
                "features": {
                    "name": "kin",
                    "columns": {
                        "position": {"space": ["x", "y"], "keypoint": KEYPOINTS},
                        "speed": {"keypoint": ["beak"]},
                    },
                    "labels": {"mapping": "mapping.txt", "branch": 0},
                },
                "train": {
                    "epochs": 2,
                    "eval_every": 1,
                    "split": {"train_fraction": 0.5, "val_fraction": 0.25, "test_fraction": 0.25},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root


def _run(project_dir: Path, overrides: list[str], name: str):
    return eto.segment.Project(project_dir / "config.yaml", *overrides, f"train.run_name={name}").train()


class TestCircle:
    """The circle (deep metric-learning) term — architecture-agnostic."""

    def test_a_run_trains_with_the_term_on_and_records_its_settings(self, project: Path) -> None:
        result = _run(
            project,
            ["train.circle.weight=0.5", "train.circle.max_frames=64", "train.device=cpu", "train.batch_size=1"],
            "circle_smoke",
        )
        assert result.test_metrics is not None
        metrics = pd.read_csv(result.run_dir / "metrics.tsv", sep="\t")
        assert np.isfinite(metrics["loss"]).all()
        recorded = yaml.safe_load((result.run_dir / "test_metrics.yaml").read_text(encoding="utf-8"))
        circle = recorded["objective"]["circle"]
        assert circle == {"weight": 0.5, "m": 0.25, "gamma": 128.0, "max_frames": 64}

    def test_default_weight_leaves_it_untrained(self, project: Path) -> None:
        result = _run(project, ["train.device=cpu", "train.batch_size=1"], "circle_off")
        recorded = yaml.safe_load((result.run_dir / "test_metrics.yaml").read_text(encoding="utf-8"))
        assert recorded["objective"]["circle"]["weight"] == 0.0


class TestEmptyBlocks:
    """``circle:`` with nothing after it, and an explicit ``null`` that means something.

    A nested block written empty is "use the defaults" — but
    ``max_frames: null`` is not the same statement: ``None`` is the *value*
    that says "pool every unpadded frame". A config reader that treats every
    null as "absent" would silently pin whatever the default happened to be.
    """

    def _written(self, project: Path, block: str) -> str:
        (project / "empty.yaml").write_text(f"base: config.yaml\ntrain:\n{block}", encoding="utf-8")
        return str(project / "empty.yaml")

    def test_an_empty_circle_block_takes_the_defaults(self, project: Path) -> None:
        from ethograph.segment.config import CircleConfig

        cfg = load_config(self._written(project, "  circle:\n  # off, as it ships\n"))
        assert isinstance(cfg.train.circle, CircleConfig)
        assert cfg.train.circle.weight == 0.0
        assert cfg.train.circle.max_frames == 2048

    def test_an_explicit_null_max_frames_survives(self, project: Path) -> None:
        cfg = load_config(self._written(project, "  circle:\n    weight: 0.5\n    max_frames: null\n"))
        assert cfg.train.circle.max_frames is None
        assert cfg.train.circle.weight == 0.5

    def test_a_pinned_max_frames_reaches_the_loss(self, project: Path) -> None:
        from ethograph.segment.losses import build_objective

        cfg = load_config(self._written(project, "  circle:\n    weight: 0.5\n    max_frames: 64\n"))
        objective, settings = build_objective(cfg, n_classes=2)
        assert objective.circle_max_frames == 64
        assert settings["circle"]["max_frames"] == 64
