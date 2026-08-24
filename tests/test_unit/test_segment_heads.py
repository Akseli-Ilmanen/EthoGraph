"""The boundary head and the query head, end to end through the real pipeline.

One synthetic session whose label bouts are where a keypoint's speed is high —
the same shape as ``test_segment_pipeline``'s fixture — with only the
architecture and the objective changed between runs. That is the whole point of
the design: an experiment is a config, not a fork.

What is asserted here is the wiring, not the accuracy. Two epochs on six
synthetic trials say nothing about F1@90; they say that the boundary target
reaches the loss, that the loss reaches the head, that the head's curve reaches
post-processing, and that a run without a head still behaves exactly as it did.
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


ASRF_OVERRIDES = [
    "model.architecture=asrf",
    "model.params={backbone: asformer, backbone_params: {num_decoders: 0, num_layers: 4}}",
    "train.boundary.weight=1.0",
    "train.boundary.tolerance_s=0.04",  # 2 frames at the fixture's 50 Hz
    "train.device=cpu",
    "train.batch_size=1",
]

BAFORMER_OVERRIDES = [
    "model.architecture=baformer",
    "model.params={num_layers: 3, num_decode: 3, num_queries: 32, num_f_maps: 32, nheads: 4}",
    "train.frame_weight=0",
    "train.device=cpu",
    "train.batch_size=1",
]


def _run(project_dir: Path, overrides: list[str], name: str):
    return eto.segment.Project(project_dir / "config.yaml", *overrides, f"train.run_name={name}").train()


class TestASRF:
    def test_a_run_writes_the_boundary_diagnostic(self, project: Path) -> None:
        from ethograph.segment.train import BOUNDARY_FILE

        result = _run(project, ASRF_OVERRIDES, "asrf_smoke")
        assert (result.run_dir / BOUNDARY_FILE).is_file()
        assert result.test_metrics is not None
        assert "boundary_f1" in result.test_metrics["raw"]

    def test_the_boundary_settings_are_recorded_in_frames_and_seconds(self, project: Path) -> None:
        result = _run(project, ASRF_OVERRIDES, "asrf_record")
        recorded = yaml.safe_load((result.run_dir / "test_metrics.yaml").read_text(encoding="utf-8"))
        boundary = recorded["objective"]["boundary"]
        assert boundary["tolerance_s"] == pytest.approx(0.04)
        assert boundary["tolerance_frames"] == 2, "the fixture runs at 50 Hz, so 0.04 s is 2 frames"
        assert boundary["weight"] == 1.0

    def test_a_positive_weight_against_a_headless_architecture_is_refused(self, project: Path) -> None:
        with pytest.raises(ValueError, match="no boundary head"):
            eto.segment.Project(
                project / "config.yaml",
                "train.boundary.weight=1.0",
                "train.device=cpu",
                "train.run_name=headless",
            ).train()


class TestBaFormer:
    def test_a_run_trains_on_its_set_objective_alone(self, project: Path) -> None:
        result = _run(project, BAFORMER_OVERRIDES, "baformer_smoke")
        assert result.test_metrics is not None
        metrics = pd.read_csv(result.run_dir / "metrics.tsv", sep="\t")
        assert len(metrics) >= 1
        assert np.isfinite(metrics["loss"]).all()

    def test_too_few_queries_names_the_setting_to_raise(self, project: Path) -> None:
        overrides = [
            *BAFORMER_OVERRIDES[:1],
            "model.params={num_layers: 2, num_decode: 2, num_queries: 2, num_f_maps: 16, nheads: 4}",
            *BAFORMER_OVERRIDES[2:],
        ]
        with pytest.raises(ValueError, match="num_queries"):
            _run(project, overrides, "baformer_tiny")


class TestCircle:
    """The circle (deep metric-learning) term — architecture-agnostic, unlike boundary/query."""

    def test_a_run_trains_with_the_term_on_and_records_its_settings(self, project: Path) -> None:
        result = _run(
            project,
            ["train.circle.weight=0.5", "train.circle.max_frames=64", "train.device=cpu", "train.batch_size=1"],
            "circle_smoke",
        )
        assert result.test_metrics is not None
        recorded = yaml.safe_load((result.run_dir / "test_metrics.yaml").read_text(encoding="utf-8"))
        circle = recorded["objective"]["circle"]
        assert circle == {"weight": 0.5, "m": 0.25, "gamma": 128.0, "max_frames": 64}

    def test_default_weight_leaves_it_untrained(self, project: Path) -> None:
        result = _run(project, ["train.device=cpu", "train.batch_size=1"], "circle_off")
        recorded = yaml.safe_load((result.run_dir / "test_metrics.yaml").read_text(encoding="utf-8"))
        assert recorded["objective"]["circle"]["weight"] == 0.0


class TestRefinementModes:
    """The four ways a prediction can be turned into intervals, all reachable from one config."""

    @pytest.mark.parametrize(
        "mode,extra",
        [
            ("none", []),
            ("predicted", []),
            ("hybrid", ["infer.postprocess.changepoint_correction=true"]),
        ],
    )
    def test_every_mode_runs_and_is_recorded(self, project: Path, mode: str, extra: list[str]) -> None:
        result = _run(
            project,
            [*ASRF_OVERRIDES, f"infer.postprocess.boundary_refinement={mode}", *extra],
            f"asrf_{mode}",
        )
        config = load_config(result.run_dir / "config.yaml")
        assert config.infer.postprocess.boundary_refinement == mode
        assert result.test_metrics is not None

    def test_hybrid_without_changepoints_says_what_to_turn_on(self, project: Path) -> None:
        with pytest.raises(ValueError, match="changepoint_correction"):
            load_config(project / "config.yaml", ["infer.postprocess.boundary_refinement=hybrid"])

    def test_an_unknown_mode_is_refused_at_config_load(self, project: Path) -> None:
        with pytest.raises(ValueError, match="boundary_refinement"):
            load_config(project / "config.yaml", ["infer.postprocess.boundary_refinement=snap"])

    def test_a_headless_run_ignores_the_mode_rather_than_failing(self, project: Path) -> None:
        """A run trained before the head existed must post-process exactly as before."""
        from ethograph.segment.config import PostprocessConfig
        from ethograph.segment.postprocess import refine_dense

        indices = np.zeros(100, dtype=np.int64)
        indices[20:40] = 1
        cfg = PostprocessConfig(boundary_refinement="predicted")
        assert np.array_equal(refine_dense(indices, None, 50.0, cfg, np.arange(100) / 50.0), indices)


class TestEmptyBlocks:
    """``boundary:`` with nothing after it, and an explicit ``null`` that means something.

    A nested block written empty is "use the defaults" — but
    ``pos_weight: null`` is not the same statement: ``None`` is the *value*
    that says "recompute the positive weight per batch". A config reader that
    treats every null as "absent" would silently pin whatever the default
    happened to be.
    """

    def _written(self, project: Path, block: str) -> str:
        (project / "empty.yaml").write_text(f"base: config.yaml\ntrain:\n{block}", encoding="utf-8")
        return str(project / "empty.yaml")

    def test_an_empty_boundary_block_takes_the_defaults(self, project: Path) -> None:
        from ethograph.segment.config import BoundaryConfig, QueryLossConfig

        cfg = load_config(self._written(project, "  boundary:\n  # off, as it ships\n  queries:\n"))
        assert isinstance(cfg.train.boundary, BoundaryConfig)
        assert isinstance(cfg.train.queries, QueryLossConfig)
        assert cfg.train.boundary.weight == 0.0

    def test_an_explicit_null_pos_weight_survives(self, project: Path) -> None:
        cfg = load_config(self._written(project, "  boundary:\n    weight: 0.5\n    pos_weight: null\n"))
        assert cfg.train.boundary.pos_weight is None
        assert cfg.train.boundary.weight == 0.5

    def test_a_pinned_pos_weight_reaches_the_loss(self, project: Path) -> None:
        from ethograph.segment.losses import build_objective

        cfg = load_config(self._written(project, "  boundary:\n    weight: 0.5\n    pos_weight: 20\n"))
        objective, settings = build_objective(cfg, n_classes=2, fs=FS)
        assert objective.boundary_loss.pos_weight == 20.0
        assert settings["boundary"]["pos_weight"] == 20.0
