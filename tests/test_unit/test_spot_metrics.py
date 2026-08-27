"""The test summary: what counts as a miss, a spurious prediction and an error, in ms.

What earns a test: the error must come back on the truth's clock at the
bin's centre (a strided run is not read as early), a miss must count against
the hit rate rather than vanish from it, and a class the split never
labels must not pretend to a score.
"""

from __future__ import annotations

import gzip
import json

import pytest

from ethograph.spot.config import config_from_dict
from ethograph.spot.inference import run_clip
from ethograph.spot.metrics import TOLERANCES_MS, evaluate_run, format_table, score_predictions


def _config(tmp_path):
    source = tmp_path / "ses-01.nc"
    source.touch()
    return config_from_dict(
        {"sessions": [str(source)], "labels": {"classes": [31, 32]}, "root": str(tmp_path)}, tmp_path
    )


def _truth(videos):
    return [
        {"video": v, "fps": 200.0, "num_frames": 1000, "events": [{"label": f"label_{c}", "frame": f} for c, f in ev]}
        for v, ev in videos.items()
    ]


def _entry(video, events, fps=100.0):
    return {
        "video": video,
        "fps": fps,
        "num_frames": 500,
        "events": [{"label": f"label_{c}", "frame": f, "score": s} for c, f, s in events],
    }


def _run(tmp_path, stride=2):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"stride": stride, "clip_len": 100, "dilate_len": 1}))
    (run_dir / "checkpoint_000.pt").write_bytes(b"x")
    return run_dir


class TestScore:
    def test_error_is_on_the_truth_clock_at_the_bin_centre(self, tmp_path):
        config = _config(tmp_path)
        clip = run_clip(_run(tmp_path, stride=2), 200.0)
        # bin 200 at stride 2 covers frames 400-401; its centre is 400.5, the truth is at 402: 1.5 frames = 7.5 ms
        scores = score_predictions([_entry("v", [(31, 200, 0.9)])], _truth({"v": [(31, 402)]}), config, clip)
        assert scores[31].errors_ms == pytest.approx([7.5])
        assert scores[31].hit_rate(10) == 1.0 and scores[31].hit_rate(5) == 0.0

    def test_a_miss_counts_against_the_hit_rate_and_a_stray_is_spurious(self, tmp_path):
        config = _config(tmp_path)
        clip = run_clip(_run(tmp_path), 200.0)
        truth = _truth({"a": [(31, 400)], "b": [(31, 600)]})
        entries = [_entry("a", [(31, 200, 0.9), (32, 100, 0.5)])]  # b unpredicted; 32 never happened in a
        scores = score_predictions(entries, truth, config, clip)
        assert (scores[31].n_truth, scores[31].n_missing, scores[31].n_predicted) == (2, 1, 1)
        assert scores[31].hit_rate(100) == 0.5
        assert scores[32].n_spurious == 1 and scores[32].n_truth == 0

    def test_an_unlabelled_class_has_no_rate(self, tmp_path):
        config = _config(tmp_path)
        scores = score_predictions([], _truth({"a": [(31, 400)]}), config, run_clip(_run(tmp_path), 200.0))
        assert scores[32].to_dict()["hit_rate"] == {f"{t}ms": None for t in TOLERANCES_MS}
        assert scores[31].to_dict()["hit_rate"]["100ms"] == 0.0


class TestEvaluateRun:
    def test_writes_the_yaml_from_the_epochs_own_predictions(self, tmp_path):
        config = _config(tmp_path)
        run_dir = _run(tmp_path)
        config.dataset_dir.mkdir(parents=True)
        (config.dataset_dir / "test.json").write_text(json.dumps(_truth({"v": [(31, 400), (32, 800)]})))
        with gzip.open(run_dir / "pred-test.0.recall.json.gz", "wt") as fh:
            json.dump([_entry("v", [(31, 200, 0.9), (32, 380, 0.9)])], fh)
        metrics = evaluate_run(config, run_dir, split="test", epoch=0)
        assert (run_dir / "test_metrics.yaml").is_file()
        assert metrics["classes"]["label_31"]["hit_rate"]["10ms"] == 1.0
        assert metrics["classes"]["label_32"]["mean_error_ms"] == pytest.approx(
            197.5
        )  # bin 380 -> 760.5 vs 800: 39.5 frames
        table = format_table(metrics)
        assert "label_31" in table and "label_32" in table


class TestCompare:
    def test_one_row_per_scored_run_flattened_by_class(self, tmp_path):
        import yaml

        from ethograph.spot.metrics import compare_runs

        config = _config(tmp_path)
        scored = _run(tmp_path)
        (tmp_path / "runs" / "unscored").mkdir()
        (tmp_path / "runs" / "unscored" / "config.json").write_text("{}")
        (scored / "test_metrics.yaml").write_text(
            yaml.safe_dump(
                {
                    "run": "r",
                    "epoch": 0,
                    "split": "test",
                    "classes": {
                        "label_31": {
                            "n_missing": 1,
                            "n_spurious": 0,
                            "median_error_ms": 2.5,
                            "mean_error_ms": 4.0,
                            "hit_rate": {"10ms": 0.8, "20ms": 0.9, "50ms": 1.0, "100ms": 1.0},
                        }
                    },
                }
            )
        )
        df = compare_runs(config)
        assert list(df["run"]) == ["r"]
        assert df.loc[0, "label_31.hit20ms"] == 0.9 and df.loc[0, "label_31.miss"] == 1
        assert (config.runs_dir / "compare.tsv").is_file()
