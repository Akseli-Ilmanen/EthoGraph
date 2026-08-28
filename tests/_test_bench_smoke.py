"""Ad-hoc smoke of scripts/bench.py on the synthetic two-session fixture (CPU, 1 epoch).

Exercises what the real bench cannot afford to get wrong: a cell trains its
folds, a second call reads them back without training, a fold whose test
evaluation is missing is the only one retrained, and the report draws.

    python tests/_test_bench_smoke.py <scratch_dir>
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(scratch: Path) -> None:
    fixture = _load("fixture", REPO / "tests" / "test_unit" / "test_segment_pipeline.py")
    config_dir = scratch / "bench"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "mapping.txt").write_text("0 background\n3 flap\n4 peck 1\n5 call 0 point\n", encoding="utf-8")
    s1 = fixture._make_session(scratch / "sessions" / "s1", "s1", [1, 2, 3, 4], seed=0)
    s2 = fixture._make_session(scratch / "sessions" / "s2", "s2", [1, 2], seed=10)
    sessions = [
        {"source": str(s1), "labels_path": str(s1.with_name("s1_labels.tsv"))},
        {"source": str(s2), "labels_path": str(s2.with_name("s2_labels.tsv"))},
    ]
    shared = {
        "sessions": sessions,
        "features": {
            "name": "all",
            "columns": {
                "position": {"space": ["x", "y"], "keypoint": ["beak", "tail"]},
                "speed": {"keypoint": ["beak"]},
            },
            "labels": {"mapping": "mapping.txt", "branch": 0},
        },
        "model": {"architecture": "mlp", "params": {"f_maps_list": [16]}},
        "train": {"epochs": 1, "eval_every": 1, "device": "cpu"},
    }
    (config_dir / "project.yaml").write_text(yaml.safe_dump(shared, sort_keys=False), encoding="utf-8")
    crow = {"base": "project.yaml", "features": {"name": "crow1"}, "sessions": sessions}
    (config_dir / "crow1.yaml").write_text(yaml.safe_dump(crow, sort_keys=False), encoding="utf-8")

    os.environ["BENCH_CONFIG_DIR"] = str(config_dir)
    bench = _load("bench", REPO / "scripts" / "bench.py")
    bench.INDIVIDUALS[:] = ["crow1"]
    bench.ARCHITECTURES[:] = ["mlp"]
    # Two arms are enough for the resume logic, and the fixture's columns
    # declare no kind, so an arm dropping one would refuse to train.
    for arm in list(bench.ARMS):
        if arm not in ("all", "no_circle"):
            del bench.ARMS[arm]

    def fold_dirs(cell_project) -> list[Path]:
        cv = cell_project.config.runs_dir / bench.cross_validation_name_for(cell_project.config)
        return sorted(p for p in cv.glob("fold-*") if p.is_dir())

    # 1. a cell trains every fold
    done = bench.cross_validate_cell("crow1", "mlp", "all")
    project = bench.project_for("crow1", "mlp", "all")
    assert set(done) == {str(s.source) for s in project.config.sessions}, done
    assert len(fold_dirs(project)) == 2, fold_dirs(project)
    assert project.config.data_dir.name == "crow1" and (project.config.data_dir / "index.tsv").is_file()
    print("[1] both folds trained under", fold_dirs(project)[0].parent)

    # 2. a second call reads them back — no new run directory
    bench.cross_validate_cell("crow1", "mlp", "all")
    assert len(fold_dirs(project)) == 2, "a finished cell was retrained"
    print("[2] finished cell read back, nothing retrained")

    # 3. a fold without its test evaluation is the only one retrained
    victim = done[str(project.config.sessions[1].source)]
    (victim / bench.EVAL_ARRAYS_FILE).unlink()
    before = fold_dirs(project)
    done = bench.cross_validate_cell("crow1", "mlp", "all")
    after = fold_dirs(project)
    new = [p for p in after if p not in before]
    assert len(new) == 1 and new[0].name.startswith(f"fold-{bench.session_id(project.config.sessions[1].source)}_"), new
    assert done[str(project.config.sessions[1].source)] == new[0]
    print("[3] only the broken fold retrained:", new[0].name)

    # 4. a fold that evaluated but never predicted gets its prediction set by inference alone
    import shutil

    source = str(project.config.sessions[0].source)
    run_dir = done[source]
    prediction_dirs = lambda: list(  # noqa: E731
        bench.prediction_run_dir(Path(source), run_dir.name, "").parent.glob(f"predictions_{run_dir.name}_*")
    )
    for d in prediction_dirs():
        shutil.rmtree(d)
    assert not bench.has_predictions(source, run_dir)
    before = fold_dirs(project)
    bench.cross_validate_cell("crow1", "mlp", "all")
    assert fold_dirs(project) == before, "re-predicting must not retrain"
    assert bench.has_predictions(source, run_dir) and len(prediction_dirs()) == 1
    print("[4] missing prediction set written by inference alone:", prediction_dirs()[0].name)

    # 5. the second arm, then the report over both
    bench.cross_validate_cell("crow1", "mlp", "no_circle")
    sys.argv = ["bench.py", "--report-only"]
    bench.main()
    assert bench.OUTPUT.is_file() and bench.OUTPUT.stat().st_size > 0
    assert bench.TABLE.is_file()
    import pandas as pd

    table = pd.read_csv(bench.TABLE, sep="\t")
    assert set(table["loss"]) == {"all", "no_circle"} and set(table["individual"]) == {"crow 1"}, table
    assert len(table) == 4, table
    print("[5] report written:", bench.OUTPUT, "rows:", len(table))
    print("ALL OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
