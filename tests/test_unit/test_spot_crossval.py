"""A fold is a project of its own: the held-out session is its whole test split, and only it is predicted.

What earns a test: the fold must reuse the frames (not decode them again),
must evaluate before predicting, and must predict *only* the session it held
out — a fold that predicts the sessions it trained on is the leak the stage
exists to avoid.
"""

from __future__ import annotations

from pathlib import Path

from ethograph.spot.config import config_from_dict
from ethograph.spot.project import Project, RunResult


def _config(tmp_path):
    sources = []
    for name in ("s1", "s2", "s3"):
        source = tmp_path / f"{name}.nc"
        source.touch()
        sources.append(str(source))
    data = {
        "sessions": sources,
        "labels": {"classes": [31], "crop": {"x0": 0, "y0": 0, "x1": 10, "y1": 10}},
        "frames": str(tmp_path / "frames"),
        "root": str(tmp_path),
    }
    return config_from_dict(data, tmp_path)


def test_each_fold_holds_out_one_session_and_predicts_only_it(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    calls: list[tuple[str, Path, object]] = []

    def record(stage):
        def method(self, *args, **kwargs):
            calls.append((stage, self.config.root, kwargs.get("sessions")))
            if stage == "train":
                return RunResult(name=self.config.train.run_name, run_dir=self.config.run_dir("r"), clip=None)
            return None

        return method

    for stage in ("materialise", "train", "evaluate", "inference"):
        monkeypatch.setattr(Project, stage, record(stage))

    folds = Project(cfg).cross_validate()

    assert [f.name for f in folds] == ["fold_s1", "fold_s2", "fold_s3"]
    assert [c[0] for c in calls] == ["materialise", "train", "evaluate", "inference"] * 3
    roots = {c[1] for c in calls}
    assert roots == {cfg.cross_validation_dir / s for s in ("s1", "s2", "s3")}
    predicted = [c[2] for c in calls if c[0] == "inference"]
    assert predicted == [[Path(s)] for s in (spec.source for spec in cfg.sessions)]


def test_a_fold_keeps_the_frames_and_names_its_holdout(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    seen: list = []

    def materialise(self, **kwargs):
        seen.append(self.config)

    monkeypatch.setattr(Project, "materialise", materialise)
    monkeypatch.setattr(Project, "train", lambda self: (_ for _ in ()).throw(StopIteration))
    try:
        Project(cfg).cross_validate(sessions=["s2"])
    except StopIteration:
        pass
    (fold,) = seen
    assert fold.frames_dir == cfg.frames_dir  # every fold reads the project's one frames folder
    assert [Path(p) for p in fold.train.split.holdout_sessions] == [Path(cfg.sessions[1].source)]
    assert fold.train.run_name == "fold_s2"
