"""Run the spot pipeline on data/spot/project.yaml, stage by stage.

    python run.py                      # everything, in order, skipping finished stages
    python run.py teacher              # one stage: materialise | teacher | baseline | distil | evaluate | inference
    python run.py evaluate             # the test summary of the newest run (test_metrics.yaml + a table)
    python run.py crossval             # leave-one-session-out: train on the others, predict + score the held-out one
    python run.py inference --run ctx2s_res10ms_distil_64d5ef46 --sessions 20260308_01   # one model, one new session
    python run.py distil inference     # several
    python run.py --limit 20 baseline  # a quick pass on the first 20 trials per session

A quick pass needs the frame budget cut as well as the trials — an epoch costs
epoch_frames, not trials (~6.5 min per 250k at 3.2 it/s):

    python run.py --limit 20 --set train.epochs=3 train.epoch_frames=50000 \
        distil.epochs=2 distil.head_epochs=2 distil.epoch_frames=50000

Stages go first on the command line, options after (both take several words).

Every stage is a method on eto.spot.Project; this file only orders them and
skips what is already done, so it can be re-run after a crash. Outputs land
under data/spot/ (frames_crop.../, dataset/, features/, teacher/, runs/) and
predictions beside each session under labels/predictions_spot_*/.
"""

from __future__ import annotations

import argparse
import logging
import sys

import ethograph as eto
from ethograph.utils.logging import enable_console_logging

CONFIG = "data/spot/project.yaml"
STAGES = ("materialise", "teacher", "baseline", "distil", "evaluate", "inference", "crossval")


def finished_run(run_dir, epochs: int) -> bool:
    return (run_dir / f"checkpoint_{epochs - 1:03d}.pt").is_file()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="*", choices=STAGES, help="which stages; default all")
    parser.add_argument("--limit", type=int, default=None, help="first N trials per session (a smoke run)")
    parser.add_argument("--sessions", nargs="*", default=None, help="sessions to predict into (default: all)")
    parser.add_argument(
        "--run", default=None, help="run to evaluate/predict with, a name under runs/ (default: newest)"
    )
    parser.add_argument("--force", action="store_true", help="rerun stages that look finished")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="trials decoded at once (default: cores minus two, capped at 16)",
    )
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    args = parser.parse_args(argv)
    stages = args.stages or list(STAGES)
    enable_console_logging("run")  # the pipelines already print their own records on import
    for noisy in ("ethograph.io", "ethograph.gui", "ethograph.labels"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    overrides = ([f"trials.limit={args.limit}"] if args.limit else []) + list(args.set)
    project = eto.spot.Project(CONFIG, *overrides)
    cfg = project.config
    log = logging.getLogger("run")
    log.info("%s | frames -> %s", project, cfg.frames_dir)

    if "materialise" in stages:
        if (cfg.dataset_dir / "train.json").is_file() and not args.force:
            log.info("materialise: dataset/ exists, features/ refreshed only — pass --force to redo")
        project.materialise(workers=args.workers)

    if "teacher" in stages:
        project.train_teacher()

    if "baseline" in stages:
        name = project.run_name()
        if finished_run(cfg.run_dir(name), cfg.train.epochs) and not args.force:
            log.info("baseline: %s finished — skipped (--force to redo)", name)
        else:
            project.train()

    student = None
    if "distil" in stages:
        student = project.distil()

    run = student.run_dir if student is not None else args.run

    if "evaluate" in stages:
        # The student this call trained, else every trained run: the baseline and each distilled student.
        from ethograph.spot.inference import resolve_run_dir, run_reads_features, teacher_runs, trained_runs

        for run_dir in [run] if run is not None else teacher_runs(cfg) + trained_runs(cfg):
            project.evaluate(run=run_dir)
            if run_reads_features(resolve_run_dir(cfg, run_dir)):
                # what the features contribute: the same model, features zeroed
                project.evaluate(run=run_dir, zero_features=True)
        table = project.compare()
        if len(table) > 1:
            log.info("compare (%s):\n%s", cfg.runs_dir / "compare.tsv", table.to_string(index=False))

    if "crossval" in stages:
        for fold in project.cross_validate(sessions=args.sessions, workers=args.workers):
            log.info("fold: %s -> %s", fold.name, fold.run_dir / "test_metrics.yaml")

    if "inference" in stages:
        for tsv in project.inference(run=run, sessions=args.sessions, workers=args.workers):
            log.info("predictions: %s", tsv)


if __name__ == "__main__":
    main(sys.argv[1:])
