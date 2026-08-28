"""Train the trunks on data/spot/project.yaml one after the other, then compare them.

    python scripts/spot_sweep.py                 # msagsm, then gsm
    python scripts/spot_sweep.py gsm             # one of them
    python scripts/spot_sweep.py msagsm gsm      # any order

Everything else — features:, clip:, train: — comes from project.yaml
(docs/add_to_docs_later/spot/config.md). Each trunk is a `Project.update()` of
the one config, so the sweep is the config plus the two keys that differ: the
architecture, and a run name carrying it (the auto name carries only the clip
and `_features`, so two trunks would otherwise land in one folder).

The dataset is materialised once. A run whose last checkpoint exists is
skipped, so the sweep can be re-run after an interruption. Every trained run
is then scored into runs/compare.tsv — with features fed in, each is scored a
second time with them zeroed (test_metrics_nofeatures.yaml).
"""

from __future__ import annotations

import logging
import sys

import ethograph as eto
from ethograph.spot.inference import resolve_run_dir, run_reads_features, teacher_runs, trained_runs
from ethograph.spot.project import architectures
from ethograph.utils.logging import enable_console_logging

CONFIG = "data/spot/project.yaml"
TRUNKS = ("msagsm", "gsm")

log = logging.getLogger("sweep")


def main(argv: list[str]) -> None:
    enable_console_logging("sweep")
    for noisy in ("ethograph.io", "ethograph.gui", "ethograph.labels"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    trunks = argv or list(TRUNKS)
    known = architectures()
    for trunk in trunks:
        if f"rny008_{trunk}" not in known:
            raise SystemExit(
                f"rny008_{trunk} is not registered. The vendored clone needs "
                "scripts/spot_windows_compat.patch applied (git apply from the clone root)."
            )

    project = eto.spot.Project(CONFIG)
    cfg = project.config
    log.info("%s | frames -> %s", project, cfg.frames_dir)
    project.materialise()

    for trunk in trunks:
        name = f"{project.run_name()}_{trunk}"
        run = project.update(f"model.architecture=rny008_{trunk}", f"train.run_name={name}")
        last = run.config.run_dir(name) / f"checkpoint_{run.config.train.epochs - 1:03d}.pt"
        if last.is_file():
            log.info("%s: finished — skipped", name)
            continue
        log.info("== %s (rny008_%s)", name, trunk)
        run.train()

    for run_dir in teacher_runs(cfg) + trained_runs(cfg):
        project.evaluate(run=run_dir)
        if run_reads_features(resolve_run_dir(cfg, run_dir)):
            project.evaluate(run=run_dir, zero_features=True)
    table = project.compare()
    if len(table) > 1:
        log.info("compare (%s):\n%s", cfg.runs_dir / "compare.tsv", table.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
