"""Predict the wake behaviours over a sleep epoch, one-minute window by window.

1. Tile ``[START_S, END_S)`` of the recording into contiguous ``WINDOW_S``
   windows and write them as the trials of a ``sleep_alignment.nwb`` beside
   the session's own alignment.
2. Build the sleep session from the session ``decoding.yaml`` lists — the
   same ``units.npz``, that alignment instead of its own, no labels — so the
   two configs can never drift apart.
3. Take the decoder from ``RUN``: a run directory (``best.pt`` inside), or a
   cross-validation folder of fold runs, in which case every fold predicts
   the windows and each writes its own prediction set. ``RUN = None`` trains
   the ``DECODER`` below instead — an ``mstcn`` with its receptive field cut
   to the length of a wake trial, so a one-minute window shows every kernel
   only the kind of context it saw in training.
4. Predictions → ``labels/predictions_{run}_{timestamp}/`` beside ``units.npz``,
   the GUI's format; ``inference.yaml`` in that folder records the windows.
   Open them in the GUI by loading ``units.npz`` with the sleep alignment.

    python run_sleep.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import ethograph as eto
from ethograph.segment import as_overrides
from ethograph.segment.inference import resolve_run_dir
from ethograph.segment.train import BEST_FILE
from ethograph.segment.windows import STATE_COLUMN, write_windows_alignment

# Seconds on the recording's clock. This recording's wake trials end at
# 3150 s and its units run to 12196 s.
START_S = 3200.0
END_S = 12190.0
WINDOW_S = 60.0

DECODING = r"configs\neural\decoding.yaml"

#: A trained run directory, or a cross-validation folder whose folds all predict; ``None`` trains DECODER.
RUN: Path | None = Path(r"configs\neural\runs\cv_rate_5ms_boxcar25ms")

TRAIN_RUN_NAME = "mstcn_rate"
#: 2**10 frames ≈ 5 s at 200 Hz — about one wake trial, so no kernel ever
#: sees a context longer than the ones it trained on.
DECODER = [
    "model.architecture=mstcn",
    "model.params.num_layers_PG=10",
    "model.params.num_layers_R=10",
    f"train.run_name={TRAIN_RUN_NAME}",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def decoders() -> list[Path]:
    """The run directories to predict with: RUN itself, its folds, or a freshly trained run."""
    if RUN is None:
        project = eto.segment.Project(DECODING, *DECODER)
        try:
            return [resolve_run_dir(project.config, TRAIN_RUN_NAME)]
        except FileNotFoundError:
            return [project.train().run_dir]
    if (RUN / BEST_FILE).is_file():
        return [RUN]
    folds = sorted(p for p in RUN.iterdir() if p.is_dir() and (p / BEST_FILE).is_file())
    if not folds:
        raise FileNotFoundError(f"{RUN} is neither a trained run (no {BEST_FILE}) nor a folder of them")
    return folds


def sleep_project() -> eto.segment.Project:
    """``decoding.yaml`` with its one session re-listed under the sleep alignment, and no labels."""
    wake = eto.segment.Project(DECODING).config.sessions
    if len(wake) != 1:
        raise ValueError(f"{DECODING} lists {len(wake)} sessions; a neural decoder has one recording")
    source = wake[0].source
    alignment = source.parent.parent / ".ethograph" / "sleep_alignment.nwb"
    write_windows_alignment(alignment, START_S, END_S, WINDOW_S)
    sessions = [
        {
            "source": str(source),
            "alignment": str(alignment),
            "labels_path": str(source.with_name(f"{source.stem}_sleep_labels.tsv")),  # never exists: no labels
            "name": "sleep",
        }
    ]
    return eto.segment.Project(
        DECODING, *as_overrides({"sessions": sessions, f"trials.where.{STATE_COLUMN}": ["sleep"]})
    )


def main() -> None:
    runs = decoders()
    sleep = sleep_project()
    for run_dir in runs:
        for path in sleep.inference(run=run_dir):
            print(f"{run_dir.name}: wrote {path}")


if __name__ == "__main__":
    main()
