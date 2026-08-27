"""The probability curves an onset-prediction run produced.

A predicted label carries **one** confidence number — the height of its
curve's tallest peak. One number cannot say whether a low score means the
model was torn between two moments or found nothing anywhere, and it cannot
show a rival peak elsewhere in the trial. The curve can, so a run keeps it:
frame-by-frame review draws it under the label it is on.

Written by :func:`~ethograph.gui.dialog_onset_model.predict_onsets`, read by
the Curation section. Numpy only, on purpose — the GUI reads these without
importing the model stack.

**One run, one folder**, beside the session in the same ``labels/`` directory
the label backups use::

    {session}.nc
    labels/
        predictions_lightgbm_20260824_151107/
            onset_curves.npz
        predictions_lightgbm_20260824_162244/
            onset_curves.npz

A run is a record of what a model said at a moment, so its folder is written
once and never edited. :func:`read_all_curves` reads every run, the newest
winning per (trial, class) — so re-predicting one class does not erase what an
earlier run said about another, and the older run is still on disk to compare.
The Curation section draws from a single chosen run instead of this merge: one
matching run is used without asking, and with several the reviewer picks —
by name, or by browsing straight to an ``onset_curves.npz`` — so it is always
clear whose confidence is on screen.

Layout of one ``onset_curves.npz`` — one entry per trial the run predicted
into::

    trials              (N,) str      trial ids, position i keys the rest
    time__{i}           (T,) float64  trial-relative, the features' own clock
    curve__{i}__{label} (T,) float32  that class's smoothed event probability
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Folder each prediction run writes into, under the session's ``labels/``:
#: ``predictions_{model}_{timestamp}`` — the convention every model here
#: shares (the LightGBM onset model, the segmentation pipeline, the pixel
#: spotter). A run is recognised by holding :data:`CURVES_FILE`, not by which
#: model wrote it.
RUN_PREFIX = "predictions_"

#: The model name the GUI's onset model writes under.
LIGHTGBM = "lightgbm"

#: The curves file inside a run folder.
CURVES_FILE = "onset_curves.npz"

#: One trial's curves: ``(time, {label: curve})``.
TrialCurves = tuple[np.ndarray, dict[int, np.ndarray]]


def labels_dir(session_path: str | Path) -> Path:
    """The ``labels/`` folder beside a session file — where backups live too."""
    return Path(session_path).parent / "labels"


def run_dir(session_path: str | Path, timestamp: str, model: str = LIGHTGBM) -> Path:
    """The folder one prediction run writes into: ``predictions_{model}_{timestamp}``."""
    return labels_dir(session_path) / f"{RUN_PREFIX}{model}_{timestamp}"


def run_timestamp(folder: Path) -> str:
    """The ``YYYYMMDD_HHMMSS`` a run folder ends in — what orders runs.

    Sorting by whole name would put every ``lightgbm`` run before every
    ``spot`` run whatever their dates; the timestamp is the last two parts.
    """
    return "_".join(folder.name.rsplit("_", 2)[-2:])


def run_dirs(session_path: str | Path) -> list[Path]:
    """Every prediction run's folder that holds curves, oldest first."""
    root = labels_dir(session_path)
    if not root.is_dir():
        return []
    return sorted((p for p in root.glob(f"{RUN_PREFIX}*") if (p / CURVES_FILE).is_file()), key=run_timestamp)


def read_curves(path: str | Path) -> dict[str, TrialCurves]:
    """One run's curves, keyed by trial id as a string.

    A missing or unreadable file reads as ``{}`` — curves are an aid to
    review, never something a session depends on.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, TrialCurves] = {}
    try:
        with np.load(path, allow_pickle=False) as npz:
            for i, trial in enumerate(str(t) for t in npz["trials"]):
                prefix = f"curve__{i}__"
                out[trial] = (
                    np.asarray(npz[f"time__{i}"], dtype=np.float64),
                    {
                        int(key[len(prefix) :]): np.asarray(npz[key], dtype=np.float64)
                        for key in npz.files
                        if key.startswith(prefix)
                    },
                )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Ignoring unreadable onset curves at %s: %s", path, exc)
        return {}
    return out


def read_all_curves(session_path: str | Path) -> dict[str, TrialCurves]:
    """Every run's curves for a session, the newest run winning per class.

    Runs are filtered by the trials table and by which classes a trial still
    lacks, so no single run holds everything. Reading oldest to newest and
    letting later runs overwrite gives the latest word on each (trial, class)
    while keeping what only an earlier run predicted.
    """
    merged: dict[str, TrialCurves] = {}
    for folder in run_dirs(session_path):
        for trial, (time, curves) in read_curves(folder / CURVES_FILE).items():
            if trial not in merged:
                merged[trial] = (time, dict(curves))
                continue
            # Same trial, later run: its time base is authoritative for the
            # classes it re-predicted, and identical for the ones it did not.
            previous = merged[trial][1]
            previous.update(curves)
            merged[trial] = (time, previous)
    return merged


def write_curves(path: str | Path, per_trial: dict[object, TrialCurves]) -> Path:
    """Write one run's curves to *path*, creating its folder."""
    path = Path(path)
    trials = sorted(per_trial, key=str)
    arrays: dict[str, np.ndarray] = {"trials": np.array([str(t) for t in trials], dtype="U")}
    for i, trial in enumerate(trials):
        time, curves = per_trial[trial]
        arrays[f"time__{i}"] = np.asarray(time, dtype=np.float64)
        for label, curve in curves.items():
            arrays[f"curve__{i}__{int(label)}"] = np.asarray(curve, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path
