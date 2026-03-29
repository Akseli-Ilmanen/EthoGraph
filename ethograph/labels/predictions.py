"""Load model predictions from per-trial files and convert to label intervals.

Predictions are stored as per-trial files in a folder:
    predictions_dlc2action/
        dlc2action_trial1.pickle    # (T, n_classes) or (T,)
        dlc2action_trial2.npy

If shape (T, n_classes): softmax probabilities → labels via argmax, confidence via 1-entropy.
If shape (T,): dense labels directly, no confidence available.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.labels.intervals import dense_to_intervals, empty_intervals

logger = logging.getLogger(__name__)


def load_prediction_file(path: str | Path) -> np.ndarray:
    """Load a prediction file (.npy or .pickle). Returns a numpy array."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    elif path.suffix in (".pickle", ".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, dict):
            for key in ("predictions", "softmax", "probs", "probabilities"):
                if key in data:
                    return np.asarray(data[key])
            first_array = next(
                (v for v in data.values() if isinstance(v, np.ndarray)), None
            )
            if first_array is not None:
                return first_array
            raise ValueError(f"No numpy array found in pickle keys: {list(data.keys())}")
        raise ValueError(f"Unexpected pickle type: {type(data)}")
    else:
        raise ValueError(f"Unsupported prediction file format: {path.suffix}")


def prediction_to_labels_and_confidence(
    pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert prediction array to dense labels and optional confidence.

    Parameters
    ----------
    pred : np.ndarray
        Shape (T, n_classes) for softmax probabilities, or (T,) for dense labels.

    Returns
    -------
    labels : np.ndarray, shape (T,)
        Dense integer labels (argmax for softmax input).
    confidence : np.ndarray or None
        Shape (T,) confidence scores. For softmax input: 1 - normalized_entropy.
        None if input is already dense labels.
    """
    pred = np.asarray(pred)

    if pred.ndim == 2:
        n_classes = pred.shape[1]
        labels = np.argmax(pred, axis=1)
        # 1 - normalized entropy: 1.0 = certain, 0.0 = uniform
        eps = 1e-10
        entropy = -np.sum(pred * np.log(pred + eps), axis=1)
        max_entropy = np.log(n_classes)
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else np.ones(len(pred))
        return labels, confidence.astype(np.float32)

    # Shape (T,) — already dense labels
    return pred.astype(int), None


def load_predictions_folder(
    folder: str | Path,
    dt,
    individual: str,
) -> tuple[pd.DataFrame, dict[int | str, np.ndarray | None]]:
    """Load all prediction files from a folder, convert to intervals + confidence.

    Parameters
    ----------
    folder : Path
        Folder containing per-trial prediction files.
    dt : TrialTree
        The data tree (needed for time coordinates to build intervals).
    individual : str
        Individual identifier to assign to predicted labels.

    Returns
    -------
    all_labels_df : pd.DataFrame
        Labels in the standard TSV format with prediction_source column.
    confidence_map : dict
        {trial: confidence_array_or_None} for GUI overlay.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Predictions folder not found: {folder}")

    pred_files = sorted(
        p for p in folder.iterdir()
        if p.suffix in (".npy", ".pickle", ".pkl")
    )

    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {folder}")

    rows = []
    confidence_map = {}

    for pred_file in pred_files:
        # Match file to trial — try extracting trial ID from filename
        trial = _extract_trial_from_filename(pred_file, dt.trials)
        if trial is None:
            logger.warning("Could not match %s to a trial, skipping", pred_file.name)
            continue

        pred = load_prediction_file(pred_file)
        labels, confidence = prediction_to_labels_and_confidence(pred)
        confidence_map[trial] = confidence

        ds = dt.trial(trial)
        time_coord = ds.time.values if "time" in ds.coords else np.arange(len(labels)) / 30.0
        time_coord = time_coord[:len(labels)]

        intervals = dense_to_intervals(labels, time_coord, [individual])
        if not intervals.empty:
            intervals.insert(0, "trial", trial)
            intervals["prediction_source"] = str(pred_file)
            intervals["human_verified"] = 0
            intervals["changepoint_corrected"] = 0
            rows.append(intervals)

    if rows:
        all_df = pd.concat(rows, ignore_index=True)
    else:
        all_df = empty_intervals()
        all_df.insert(0, "trial", pd.Series(dtype=object))

    return all_df, confidence_map


def _extract_trial_from_filename(path: Path, trial_list: list) -> int | str | None:
    """Try to extract a trial ID from a prediction filename."""
    stem = path.stem

    # Try direct match: "trial_1", "trial1", etc.
    for trial in trial_list:
        trial_str = str(trial)
        if trial_str in stem:
            return trial

    # Try extracting trailing number
    match = re.search(r"(\d+)$", stem)
    if match:
        num = int(match.group(1))
        if num in trial_list:
            return num
        num_str = str(num)
        if num_str in [str(t) for t in trial_list]:
            return num

    return None
