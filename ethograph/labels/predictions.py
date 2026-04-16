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

from ethograph.labels.intervals import empty_intervals
from ethograph.labels.ml import dense_to_intervals

logger = logging.getLogger(__name__)


def load_prediction_file(path: str | Path) -> np.ndarray:
    """Load a prediction file (.npy or .pickle). Returns a numpy array.

    For .npy files with shape (T,) (confidence or dense labels), uses
    memory-mapping (mmap_mode='r') so no data is copied into RAM until accessed.
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode='r')
        return arr
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


class PredictionsStore:
    """Lazy per-trial loader for a predictions folder.

    Scans the folder at construction time (fast — filesystem only, no file reads).
    Individual trial data is loaded on demand via :meth:`get_confidence`.

    Supports ``.npy`` (memory-mapped when shape is 1-D) and ``.pkl``/``.pickle``
    formats. Additional formats can be added to ``load_prediction_file``.

    Parameters
    ----------
    folder : str or Path
        Folder containing per-trial prediction files.

    Example
    -------
    ::

        store = PredictionsStore("predictions_cetnet_20260330/uncorr")
        confidence = store.get_confidence(trial=5, dt=dt)
        labels_df, levels = store.load_all(dt, individual="Poppy", threshold=0.75)
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Predictions folder not found: {self.folder}")
        self._files: list[Path] = sorted(
            p for p in self.folder.iterdir()
            if p.suffix in (".npy", ".pickle", ".pkl")
        )
        self._index: dict = {}  # {trial: Path} — populated lazily on first access

    def _resolve(self, trial_list: list) -> None:
        """Build the trial→file index if not already done."""
        if self._index:
            return
        for p in self._files:
            trial = _extract_trial_from_filename(p, trial_list)
            if trial is not None:
                self._index[trial] = p

    def get_file(self, trial, trial_list: list) -> Path | None:
        """Return the prediction file path for a trial, or None."""
        self._resolve(trial_list)
        return self._index.get(trial)

    def get_confidence(self, trial, dt) -> np.ndarray | None:
        """Load and return the confidence array for one trial.

        For ``.npy`` probability files the array is memory-mapped; for ``.pkl``
        files the full file is read (typically ~150 KB — a few milliseconds).
        The returned array is not cached — call again to re-load if needed.
        """
        path = self.get_file(trial, dt.trials)
        if path is None:
            return None
        pred = load_prediction_file(path)
        _, confidence = prediction_to_labels_and_confidence(pred)
        return confidence

    def load_all(
        self,
        dt,
        individual: str,
        confidence_threshold: float = 0.75,
        segment_confidence_threshold: float = 0.6,
    ) -> tuple[pd.DataFrame, dict[int | str, str]]:
        """Load all trials — convert to intervals and compute confidence levels.

        Confidence arrays are computed in one pass then discarded; only the
        per-trial high/low classification is kept.  The same two-condition
        criterion used in the confidence PDF is applied: a trial is "low" if
        its overall mean confidence < *confidence_threshold* OR any labeled
        segment's mean confidence < *segment_confidence_threshold*.

        Parameters
        ----------
        dt : TrialTree
        individual : str
        confidence_threshold : float
            Frame-level threshold; overall trial mean below this → "low".
        segment_confidence_threshold : float
            Segment-level threshold; any segment mean below this → "low".

        Returns
        -------
        all_labels_df : pd.DataFrame
        confidence_levels : dict
            ``{trial: "low" | "high"}``
        """
        if not self._files:
            raise FileNotFoundError(f"No prediction files found in {self.folder}")

        rows: list[pd.DataFrame] = []
        confidence_levels: dict = {}

        for pred_file in self._files:
            trial = _extract_trial_from_filename(pred_file, dt.trials)
            if trial is None:
                logger.warning("Could not match %s to a trial, skipping", pred_file.name)
                continue

            pred = load_prediction_file(pred_file)
            labels, confidence = prediction_to_labels_and_confidence(pred)

            ds = dt.trial(trial)
            time_coord = ds.time.values if "time" in ds.coords else np.arange(len(labels)) / 30.0
            assert len(labels) == len(time_coord), (
                f"Trial {trial}: predictions length {len(labels)} != time_coord length {len(time_coord)}"
            )

            intervals = dense_to_intervals(labels, [individual], time_coord=time_coord)
            if not intervals.empty:
                intervals.insert(0, "trial", trial)
                intervals["prediction_source"] = str(pred_file)
                intervals["human_verified"] = 0
                intervals["changepoint_corrected"] = 0
                rows.append(intervals)

            if confidence is not None:
                mean_conf = float(np.mean(confidence))
                has_low_segment = False
                if not intervals.empty and "onset_s" in intervals.columns:
                    for _, seg in intervals.iterrows():
                        seg_mask = (time_coord >= seg["onset_s"]) & (time_coord <= seg["offset_s"])
                        if seg_mask.any() and float(np.mean(confidence[seg_mask])) < segment_confidence_threshold:
                            has_low_segment = True
                            break
                low = mean_conf < confidence_threshold or has_low_segment
                confidence_levels[trial] = "low" if low else "high"

        if rows:
            all_df = pd.concat(rows, ignore_index=True)
        else:
            all_df = empty_intervals()
            all_df.insert(0, "trial", pd.Series(dtype=object))

        return all_df, confidence_levels


def _extract_trial_from_filename(path: Path, trial_list: list) -> int | str | None:
    """Try to extract a trial ID from a prediction filename."""
    stem = path.stem

    # Extract the number after 'trial' (e.g. cetnet_trial10_uncorr -> 10)
    match = re.search(r'trial(\d+)', stem)
    if match:
        num = int(match.group(1))
        if num in trial_list:
            return num
        num_str = str(num)
        if num_str in [str(t) for t in trial_list]:
            return num

    # Fallback: trailing number in stem
    match = re.search(r'(\d+)$', stem)
    if match:
        num = int(match.group(1))
        if num in trial_list:
            return num
        num_str = str(num)
        if num_str in [str(t) for t in trial_list]:
            return num

    return None
