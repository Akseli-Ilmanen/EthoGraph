"""Figure 1: predictions with and without changepoint snapping, against ground truth.

One cross-validation fold — three Freddy sessions train, ``TEST_SESSION`` is
held out — predicted twice from that single trained run. The two prediction
sets differ only in ``infer.postprocess.changepoint_correction``: purge,
stitch and the per-label thresholds are on both times, so the figure shows
what snapping to changepoints does and nothing else.

Rows: curated ground truth, predictions without snapping, predictions with
it, and the model's per-frame confidence. Written to ``{root}/plots``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ethograph.segment as seg
from ethograph.labels.intervals import LABELING_AUTOMATED, load_label_mapping
from ethograph.labels.plots import plot_label_segments
from ethograph.labels.predictions import PredictionsStore, prediction_to_labels_and_confidence
from ethograph.labels.tsv_store import load_labels_tsv
from ethograph.segment.samples import ClassTable, class_table

CONFIG_PATH = Path(r"C:\Users\aksel\Documents\Code\ethograph\configs\paper\data\fig1_c2f_tcn_crow1.yaml")
TEST_SESSION = (
    r"C:\Users\aksel\Documents\AK_data\derivatives\sub-03_id-Freddy"
    r"\ses-000_date-20250526_01\behav\Trial_data3.nc"
)
TRIAL = 33
VAL_FRACTION = 0.0  # default hyperparameters: best.pt is the last epoch
ROW_LABELS = ("ground truth", "predicted", "changepoint-corrected", "confidence")


def confidence_curve(npz_path: Path, trial: int | str) -> tuple[np.ndarray, np.ndarray]:
    """One trial's time axis and per-frame confidence, from a run's ``_probs.npz``."""
    marker = f"_trial{trial}_"
    with np.load(npz_path) as npz:
        key = next(k for k in npz.files if marker in k and not k.endswith(("_time", "_boundary")))
        probs = np.asarray(npz[key], dtype=np.float64)
        time = np.asarray(npz[f"{key}_time"], dtype=np.float64)
    _, confidence = prediction_to_labels_and_confidence(probs)
    if confidence is None:
        raise ValueError(f"{npz_path}: {key!r} is not (T, C) probabilities")
    return time, confidence


def trial_labels(df: pd.DataFrame, trial: int | str) -> pd.DataFrame:
    """*trial*'s rows, restricted to the classes this model predicts."""
    return df[(df["trial"] == trial) & (df["event_type"] == "state")]


def main() -> Path:
    project = seg.Project(CONFIG_PATH)
    project.materialise()
    folds = project.cross_validate(folds=[TEST_SESSION], val_fraction=VAL_FRACTION)

    fold = folds.iloc[0]
    run_dir = Path(fold["run_dir"])
    with_cp = PredictionsStore(Path(fold["predictions"]).parent)

    # The same trained run and the same postprocessing, snapping off.
    no_cp = seg.Project(CONFIG_PATH, "infer.postprocess.changepoint_correction=false")
    without_cp = PredictionsStore(no_cp.inference(run=run_dir, sessions=[TEST_SESSION])[0].parent)

    config = project.config
    mapping = load_label_mapping(config.features.labels.mapping)

    ground_truth = load_labels_tsv(config.select_sessions([TEST_SESSION])[0].labels_path)
    rows = [
        trial_labels(ground_truth[ground_truth["labeling_method"] != LABELING_AUTOMATED], TRIAL),
        trial_labels(load_labels_tsv(without_cp.tsv_path), TRIAL),
        trial_labels(load_labels_tsv(with_cp.tsv_path), TRIAL),
    ]
    if with_cp.npz_path is None:
        raise FileNotFoundError(f"{with_cp.folder} holds no *_probs.npz to read the confidence from")
    time, confidence = confidence_curve(with_cp.npz_path, TRIAL)

    fig, axs = plt.subplots(4, 1, figsize=(15, 5), sharex=True)
    for ax, df in zip(axs[:3], rows):
        plot_label_segments(ax, df, mapping)
    axs[3].plot(time, confidence, color="black", lw=0.8)
    axs[3].set_ylim(0, 1.05)

    for ax, row_label in zip(axs, ROW_LABELS):
        ax.set_ylabel(row_label, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(time[0], time[-1])

    out_dir = project.root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    # PDF, not EPS: the label rectangles are drawn with alpha, which EPS cannot carry.
    path = out_dir / f"{CONFIG_PATH.stem}_trial{TRIAL}.pdf"
    fig.savefig(path, bbox_inches="tight")
    return path


if __name__ == "__main__":
    print(main())
