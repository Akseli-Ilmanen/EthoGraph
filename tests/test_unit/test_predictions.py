"""PredictionsStore: reading a segment.inference prediction folder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ethograph.labels.predictions import PredictionsStore
from ethograph.labels.tsv_store import TSV_COLUMNS, save_labels_tsv


def _prediction_folder(tmp_path: Path, keys: dict[str, np.ndarray], name: str = "sess_predictions.tsv") -> Path:
    folder = tmp_path / "predictions_mstcn_20260101_000000"
    folder.mkdir()
    df = pd.DataFrame(columns=TSV_COLUMNS)
    df.loc[0] = {
        "trial": 1,
        "individual": "A",
        "individual_rec": "",
        "labels": 3,
        "onset_s": 1.0,
        "offset_s": 2.0,
        "event_type": "state",
        "confidence": 0.9,
        "labeling_method": "automated",
        "changepoint_corrected": 0,
        "prediction_source": "mstcn_20260101_000000",
        "n_samples": 100,
    }
    save_labels_tsv(folder / name, df)
    if keys:
        np.savez_compressed(folder / "sess_probs.npz", **keys)
    return folder


def _probs(high: float, n: int = 10) -> np.ndarray:
    """(n, 2) softmax-like rows: constant high confidence in one class."""
    p = np.empty((n, 2), dtype=np.float32)
    p[:, 0] = high
    p[:, 1] = 1 - high
    return p


def test_refuses_a_folder_with_no_predictions_tsv(tmp_path: Path):
    empty = tmp_path / "not_a_predictions_folder"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        PredictionsStore(empty)


def test_reads_a_legacy_labels_tsv_folder(tmp_path: Path):
    """A folder written before predictions were renamed off `_labels.tsv` still loads."""
    folder = _prediction_folder(tmp_path, {}, name="sess_labels.tsv")
    store = PredictionsStore(folder)
    assert store.tsv_path.name == "sess_labels.tsv"


def test_load_all_reads_the_tsv_unmodified(tmp_path: Path):
    folder = _prediction_folder(tmp_path, {})
    store = PredictionsStore(folder)
    df, levels = store.load_all(dt=None, individual="A")
    assert list(df["trial"]) == [1]
    assert df.iloc[0]["confidence"] == pytest.approx(0.9)
    assert levels == {}


def test_get_confidence_picks_the_matching_individual(tmp_path: Path):
    folder = _prediction_folder(
        tmp_path,
        {
            "sess_trial1_A": _probs(high=0.95),
            "sess_trial1_A_time": np.linspace(0, 1, 10),
            "sess_trial1_B": _probs(high=0.55),
            "sess_trial1_B_time": np.linspace(0, 1, 10),
        },
    )
    store = PredictionsStore(folder)

    conf_a = store.get_confidence(1, dt=None, individual="A")
    conf_b = store.get_confidence(1, dt=None, individual="B")
    assert conf_a is not None and conf_b is not None
    assert conf_a.mean() > conf_b.mean()

    assert store.get_confidence(2, dt=None, individual="A") is None  # no such trial


def test_get_confidence_none_without_npz(tmp_path: Path):
    folder = _prediction_folder(tmp_path, {})
    store = PredictionsStore(folder)
    assert store.get_confidence(1, dt=None, individual="A") is None
