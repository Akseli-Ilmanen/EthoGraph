"""Feature engineering: write the materialised dataset.

Layout (the action-segmentation literature's, so third-party models plug in
directly), under ``{root}/data/{features.name}/``::

    features / {key}.npy  # (F, T) float32, session-level preprocessed
    groundTruth / {key}.txt  # one class name per frame
    mapping.txt  # "{index} {name}" — contiguous, 0 = background
    index.tsv  # key, session_id, source, trial, individual, n_frames, fs, n_labelled
    columns.yaml  # the column layout (names, normalise flags, vector groups)
    classes.yaml  # class index ↔ label id

``key`` is ``{session_id}_trial{trial}_{individual}``. Role assignment and
normalisation statistics belong to a run, not to the dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ethograph.io import schema
from ethograph.segment.config import SegmentConfig
from ethograph.segment.samples import (
    ClassTable,
    ColumnLayout,
    build_sample_features,
    class_table,
    dense_targets,
    sample_key,
)
from ethograph.segment.sessions import Session, filter_trials, open_session
from ethograph.utils.logging import log_to_file

logger = logging.getLogger(__name__)

INDEX_FILE = "index.tsv"
COLUMNS_FILE = "columns.yaml"
CLASSES_FILE = "classes.yaml"
MAPPING_FILE = "mapping.txt"


def materialise(config: SegmentConfig, sessions: list[Session] | None = None) -> Path:
    """Write the materialised dataset for every session in the config."""
    data_dir = config.data_dir
    with log_to_file(data_dir / "materialise.log"):
        return _materialise_run(config, data_dir, sessions)


def _materialise_run(config: SegmentConfig, data_dir: Path, sessions: list[Session] | None) -> Path:
    for sub in ("features", "groundTruth"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    classes = class_table(config)

    rows: list[dict] = []
    layout: ColumnLayout | None = None
    sessions = sessions if sessions is not None else [open_session(s, config) for s in config.sessions]
    for session in sessions:
        trials = filter_trials(session, config.trials)
        individuals = session.individuals(config)
        logger.info("%s: %d trials × %d individuals", session.id, len(trials), len(individuals))
        if not session.declares_schema():
            logger.warning(
                "%s declares no `kind` on any variable, so train.drop_kinds has nothing to drop and "
                "normalise=0 columns will be z-scored. Describe the variables when you build the session "
                "(ethograph.io.schema.describe), or write %s for a backend without attrs.",
                session.id,
                schema.sidecar_path(session.source),
            )
        for window in session.trial_windows(trials):
            labels = session.curated_labels(window.trial)
            for individual in individuals:
                key = sample_key(session.id, window.trial, individual)
                time, x, sample_layout = build_sample_features(config, session, window, individual, individuals)
                if layout is None:
                    layout = sample_layout
                else:
                    layout.check(sample_layout, f"{session.id} trial {window.trial} individual {individual}")
                    _check_fs(layout.fs, sample_layout.fs, key)
                y, n_labelled = dense_targets(labels, time, individual, classes)
                np.save(data_dir / "features" / f"{key}.npy", x)
                _write_ground_truth(data_dir / "groundTruth" / f"{key}.txt", y, classes)
                rows.append(
                    {
                        "key": key,
                        "session_id": session.id,
                        "source": str(session.source),
                        "trial": window.trial,
                        "individual": individual,
                        "n_frames": int(x.shape[1]),
                        "fs": float(sample_layout.fs),
                        "n_labelled": int(n_labelled),
                    }
                )
    if layout is None or not rows:
        raise ValueError("No samples were materialised — check trials.where and the sessions' trials.")

    pd.DataFrame(rows).to_csv(data_dir / INDEX_FILE, sep="\t", index=False)
    (data_dir / COLUMNS_FILE).write_text(yaml.safe_dump(layout.to_dict(), sort_keys=False), encoding="utf-8")
    (data_dir / CLASSES_FILE).write_text(yaml.safe_dump(classes.to_dict(), sort_keys=False), encoding="utf-8")
    (data_dir / MAPPING_FILE).write_text(
        "".join(f"{i} {name}\n" for i, name in enumerate(classes.names)), encoding="utf-8"
    )
    logger.info("Materialised %d samples × %d columns → %s", len(rows), layout.n_features, data_dir)
    return data_dir


def _check_fs(fs_ref: float, fs: float, key: str) -> None:
    if not np.isclose(fs_ref, fs, rtol=1e-3):
        raise ValueError(f"{key}: sampling rate {fs:.6g} Hz differs from the dataset's {fs_ref:.6g} Hz")


def _write_ground_truth(path: Path, y: np.ndarray, classes: ClassTable) -> None:
    names = classes.names
    path.write_text("".join(f"{names[int(i)]}\n" for i in y), encoding="utf-8")


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------


def read_index(data_dir: Path) -> pd.DataFrame:
    path = data_dir / INDEX_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{data_dir} holds no materialised dataset — call Project.materialise() first")
    return pd.read_csv(path, sep="\t", dtype={"individual": str})


def read_layout(data_dir: Path) -> ColumnLayout:
    return ColumnLayout.from_dict(yaml.safe_load((data_dir / COLUMNS_FILE).read_text(encoding="utf-8")))


def read_classes(data_dir: Path) -> ClassTable:
    return ClassTable.from_dict(yaml.safe_load((data_dir / CLASSES_FILE).read_text(encoding="utf-8")))


def load_sample(data_dir: Path, key: str, classes: ClassTable) -> tuple[np.ndarray, np.ndarray]:
    """``(x (F, T) float32, y (T,) int64)`` of one materialised sample."""
    x = np.load(data_dir / "features" / f"{key}.npy")
    names = (data_dir / "groundTruth" / f"{key}.txt").read_text(encoding="utf-8").splitlines()
    index_of = {name: i for i, name in enumerate(classes.names)}
    y = np.fromiter((index_of[n] for n in names), dtype=np.int64, count=len(names))
    n = min(x.shape[1], len(y))
    return x[:, :n], y[:n]
