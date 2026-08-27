"""Inference: a trained run over sessions → one prediction set per session.

Beside each session's own ``labels/`` folder (the same one label backups and
LightGBM onset-model runs use, see :mod:`ethograph.labels.onset_curves`), one
folder per call to :func:`infer`, named ``predictions_{run_name}_{timestamp}``
so a re-run never overwrites an earlier one::

    {stem}_labels.tsv   # the GUI's native labels format, labeling_method=automated
    {stem}_probs.npz    # per sample: "{key}" → (T, C) float16, "{key}_time" → (T,)

The TSV is what the GUI loads and compares; the ``.npz`` exists only for the
confidence overlay.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml

from ethograph.labels.intervals import LABELING_AUTOMATED, NO_RECIPIENT
from ethograph.labels.ml import dense_to_intervals
from ethograph.labels.onset_curves import labels_dir
from ethograph.labels.tsv_store import save_labels_tsv
from ethograph.segment.config import SegmentConfig, load_config
from ethograph.segment.materialise import COLUMNS_FILE
from ethograph.segment.models import as_output, build_model
from ethograph.segment.postprocess import postprocess_intervals
from ethograph.segment.preprocess import NormStats
from ethograph.segment.samples import ClassTable, ColumnLayout, build_sample_features, sample_key
from ethograph.segment.sessions import Session, changepoint_times, filter_trials, open_session
from ethograph.segment.train import BEST_FILE, STATS_FILE
from ethograph.utils.device import resolve_device
from ethograph.utils.logging import log_to_file

logger = logging.getLogger(__name__)

PREDICTIONS_PREFIX = "predictions"


def prediction_run_dir(session_path: Path, run_name: str, timestamp: str) -> Path:
    """Where one inference run's outputs for one session are written.

    One folder per call, so cross-validation folds and repeated ad-hoc
    ``infer()`` calls never overwrite each other's predictions.
    """
    return labels_dir(session_path) / f"{PREDICTIONS_PREFIX}_{run_name}_{timestamp}"


@dataclass
class Run:
    run_dir: Path
    config: SegmentConfig
    layout: ColumnLayout
    classes: ClassTable
    stats: NormStats
    model: torch.nn.Module
    device: torch.device
    #: The ablation this run was trained with (``None`` = every column).
    keep: np.ndarray | None = None

    @property
    def name(self) -> str:
        return self.run_dir.name


def resolve_run_dir(config: SegmentConfig, run: str | Path | None) -> Path:
    """Resolve *run* (a path, an exact run dir name, or a base name) to a trained run directory.

    ``train()`` names each run ``{base_name}_{timestamp}`` so it never
    overwrites another run. A *run* (or, when ``None``, ``config.infer.run``
    / the config's own base name) that does not match a directory exactly is
    tried as that base name, resolving to the most recently trained run for
    it.
    """
    from ethograph.segment.train import run_name_for

    run = run or config.infer.run or run_name_for(config)
    candidate = Path(run)
    if candidate.is_dir() and (candidate / BEST_FILE).is_file():
        return candidate.resolve()
    run_dir = config.run_dir(str(run))
    if (run_dir / BEST_FILE).is_file():
        return run_dir
    return _latest_run_dir(config, str(run))


def _latest_run_dir(config: SegmentConfig, base_name: str) -> Path:
    """The most recently trained ``{base_name}_{timestamp}`` run directory."""
    candidates = sorted(p.name for p in config.runs_dir.glob(f"{base_name}_*") if (p / BEST_FILE).is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No trained run named {base_name!r} (or {base_name}_<timestamp>) under {config.runs_dir}"
        )
    return config.run_dir(candidates[-1])


def load_run(run_dir: Path, device: str | None = None) -> Run:
    run_dir = Path(run_dir)
    config = load_config(run_dir / "config.yaml")
    layout = ColumnLayout.from_dict(yaml.safe_load((run_dir / COLUMNS_FILE).read_text(encoding="utf-8")))
    classes = ClassTable.from_dict(yaml.safe_load((run_dir / "classes.yaml").read_text(encoding="utf-8")))
    stats = NormStats.load(run_dir / STATS_FILE)
    dev = torch.device(resolve_device(device or config.train.device))
    # columns.yaml is the *full* layout, so a freshly built sample can be
    # checked against it; the run's own drop_kinds then re-derives the
    # ablation it was trained with.
    keep = layout.keep_mask(config.train.drop_kinds)
    n_features = int(keep.sum())
    model = build_model(config.model.architecture, config.model.params, n_features, classes.n_classes)
    model.load_state_dict(torch.load(run_dir / BEST_FILE, map_location=dev, weights_only=True))
    model.to(dev).eval()
    return Run(run_dir, config, layout, classes, stats, model, dev, keep=None if keep.all() else keep)


def predict_probabilities(run: Run, x: np.ndarray) -> np.ndarray:
    """``(F, T)`` preprocessed features → ``(T, C)`` probabilities."""
    if run.keep is not None:
        x = x[run.keep]
    xn = torch.from_numpy(np.ascontiguousarray(run.stats.apply(x))).unsqueeze(0).to(run.device)
    mask = torch.ones(1, 1, xn.shape[-1], device=run.device)
    with torch.no_grad():
        output = as_output(run.model(xn, mask))
        return torch.softmax(output.logits[-1], dim=1)[0].T.cpu().numpy()


def _segment_confidence(conf: np.ndarray, time: np.ndarray, onset: float, offset: float) -> float:
    m = (time >= onset) & (time <= offset)
    return float(conf[m].mean()) if m.any() else float(conf.max())


def infer_session(config: SegmentConfig, run: Run, session: Session, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Predict every (trial, individual) of *session*; returns the two written paths."""
    trials = filter_trials(session, config.trials)
    individuals = session.individuals(config)
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    pcfg = config.infer.postprocess
    step = int(run.config.train.subsample)
    trials_without_cp: list[int | str] = []
    for window in session.trial_windows(trials):
        for individual in individuals:
            time, x, layout = build_sample_features(config, session, window, individual, individuals)
            run.layout.check(layout, f"{session.id} trial {window.trial} individual {individual}")
            if step > 1:  # the rate this run was trained at, so its receptive field means the same thing
                time, x = time[::step], x[:, ::step]
            probs = predict_probabilities(run, x)
            cp = (
                changepoint_times(session, window.trial, {**pcfg.changepoints}) if pcfg.changepoint_correction else None
            )
            if cp is not None and len(cp) == 0:
                trials_without_cp.append(window.trial)
            indices = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            ids = run.classes.ids(indices)
            intervals = dense_to_intervals(ids, [individual], time_coord=time)
            intervals = postprocess_intervals(intervals, pcfg, cp)
            key = sample_key(session.id, window.trial, individual)
            arrays[key] = probs.astype(np.float16)
            arrays[f"{key}_time"] = time.astype(np.float64)
            for _, seg in intervals.iterrows():
                rows.append(
                    {
                        "trial": window.trial,
                        "individual": individual,
                        "individual_rec": NO_RECIPIENT,
                        "labels": int(seg["labels"]),
                        "onset_s": float(seg["onset_s"]),
                        "offset_s": float(seg["offset_s"]),
                        "event_type": "state",
                        "confidence": _segment_confidence(conf, time, float(seg["onset_s"]), float(seg["offset_s"])),
                        "labeling_method": LABELING_AUTOMATED,
                        "changepoint_corrected": int(
                            bool(pcfg.changepoint_correction and cp is not None and len(cp) > 0)
                        ),
                        "prediction_source": run.name,
                        "n_samples": int(len(time)),
                    }
                )
    if trials_without_cp:
        logger.warning(
            "%s: changepoint_correction is on but %d/%d trials have no changepoints — those boundaries "
            "were not snapped. Check that the session's changepoint variables are described "
            "(ethograph.io.schema.changepoint_attrs) and that infer.postprocess.changepoints selects them.",
            session.id,
            len(trials_without_cp),
            len(trials) * len(individuals),
        )
    out_dir = (
        out_dir
        if out_dir is not None
        else prediction_run_dir(session.source, run.name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / f"{session.stem}_predictions.tsv"
    npz_path = out_dir / f"{session.stem}_probs.npz"
    df = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=[
                "trial",
                "individual",
                "individual_rec",
                "labels",
                "onset_s",
                "offset_s",
                "event_type",
                "confidence",
                "labeling_method",
                "changepoint_corrected",
                "prediction_source",
                "n_samples",
            ]
        )
    )
    save_labels_tsv(tsv_path, df)
    np.savez_compressed(npz_path, **arrays)
    logger.info("%s: %d predicted labels → %s", session.id, len(rows), tsv_path)
    return tsv_path, npz_path


def inference(
    config: SegmentConfig,
    run: str | Path | None = None,
    sessions: Iterable[str | Path] | None = None,
) -> list[Path]:
    """Predict every session of the config; *sessions* narrows that to a few.

    A session is named by its full ``source`` path or just the file's stem,
    which is how a cross-validation fold asks for the one session it held
    out.
    """
    loaded = load_run(resolve_run_dir(config, run))
    specs = config.select_sessions(sessions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with log_to_file(loaded.run_dir / "infer.log"):
        written = []
        for spec in specs:
            session = open_session(spec, config)
            out_dir = prediction_run_dir(session.source, loaded.name, timestamp)
            tsv, _ = infer_session(config, loaded, session, out_dir=out_dir)
            written.append(tsv)
        return written
