"""Building one sample — shared by feature engineering and inference.

A **sample** is one (trial, individual): that individual's feature columns
over the trial and its labels as actor. The column layout is the config's
``features.columns`` with the individual dim pinned to the sample's
individual (spelled ``self`` in layout names) and a second individual dim,
``other``, enumerating the remaining individuals in dataset order (spelled
``other1``, ``other2``, …). Every session must therefore carry the same
number of individuals — a layout that differs from the materialised one is
an error, never a silent column shuffle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ethograph.features.changepoints import CP_BINARY_SUFFIX
from ethograph.features.columns import (
    FeatureColumn,
    column_name,
    enumerate_columns,
    expand_dim_values,
    extract_features,
    sampling_rate,
)
from ethograph.io.catalog import INDIVIDUAL_DIMS, SPACE_DIM
from ethograph.io.schema import is_normalise, kind_of
from ethograph.labels.intervals import load_label_mapping, states_only
from ethograph.segment.config import SegmentConfig
from ethograph.segment.preprocess import preprocess_session_level
from ethograph.segment.sessions import Session, TrialWindow, individual_dim_name

OTHER_DIM = "other"
SELF_TOKEN = "self"
BACKGROUND_NAME = "background"


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


@dataclass
class ClassTable:
    """Contiguous class indices ↔ label ids of one branch; index 0 is background."""

    label_ids: list[int]
    names: list[str]

    def __post_init__(self) -> None:
        if self.label_ids[0] != 0 or self.names[0] != BACKGROUND_NAME:
            raise ValueError("ClassTable must start with background (id 0)")
        self.id_to_index = {lid: i for i, lid in enumerate(self.label_ids)}

    @property
    def n_classes(self) -> int:
        return len(self.label_ids)

    def to_dict(self) -> dict:
        return {"label_ids": list(self.label_ids), "names": list(self.names)}

    @classmethod
    def from_dict(cls, data: dict) -> ClassTable:
        return cls([int(i) for i in data["label_ids"]], [str(n) for n in data["names"]])

    def indices(self, label_ids: np.ndarray) -> np.ndarray:
        """Label ids → class indices; ids outside the table become background."""
        out = np.zeros(len(label_ids), dtype=np.int64)
        for i, lid in enumerate(label_ids):
            out[i] = self.id_to_index.get(int(lid), 0)
        return out

    def ids(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.label_ids, dtype=np.int64)[np.asarray(indices, dtype=np.int64)]


def class_table(config: SegmentConfig) -> ClassTable:
    """The branch's state classes from ``mapping.txt`` (plus background)."""
    labels_cfg = config.features.labels
    assert labels_cfg is not None
    mapping = load_label_mapping(labels_cfg.mapping)
    chosen = []
    for lid, info in sorted(mapping.items()):
        if lid == 0:
            continue
        if int(info.get("branch", 0)) != labels_cfg.branch:
            continue
        if info.get("event_type", "state") != "state":
            continue
        if labels_cfg.classes is not None and lid not in labels_cfg.classes:
            continue
        chosen.append((int(lid), str(info["name"])))
    if labels_cfg.classes is not None:
        missing = set(labels_cfg.classes) - {lid for lid, _ in chosen}
        if missing:
            raise ValueError(
                f"features.labels.classes names ids {sorted(missing)} that are not state classes of branch "
                f"{labels_cfg.branch} in {labels_cfg.mapping}"
            )
    if not chosen:
        raise ValueError(f"Branch {labels_cfg.branch} of {labels_cfg.mapping} has no state classes to predict.")
    return ClassTable([0, *(lid for lid, _ in chosen)], [BACKGROUND_NAME, *(name for _, name in chosen)])


# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------


@dataclass
class ColumnLayout:
    """The model's input layout: column names with ``self``/``otherN`` tokens."""

    names: list[str]
    features: list[str]
    normalise: list[bool]
    #: Index groups of columns spanning the space dim of one vector (x, y[, z]).
    vector_groups: list[list[int]] = field(default_factory=list)
    fs: float = 0.0
    #: Each column's ``attrs["kind"]`` (``None`` when the dataset does not say).
    #: Recorded so an ablation can drop a whole category at train time,
    #: without re-materialising the dataset.
    kinds: list[str | None] = field(default_factory=list)
    #: The changepoint expansion's resolved scales (``sigmas``, ``horizon``,
    #: ``max_length``, ``note``), when the config has one — part of the
    #: layout because two datasets with the same names but different
    #: horizons are different inputs. ``materialise`` writes it; train and
    #: infer read their scales back from it.
    changepoint_features: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.kinds:
            self.kinds = [None] * len(self.names)

    @property
    def n_features(self) -> int:
        return len(self.names)

    def candidate_columns(self) -> np.ndarray:
        """Indices of the raw changepoint-mask copies (``{var}_cp_binary``): the candidate frames a loss may read.

        Read off the *full* layout, never an ablated one — which candidates
        exist is a property of the data, not of what the model was shown.
        """
        return np.array([i for i, f in enumerate(self.features) if f.endswith(CP_BINARY_SUFFIX)], dtype=int)

    def keep_mask(self, drop_kinds: Iterable[str]) -> np.ndarray:
        """Boolean mask over columns, ``False`` for every column of a dropped kind.

        Columns whose kind is unknown are always kept: dropping happens only
        on a positive declaration, so an undeclared dataset ablates to itself.
        """
        unwanted = set(drop_kinds)
        if not unwanted:
            return np.ones(len(self.names), dtype=bool)
        return np.array([k not in unwanted for k in self.kinds], dtype=bool)

    def subset(self, mask: np.ndarray) -> ColumnLayout:
        """This layout restricted to the columns *mask* keeps (indices renumbered)."""
        keep = [i for i, on in enumerate(np.asarray(mask, dtype=bool)) if on]
        remap = {old: new for new, old in enumerate(keep)}
        return ColumnLayout(
            names=[self.names[i] for i in keep],
            features=[self.features[i] for i in keep],
            normalise=[self.normalise[i] for i in keep],
            vector_groups=[[remap[i] for i in group] for group in self.vector_groups if all(i in remap for i in group)],
            fs=self.fs,
            kinds=[self.kinds[i] for i in keep],
            changepoint_features=self.changepoint_features,
        )

    def to_dict(self) -> dict:
        out = {
            "fs": float(self.fs),
            "names": list(self.names),
            "features": list(self.features),
            "normalise": [bool(v) for v in self.normalise],
            "vector_groups": [list(map(int, g)) for g in self.vector_groups],
            "kinds": [None if k is None else str(k) for k in self.kinds],
        }
        if self.changepoint_features is not None:
            out["changepoint_features"] = dict(self.changepoint_features)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> ColumnLayout:
        return cls(
            names=list(data["names"]),
            features=list(data["features"]),
            normalise=[bool(v) for v in data["normalise"]],
            vector_groups=[list(g) for g in data.get("vector_groups", [])],
            fs=float(data["fs"]),
            kinds=list(data.get("kinds") or []),
            changepoint_features=data.get("changepoint_features"),
        )

    def check(self, other: ColumnLayout, what: str) -> None:
        if self.names != other.names:
            missing = [n for n in self.names if n not in other.names]
            extra = [n for n in other.names if n not in self.names]
            raise ValueError(
                f"{what}: column layout differs from the materialised dataset.\n"
                f"  missing: {missing}\n  extra:   {extra}"
            )
        mine, theirs = self.changepoint_features, other.changepoint_features
        if mine is not None and theirs is not None:
            keys = ("sigmas", "horizon", "max_length")
            if any(mine.get(k) != theirs.get(k) for k in keys):
                got = {k: theirs.get(k) for k in keys}
                want = {k: mine.get(k) for k in keys}
                raise ValueError(
                    f"{what}: changepoint features were expanded at different scales than the "
                    f"materialised dataset's ({got} vs {want})."
                )


def sample_features_spec(
    config: SegmentConfig,
    session: Session,
    loader: Any,
    individual: str,
    others: list[str],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, str]]:
    """The per-sample ``features`` spec plus the value→token map for layout names."""
    tokens: dict[str, str] = {individual: SELF_TOKEN}
    tokens.update({o: f"other{i + 1}" for i, o in enumerate(others)})
    spec: dict[str, dict[str, list[str]]] = {}
    for feature, dims in config.features.columns.items():
        out: dict[str, list[str]] = {}
        for dim, values in (dims or {}).items():
            if dim == OTHER_DIM and values == "*":
                out[dim] = list(others)
            else:
                out[dim] = expand_dim_values(values)
        ind_dim = individual_dim_name(loader, feature)
        if ind_dim is not None:
            out[ind_dim] = [individual]
        spec[feature] = out
    return spec, tokens


def sample_columns(spec: dict[str, dict[str, list[str]]], config: SegmentConfig) -> list[FeatureColumn]:
    """The sample's columns: *spec* expanded the way ``features`` asks for.

    Every per-column list below (names, kinds, normalise flags) is built from
    this one enumeration, so they index the matrix ``extract_features``
    returns — which expands the same way.
    """
    return enumerate_columns(spec, sin_cos=config.features.sin_cos)


def layout_names(columns: list[FeatureColumn], tokens: dict[str, str]) -> list[str]:
    """Column names, with the individual axis canonicalised to one spelling.

    A session may spell its individual dim ``individual`` or ``individuals``
    (:data:`~ethograph.io.catalog.INDIVIDUAL_DIMS`); the *value* along it is
    already normalised to ``self``/``otherN`` so a model can run on a session
    naming different animals — canonicalising the *key* too is what makes
    that hold across a session that spells the dim differently as well.
    """
    names = []
    for col in columns:
        sel = {}
        for d, v in col.selections.items():
            if d in _INDIVIDUAL_LIKE:
                sel[_CANONICAL_INDIVIDUAL_DIM] = tokens.get(v, v)
            elif d == OTHER_DIM:
                sel[d] = tokens.get(v, v)
            else:
                sel[d] = v
        names.append(column_name(col.feature, sel, col.derivative, col.circular))
    return names


_INDIVIDUAL_LIKE = set(INDIVIDUAL_DIMS)
_CANONICAL_INDIVIDUAL_DIM = INDIVIDUAL_DIMS[0]


def _vector_groups(columns: list[FeatureColumn]) -> list[list[int]]:
    """Index groups of the columns spanning one vector's space dim.

    A derived column (a derivative, an angle component) belongs to no vector:
    the geometric augmentations rotate and mirror these groups, which only
    means something for the coordinates themselves.
    """
    groups: dict[tuple, list[int]] = {}
    for i, col in enumerate(columns):
        if SPACE_DIM not in col.selections or col.derivative or col.circular:
            continue
        key = (col.feature, tuple((d, v) for d, v in col.selections.items() if d != SPACE_DIM))
        groups.setdefault(key, []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def _normalise_flags(
    columns: list[FeatureColumn], session: Session, trial: int | str, config: SegmentConfig
) -> list[bool]:
    """Which columns are z-scored (and percentile-clipped).

    An angle component is never one of them: ``sin``/``cos`` already live in
    ``[-1, 1]`` and mean what they say there — the same statement
    ``attrs["normalise"] = 0`` makes about a unit vector.
    """
    exclude = set(config.features.preprocess.zscore_exclude)
    flags = []
    for col in columns:
        declared = is_normalise(session.variable_attrs(col.feature, trial))
        flags.append(col.circular is None and declared and col.feature not in exclude)
    return flags


def _column_kinds(columns: list[FeatureColumn], session: Session, trial: int | str) -> list[str | None]:
    return [kind_of(session.variable_attrs(col.feature, trial)) for col in columns]


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


@dataclass
class SampleData:
    key: str
    session_id: str
    trial: int | str
    individual: str
    time: np.ndarray  # trial-relative
    x: np.ndarray  # (F, T) float32, session-level preprocessed
    layout: ColumnLayout
    y: np.ndarray | None = None  # (T,) class indices
    n_labelled: int = 0


def sample_key(session_id: str, trial: int | str, individual: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(individual))
    return f"{session_id}_trial{trial}_{safe}"


def others_of(individual: str, individuals: list[str]) -> list[str]:
    return [i for i in individuals if i != individual]


def build_sample_features(
    config: SegmentConfig,
    session: Session,
    window: TrialWindow,
    individual: str,
    individuals: list[str],
) -> tuple[np.ndarray, np.ndarray, ColumnLayout]:
    """Select, preprocess and lay out one sample → ``(time, x (F, T), layout)``."""
    others = others_of(individual, individuals)
    spec, tokens = sample_features_spec(config, session, window.loader, individual, others)
    sin_cos = list(config.features.sin_cos)
    time, data = extract_features(
        window.loader,
        spec,
        window.t0,
        window.t1,
        sin_cos=sin_cos,
        units=_declared_units(session, window.trial, sin_cos),
    )
    time = time - window.shift
    columns = sample_columns(spec, config)
    data = _apply_likelihood_threshold(data, columns, config, window, individual)
    normalise = _normalise_flags(columns, session, window.trial, config)
    data = preprocess_session_level(data, config.features.preprocess, np.asarray(normalise, dtype=bool))
    layout = ColumnLayout(
        names=layout_names(columns, tokens),
        features=[c.feature for c in columns],
        normalise=normalise,
        vector_groups=_vector_groups(columns),
        fs=sampling_rate(time),
        kinds=_column_kinds(columns, session, window.trial),
    )
    return time, data.T.astype(np.float32), layout


def _declared_units(session: Session, trial: int | str, features: list[str]) -> dict[str, Any]:
    """The ``units`` attr of each of *features* that declares one."""
    units = {name: session.variable_attrs(name, trial).get("units") for name in features}
    return {name: value for name, value in units.items() if value is not None}


def _apply_likelihood_threshold(
    data: np.ndarray,
    columns: list[FeatureColumn],
    config: SegmentConfig,
    window: TrialWindow,
    individual: str,
) -> np.ndarray:
    pre = config.features.preprocess
    if pre.likelihood_threshold is None:
        return data
    loader = window.loader
    if (
        pre.likelihood_feature not in loader.feature_dims(pre.likelihood_feature)
        and loader.select(pre.likelihood_feature, {}, window.t0, window.t1) is None
    ):
        raise ValueError(
            f"preprocess.likelihood_threshold is set but the session has no {pre.likelihood_feature!r} feature"
        )
    ind_dim = individual_dim_name(loader, pre.likelihood_feature)
    data = np.array(data, dtype=np.float64, copy=True)
    for i, col in enumerate(columns):
        if "keypoint" not in col.selections:
            continue
        sel = {"keypoint": col.selections["keypoint"]}
        if ind_dim is not None:
            sel[ind_dim] = individual
        conf = loader.select(pre.likelihood_feature, sel, window.t0, window.t1)
        if conf is None:
            continue
        values = np.asarray(conf.data, dtype=np.float64)
        n = min(len(values), data.shape[0])
        low = values[:n] < pre.likelihood_threshold
        data[:n, i][low] = np.nan
    return data


def dense_targets(
    labels: pd.DataFrame, time: np.ndarray, individual: str, classes: ClassTable
) -> tuple[np.ndarray, int]:
    """Per-frame class indices for *individual* from its state labels.

    Interval ends are inclusive (the convention ``dense_to_intervals`` reads
    back). Returns the target and how many rows contributed.
    """
    y = np.zeros(len(time), dtype=np.int64)
    if labels.empty:
        return y, 0
    df = states_only(labels)
    df = df[df["individual"].astype(str) == str(individual)]
    df = df[df["labels"].astype(int).isin(classes.label_ids[1:])]
    df = df.sort_values("onset_s")
    n = 0
    for _, row in df.iterrows():
        i0, i1 = frame_span(time, float(row["onset_s"]), float(row["offset_s"]))
        if i1 <= i0:
            continue
        y[i0:i1] = classes.id_to_index[int(row["labels"])]
        n += 1
    return y, n


def frame_span(time: np.ndarray, onset: float, offset: float) -> tuple[int, int]:
    """``[i0, i1)`` frame slice of an interval whose ends are inclusive, with
    half-a-frame tolerance so a boundary sitting on a sample counts."""
    tol = 0.5 / sampling_rate(time) if len(time) > 1 else 0.0
    i0 = int(np.searchsorted(time, onset - tol, side="left"))
    i1 = int(np.searchsorted(time, offset + tol, side="right"))
    return i0, i1
