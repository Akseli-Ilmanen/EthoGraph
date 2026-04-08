"""External format converters: crowsetta, NWB, pynapple, and other label I/O."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.labels.intervals import _rows_to_df, empty_intervals, load_mapping
from ethograph.labels.tsv_store import (
    TRIAL_META_DEFAULTS,
    init_empty_labels,
    labels_tsv_path,
    load_labels_tsv,
)

logger = logging.getLogger(__name__)

CROWSETTA_SEQ_FORMATS = [
    "aud-seq",
    "simple-seq",
    "generic-seq",
    "notmat",
    "textgrid",
    "timit",
    "yarden",
]


# ---------------------------------------------------------------------------
# Base converter
# ---------------------------------------------------------------------------


class LabelConverter:
    """Base class for converting external label sources to ethograph intervals.

    Subclasses override :meth:`extract` to pull intervals from their source
    (NWB, pynapple, crowsetta, …).  The shared :meth:`resolve_labels` method
    centralises the "TSV on disk → extract from source → empty" fallback chain
    used by every ``LoadResult``-producing function in ``data_loader``.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._label_map: dict[str, int] = {}

    @property
    def label_map(self) -> dict[str, int]:
        return self._label_map

    def extract(self, trials_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return an all-labels DataFrame (with ``trial`` column).

        Parameters
        ----------
        trials_df
            Must contain ``trial``, ``start_time``, ``stop_time`` columns.
            Required for sources with global timestamps (NWB, pynapple).
        """
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------

    def _global_to_trial_rows(
        self,
        epochs: list[dict],
        trials_df: pd.DataFrame,
    ) -> list[dict]:
        """Convert global-time epochs to trial-relative interval rows."""
        has_trial_col = "trial" in trials_df.columns
        rows: list[dict] = []
        for idx, (_, trial_row) in enumerate(trials_df.iterrows()):
            t_start, t_stop = trial_row["start_time"], trial_row["stop_time"]
            trial_id = trial_row["trial"] if has_trial_col else idx
            for ep in epochs:
                if ep["offset_s"] <= t_start or ep["onset_s"] >= t_stop:
                    continue
                label_id = self._label_map.get(ep["label_name"], 0)
                if label_id == 0:
                    continue
                rows.append({
                    "onset_s": max(0.0, ep["onset_s"] - t_start),
                    "offset_s": min(t_stop - t_start, ep["offset_s"] - t_start),
                    "labels": label_id,
                    "individual": ep.get("individual", "individual_0"),
                    "trial": trial_id,
                })
        return rows

    def _rows_to_labels_df(self, rows: list[dict]) -> pd.DataFrame:
        """Build a TSV-compatible all-labels DataFrame from row dicts."""
        if not rows:
            return init_empty_labels([])
        df = pd.DataFrame(rows)
        for col, default in TRIAL_META_DEFAULTS.items():
            df[col] = default
        return df

    # -- resolve (TSV → extract → empty) ----------------------------------

    def resolve_labels(
        self,
        source_path: str | Path,
        trial_ids: list,
        trials_df: pd.DataFrame | None = None,
        labels_path: Path | None = None,
    ) -> pd.DataFrame:
        """Load labels with fallback: existing TSV → extract from source → empty.

        Parameters
        ----------
        source_path
            Primary data file; used to derive the default TSV path.
        trial_ids
            Trial identifiers (for the empty-labels fallback).
        trials_df
            Passed to :meth:`extract` for global→trial conversion.
        labels_path
            Override the TSV path (e.g. for NWB project directories).
        """
        tsv = labels_path if labels_path is not None else labels_tsv_path(Path(source_path))
        if tsv.exists():
            logger.info("Loaded labels from %s", tsv.name)
            return load_labels_tsv(tsv)
        df = self.extract(trials_df)
        if not df.empty:
            logger.info("Extracted %d label intervals via %s", len(df), self.name)
            return df
        return init_empty_labels(trial_ids)


def resolve_labels_tsv(
    source_path: str | Path,
    trial_ids: list,
    labels_path: Path | None = None,
) -> pd.DataFrame:
    """Load labels from TSV if it exists, otherwise return empty.

    Use this when there is no converter (e.g. xarray/nc files).
    """
    tsv = labels_path if labels_path is not None else labels_tsv_path(Path(source_path))
    if tsv.exists():
        logger.info("Loaded labels from %s", tsv.name)
        return load_labels_tsv(tsv)
    return init_empty_labels(trial_ids)


# ---------------------------------------------------------------------------
# Standalone helpers (kept for backwards compat / direct use)
# ---------------------------------------------------------------------------


def build_mapping_from_labels(string_labels: list[str]) -> dict[str, int]:
    """Build a name->id mapping from a list of unique string labels.

    Sorts labels alphabetically; 0 is reserved for 'background'.
    """
    unique = sorted(set(string_labels))
    mapping = {"background": 0}
    for i, name in enumerate(unique, start=1):
        mapping[name] = i
    return mapping


def crowsetta_to_intervals(
    file_path: str | Path,
    format_name: str,
    name_to_id: dict[str, int],
    individual: str = "ind0",
) -> pd.DataFrame:
    """Convert a crowsetta annotation file to an intervals DataFrame."""
    import crowsetta

    scribe = crowsetta.Transcriber(format=format_name)
    annot = scribe.from_file(file_path).to_annot()
    seq = annot.seq

    rows: list[dict] = []
    for segment in seq.segments:
        label_str = str(segment.label)
        label_id = name_to_id.get(label_str)
        if label_id is None or label_id == 0:
            continue
        rows.append({
            "onset_s": float(segment.onset_s),
            "offset_s": float(segment.offset_s),
            "labels": label_id,
            "individual": individual,
        })

    return _rows_to_df(rows)


def extract_crowsetta_labels(
    file_path: str | Path,
    format_name: str,
) -> list[str]:
    """Extract unique string labels from a crowsetta annotation file."""
    import crowsetta

    scribe = crowsetta.Transcriber(format=format_name)
    annot = scribe.from_file(file_path).to_annot()
    return list({str(seg.label) for seg in annot.seq.segments})


def write_mapping_file(
    output_path: str | Path,
    name_to_id: dict[str, int],
) -> None:
    """Write a mapping file in '<id> <name>' format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{idx} {name}" for name, idx in sorted(name_to_id.items(), key=lambda x: x[1])]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_crowsetta_mapping(
    file_path: str | Path,
    format_name: str,
    mapping_path: str | Path,
    configs_dir: str | Path,
) -> tuple[dict[str, int], str | None, str | None]:
    """Check existing mapping against crowsetta labels; create new if needed."""
    file_labels = extract_crowsetta_labels(file_path, format_name)

    mapping_path = Path(mapping_path)
    configs_dir = Path(configs_dir)

    existing_names: set[str] = set()
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        existing_names.add(parts[1])
        except (OSError, UnicodeDecodeError):
            pass

    non_bg_labels = [l for l in file_labels if l.lower() not in ("background", "sil")]
    if not non_bg_labels:
        non_bg_labels = file_labels

    overlap = existing_names & set(non_bg_labels)

    if overlap == set(non_bg_labels) and len(overlap) > 0:
        class_to_idx, _ = load_mapping(str(mapping_path))
        return class_to_idx, None, None

    warning = None
    if overlap and overlap != set(non_bg_labels):
        missing = set(non_bg_labels) - overlap
        warning = (
            f"Mapping file contains {len(overlap)} of {len(non_bg_labels)} labels. "
            f"Missing: {sorted(missing)}"
        )

    name_to_id = build_mapping_from_labels(non_bg_labels)
    new_path = configs_dir / f"mapping_{format_name.replace('-', '_')}.txt"
    write_mapping_file(new_path, name_to_id)

    return name_to_id, str(new_path), warning


# ---------------------------------------------------------------------------
# Crowsetta converter
# ---------------------------------------------------------------------------


class CrowsettaLabelConverter(LabelConverter):
    """Convert crowsetta annotation files to ethograph intervals.

    Crowsetta labels are already in file-local time, so no trial table
    is needed for time conversion.  If a ``trials_df`` is provided the
    first trial id is attached; otherwise ``trial=1``.
    """

    name = "crowsetta"

    def __init__(
        self,
        file_path: str | Path,
        format_name: str,
        name_to_id: dict[str, int],
        individual: str = "ind0",
    ) -> None:
        super().__init__()
        self._file_path = file_path
        self._format_name = format_name
        self._label_map = dict(name_to_id)
        self._individual = individual

    def extract(self, trials_df: pd.DataFrame | None = None) -> pd.DataFrame:
        df = crowsetta_to_intervals(
            self._file_path, self._format_name, self._label_map, self._individual,
        )
        if df.empty:
            return init_empty_labels([])
        trial_id = trials_df.iloc[0]["trial"] if trials_df is not None and len(trials_df) > 0 else 1
        df["trial"] = trial_id
        for col, default in TRIAL_META_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        return df


# ---------------------------------------------------------------------------
# NWB interval label converter
# ---------------------------------------------------------------------------


class NWBLabelConverter(LabelConverter):
    """Extract behavioural-epoch labels from an NWB file.

    Supports lazy (path-based) and eager (pynwb-object) construction.
    Epochs are converted to trial-relative intervals via ``trials_df``.
    """

    name = "nwb_intervals"

    def __init__(self, nwb_path: str | Path | None = None, *, nwb=None) -> None:
        super().__init__()
        self._nwb_path = str(nwb_path) if nwb_path else None
        self._epochs: list[dict] | None = None
        if nwb is not None:
            self._load(nwb)

    def _ensure_loaded(self) -> None:
        if self._epochs is not None:
            return
        if self._nwb_path is None:
            self._epochs = []
            return
        import pynwb
        with pynwb.NWBHDF5IO(self._nwb_path, "r") as io:
            self._load(io.read())

    def _load(self, nwb) -> None:
        self._epochs = self._extract_behavioral_epochs(nwb)
        self._label_map = build_mapping_from_labels(
            sorted({e["label_name"] for e in self._epochs})
        )

    def extract(self, trials_df: pd.DataFrame | None = None) -> pd.DataFrame:
        self._ensure_loaded()
        if not self._epochs or trials_df is None or trials_df.empty:
            return init_empty_labels([])
        rows = self._global_to_trial_rows(self._epochs, trials_df)
        return self._rows_to_labels_df(rows)

    # kept for backwards compat with wizard_nwb.py
    def from_nwb(self, nwb, trials_df: pd.DataFrame) -> pd.DataFrame:
        """Extract labels from NWB and return a TSV-compatible all-labels DataFrame."""
        if self._epochs is None:
            self._load(nwb)
        return self.extract(trials_df)

    def _extract_behavioral_epochs(self, nwb) -> list[dict]:
        import pynwb
        individual = _get_nwb_individual(nwb)
        epochs: list[dict] = []

        if nwb.epochs is not None and len(nwb.epochs) > 0:
            self._collect_time_intervals(nwb.epochs, "epochs", individual, epochs)

        for mod_key, mod in nwb.processing.items():
            for iface_key, iface in mod.data_interfaces.items():
                if isinstance(iface, pynwb.epoch.TimeIntervals):
                    self._collect_time_intervals(iface, f"{mod_key}/{iface_key}", individual, epochs)
                elif isinstance(iface, pynwb.behavior.BehavioralEpochs):
                    for series_name, series in iface.interval_series.items():
                        self._collect_interval_series(series, f"{mod_key}/{iface_key}/{series_name}", individual, epochs)

        return epochs

    @staticmethod
    def _collect_time_intervals(table, source: str, individual: str, out: list[dict]) -> None:
        df = table.to_dataframe()
        label_col = "label" if "label" in df.columns else None
        for _, row in df.iterrows():
            out.append({
                "onset_s": float(row["start_time"]),
                "offset_s": float(row["stop_time"]),
                "label_name": str(row[label_col]) if label_col else source.split("/")[-1],
                "individual": individual,
                "source": source,
            })

    @staticmethod
    def _collect_interval_series(series, source: str, individual: str, out: list[dict]) -> None:
        data = np.asarray(series.data[:])
        timestamps = np.asarray(series.timestamps[:])
        label_name = getattr(series, "name", source.split("/")[-1])
        starts = np.where(data > 0)[0]
        for i in starts:
            j = next((k for k in range(i + 1, len(data)) if data[k] < 0), None)
            if j is not None:
                out.append({
                    "onset_s": float(timestamps[i]),
                    "offset_s": float(timestamps[j]),
                    "label_name": label_name,
                    "individual": individual,
                    "source": source,
                })


# ---------------------------------------------------------------------------
# Pynapple interval label converter
# ---------------------------------------------------------------------------


class PynappleLabelConverter(LabelConverter):
    """Extract labels from pynapple IntervalSet objects.

    Collects every ``nap.IntervalSet`` in the data dict whose key is
    **not** ``"trials"`` or ``"epochs"`` (those are trial boundaries,
    not labels).  Each IntervalSet name becomes a label class.
    """

    name = "pynapple_intervals"

    SKIP_KEYS = frozenset({"trials", "epochs"})

    def __init__(self, data: dict, trials_ep=None) -> None:
        super().__init__()
        self._trials_df = trials_df_from_intervalset(trials_ep)
        self._epochs = self._extract_interval_epochs(data)
        if self._epochs:
            self._label_map = build_mapping_from_labels(
                sorted({e["label_name"] for e in self._epochs})
            )

    def _extract_interval_epochs(self, data: dict) -> list[dict]:
        import pynapple as nap

        epochs: list[dict] = []
        for key, obj in data.items():
            if not isinstance(obj, nap.IntervalSet):
                continue
            if key.lower() in self.SKIP_KEYS:
                continue
            for i in range(len(obj)):
                epochs.append({
                    "onset_s": float(obj.start[i]),
                    "offset_s": float(obj.end[i]),
                    "label_name": key,
                    "individual": "individual_0",
                })
        return epochs

    def extract(self, trials_df: pd.DataFrame | None = None) -> pd.DataFrame:
        t_df = trials_df if trials_df is not None else self._trials_df
        if not self._epochs or t_df.empty:
            return init_empty_labels([])
        rows = self._global_to_trial_rows(self._epochs, t_df)
        return self._rows_to_labels_df(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_nwb_individual(nwb) -> str:
    subject = getattr(nwb, "subject", None)
    if subject:
        sid = getattr(subject, "subject_id", None)
        if sid:
            return str(sid)
    return "individual_0"


def trials_df_from_intervalset(trials_ep) -> pd.DataFrame:
    """Build a trials DataFrame from a pynapple IntervalSet."""
    if trials_ep is None or len(trials_ep) == 0:
        return pd.DataFrame(columns=["trial", "start_time", "stop_time"])
    return pd.DataFrame({
        "trial": list(range(1, len(trials_ep) + 1)),
        "start_time": [float(s) for s in trials_ep.start],
        "stop_time": [float(e) for e in trials_ep.end],
    })
