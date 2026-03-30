"""External format converters: crowsetta, NWB, and other label I/O formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ethograph.labels.intervals import _rows_to_df, empty_intervals, load_mapping

CROWSETTA_SEQ_FORMATS = [
    "aud-seq",
    "simple-seq",
    "generic-seq",
    "notmat",
    "textgrid",
    "timit",
    "yarden",
]


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


def build_mapping_from_labels(string_labels: list[str]) -> dict[str, int]:
    """Build a name->id mapping from a list of unique string labels.

    Sorts labels alphabetically; 0 is reserved for 'background'.
    """
    unique = sorted(set(string_labels))
    mapping = {"background": 0}
    for i, name in enumerate(unique, start=1):
        mapping[name] = i
    return mapping


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
# NWB interval label converter
# ---------------------------------------------------------------------------

class NWBLabelConverter:
    name = "interval_labels"

    def __init__(self, include_sources: set[str] | None = None):
        self._epochs: list[dict] | None = None
        self._label_map: dict[str, int] = {}

    @property
    def label_map(self) -> dict[str, int]:
        return self._label_map

    def _load(self, nwb) -> None:
        self._epochs = self._extract_behavioral_epochs(nwb)
        self._label_map = build_mapping_from_labels(
            sorted({e["label_name"] for e in self._epochs})
        )

    def from_nwb(self, nwb, trials_df: pd.DataFrame) -> pd.DataFrame:
        """Extract labels from NWB and return a TSV-compatible all-labels DataFrame."""
        from ethograph.labels.tsv_store import TRIAL_META_DEFAULTS

        if self._epochs is None:
            self._load(nwb)

        individual = _get_nwb_individual(nwb)
        all_rows = []
        for _, row in trials_df.iterrows():
            t_start, t_stop = row["start_time"], row["stop_time"]
            trial_rows = [
                {
                    "onset_s": max(0.0, ep["onset_s"] - t_start),
                    "offset_s": min(t_stop - t_start, ep["offset_s"] - t_start),
                    "labels": self._label_map.get(ep["label_name"], 0),
                    "individual": ep.get("individual", individual),
                    "trial": row["trial"],
                }
                for ep in self._epochs
                if ep["offset_s"] > t_start
                and ep["onset_s"] < t_stop
                and self._label_map.get(ep["label_name"], 0) != 0
            ]
            all_rows.extend(trial_rows)

        if not all_rows:
            from ethograph.labels.tsv_store import init_empty_labels
            return init_empty_labels([])

        df = pd.DataFrame(all_rows)
        for col, default in TRIAL_META_DEFAULTS.items():
            df[col] = default
        return df

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


def _get_nwb_individual(nwb) -> str:
    subject = getattr(nwb, "subject", None)
    if subject:
        sid = getattr(subject, "subject_id", None)
        if sid:
            return str(sid)
    return "individual_0"
