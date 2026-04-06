"""NWB metadata probing and trial table reading for the wizard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import pynwb
    from pynwb import NWBFile
except ImportError:
    pynwb = None
    NWBFile = None


# ---------------------------------------------------------------------------
# NWB metadata probing
# ---------------------------------------------------------------------------


def _has_valid_timing(iface: Any) -> bool:
    """Return True if the interface has either an explicit timestamps array or a rate."""
    if getattr(iface, "timestamps", None) is not None:
        return True
    return getattr(iface, "rate", None) is not None


def _get_absolute_timestamps(iface: Any) -> np.ndarray:
    """Return timestamps in absolute session time for any NWB TimeSeries.

    NWB supports two timing schemes:
    - ``timestamps``: explicit array already in absolute session time.
    - ``rate`` + ``starting_time``: regularly sampled; absolute times are
      ``starting_time + arange(n) / rate``.  ``starting_time`` defaults to 0
      when absent (i.e. recording starts at session time 0).
    """
    if getattr(iface, "timestamps", None) is not None:
        return np.asarray(iface.timestamps[:], dtype=np.float64)
    n = iface.data.shape[0] if hasattr(iface.data, "shape") else len(iface.data)
    t0 = float(iface.starting_time) if getattr(iface, "starting_time", None) is not None else 0.0
    return t0 + np.arange(n, dtype=np.float64) / float(iface.rate)


def probe_behavioral_series(nwb: NWBFile) -> list[dict]:
    """List all behavioral time-series interfaces available for import."""
    SKIP_MODULES = {"ecephys", "ophys", "ogen"}
    results = []
    for mod_name, mod in nwb.processing.items():
        if mod_name in SKIP_MODULES:
            continue
        for iface_name, iface in mod.data_interfaces.items():
            if hasattr(iface, "pose_estimation_series"):
                continue
            if not hasattr(iface, "data"):
                continue
            if not _has_valid_timing(iface):
                continue
            try:
                n = iface.data.shape[0] if hasattr(iface.data, "shape") else len(iface.data)
            except Exception:
                n = 0
            results.append({
                "source": f"{mod_name}/{iface_name}",
                "module": mod_name,
                "interface": iface_name,
                "n_samples": n,
            })
    return results


def probe_electrical_series(nwb: NWBFile) -> list[dict]:
    """List all ElectricalSeries in nwb.acquisition."""
    results = []
    for name, obj in nwb.acquisition.items():
        if not isinstance(obj, pynwb.ecephys.ElectricalSeries):
            continue
        n_samples = obj.data.shape[0] if hasattr(obj.data, "shape") else len(obj.data)
        n_channels = obj.data.shape[1] if hasattr(obj.data, "shape") and obj.data.ndim > 1 else 1
        rate = float(obj.rate) if obj.rate else None
        results.append({
            "name": name,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "rate": rate,
        })
    return results


def probe_label_sources(nwb: NWBFile) -> list[dict]:
    """List all potential interval label sources in the NWB file."""
    results = []

    if nwb.epochs is not None:
        try:
            n = len(nwb.epochs)
        except Exception:
            n = 0
        results.append({"source": "epochs", "description": f"nwb.epochs ({n} rows)"})

    for mod_name, mod in nwb.processing.items():
        for iface_name, iface in mod.data_interfaces.items():
            if isinstance(iface, pynwb.epoch.TimeIntervals):
                try:
                    n = len(iface)
                except Exception:
                    n = 0
                results.append({
                    "source": f"{mod_name}/{iface_name}",
                    "description": f"TimeIntervals: {mod_name}/{iface_name} ({n} rows)",
                })
            elif isinstance(iface, pynwb.behavior.BehavioralEpochs):
                for series_name in iface.interval_series:
                    results.append({
                        "source": f"{mod_name}/{iface_name}/{series_name}",
                        "description": f"IntervalSeries: {mod_name}/{iface_name}/{series_name}",
                    })
    return results


# ---------------------------------------------------------------------------
# Trials table reader
# ---------------------------------------------------------------------------


def read_trials_table(nwb: NWBFile) -> pd.DataFrame:
    if nwb.trials is None or len(nwb.trials) == 0:
        duration = _get_max_duration(nwb)
        return pd.DataFrame([{"trial": 1, "start_time": 0.0, "stop_time": duration}])

    df = nwb.trials.to_dataframe()

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(_resolve)

    if "trial" not in df.columns:
        df = df.reset_index(drop=True)
        df["trial"] = df.index + 1
    return df


def _resolve(val):
    if hasattr(val, 'data'):  # h5py / NWB lazy wrapper
        val = val.data
    if hasattr(val, '__array__'):
        val = val.item() if val.ndim == 0 else val.tolist()
    return val


def _get_max_duration(nwb: NWBFile) -> float:
    durations = []
    for ts in nwb.acquisition.values():
        if isinstance(ts, pynwb.TimeSeries):
            dur = _ts_duration(ts)
            if dur is not None:
                durations.append(dur)
    return max(durations) if durations else 1.0


def _ts_duration(ts: Any) -> float | None:
    n = ts.data.shape[0] if hasattr(ts.data, "shape") else len(ts.data)
    if ts.timestamps is not None and len(ts.timestamps) > 0:
        return float(ts.timestamps[-1])
    if ts.rate and ts.rate > 0:
        start = float(ts.starting_time) if ts.starting_time else 0.0
        return start + n / float(ts.rate)
    return None

