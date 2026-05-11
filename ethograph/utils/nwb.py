"""NWB session creation, alignment helpers, timing utilities, and H5 protocols.

Also re-exports names that moved to ``utils.dandi`` and ``io.nwb_import``
for backwards compatibility.

Public names (H5 layer)
-----------------------
H5Like, H5Group, H5Dataset, NWBBackend, open_nwb
"""

from __future__ import annotations

import enum
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

try:
    import pynwb
    from pynwb import NWBFile
except ImportError:
    pynwb = None
    NWBFile = None


# ---------------------------------------------------------------------------
# Protocols: anything that looks like an h5py.File / Group / Dataset
# ---------------------------------------------------------------------------


@runtime_checkable
class H5Like(Protocol):
    def visititems(self, func: Any) -> None: ...
    def __getitem__(self, key: str) -> Any: ...


@runtime_checkable
class H5Group(Protocol):
    attrs: Any

    def get(self, key: str) -> Any: ...
    def keys(self) -> Any: ...


@runtime_checkable
class H5Dataset(Protocol):
    shape: tuple[int, ...]
    dtype: Any

    def __getitem__(self, key: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------


class NWBBackend(enum.Enum):
    LOCAL = "local"
    REMFILE = "remfile"
    LINDI = "lindi"


# ---------------------------------------------------------------------------
# File opener: local, remfile, lindi → h5py-like handle
# ---------------------------------------------------------------------------


@contextmanager
def open_nwb(
    source: str | Path | H5Like,
    backend: NWBBackend | None = None,
) -> Iterator[H5Like]:
    if isinstance(source, H5Like):
        yield source
        return

    source = str(source)
    resolved_backend = backend or _infer_backend(source)
    closables: list[Any] = []

    try:
        if resolved_backend == NWBBackend.LINDI:
            import lindi

            if source.endswith((".lindi.json", ".lindi.tar", ".lindi.d")):
                f = lindi.LindiH5pyFile.from_lindi_file(source)
            else:
                f = lindi.LindiH5pyFile.from_hdf5_file(source)
            closables.append(f)
            yield f

        elif resolved_backend == NWBBackend.REMFILE:
            import h5py
            import remfile

            rem = remfile.File(source)
            closables.append(rem)
            h5 = h5py.File(rem, "r")
            closables.append(h5)
            yield h5

        else:
            import h5py

            h5 = h5py.File(source, "r")
            closables.append(h5)
            yield h5

    finally:
        for obj in reversed(closables):
            try:
                obj.close()
            except Exception:
                pass


def _infer_backend(source: str) -> NWBBackend:
    if any(source.endswith(ext) for ext in (".lindi.json", ".lindi.tar", ".lindi.d")):
        return NWBBackend.LINDI
    if source.startswith(("http://", "https://", "s3://")):
        return NWBBackend.LINDI
    return NWBBackend.LOCAL


# ---------------------------------------------------------------------------
# Timing helpers (used by nwb_alignment, plots_ephystrace, and creation code)
# ---------------------------------------------------------------------------


def resolve_timeseries_timing(iface: Any) -> tuple[float, float]:
    """Extract (rate_hz, starting_time_s) from any NWB TimeSeries.

    Handles both NWB timing schemes:
    - ``rate`` + ``starting_time``: returns them directly.
    - ``timestamps``: derives rate from median inter-sample interval,
      starting_time from ``timestamps[0]``.

    Raises ``ValueError`` if neither scheme is available.
    """
    if getattr(iface, "rate", None) is not None and iface.rate:
        t0 = float(iface.starting_time) if getattr(iface, "starting_time", None) is not None else 0.0
        return float(iface.rate), t0
    ts = getattr(iface, "timestamps", None)
    if ts is not None and len(ts) >= 2:
        ts_arr = np.asarray(ts[: min(len(ts), 10_000)], dtype=np.float64)
        diffs = np.diff(ts_arr)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            rate = 1.0 / float(np.median(diffs))
            return rate, float(ts_arr[0])
    raise ValueError(f"TimeSeries '{getattr(iface, 'name', '?')}' has neither rate nor timestamps.")
