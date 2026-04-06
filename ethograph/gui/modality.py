"""Unified data source and buffering abstractions for all modalities.

Replaces the per-plot buffer classes (AudioTraceBuffer, SpectrogramBuffer,
EphysTraceBuffer, LinePlot inline buffer) with a single generic
``WindowedBuffer`` that wraps any ``ModalitySource``.

Key types:
    SourceData      -- (timestamps, values) pair returned by file-based sources
    ModalitySource  -- Protocol: anything that provides time-aligned data
    FileSource      -- Wraps audioio/ephys/memmap loaders as ModalitySource
    XarraySource    -- Wraps xr.Dataset as a ModalitySource (returns dataset slices)
    WindowedBuffer  -- Generic viewport-aware cache for any source
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ethograph.io.time_model import TimeRange

if TYPE_CHECKING:
    import xarray as xr


@dataclass(slots=True)
class SourceData:
    """Raw data slice returned by a file-based ModalitySource."""

    timestamps: np.ndarray
    values: np.ndarray

    def __len__(self) -> int:
        return len(self.timestamps)

    def empty(self) -> bool:
        return len(self.timestamps) == 0

    def slice_time(self, t0: float, t1: float) -> SourceData:
        mask = (self.timestamps >= t0) & (self.timestamps <= t1)
        return SourceData(self.timestamps[mask], self.values[mask])


@runtime_checkable
class ModalitySource(Protocol):
    """Uniform interface for any time-aligned data provider.

    Implementations: FileSource (audio, ephys), XarraySource (features),
    VideoSource (frame-based).
    """

    @property
    def name(self) -> str: ...

    @property
    def time_range(self) -> TimeRange: ...

    @property
    def sampling_rate(self) -> float: ...

    @property
    def identity(self) -> str: ...

    def get_data(self, t0: float, t1: float): ...


class FileSource:
    """Wraps a file-based loader (audioio, ephys, memmap) as a ModalitySource.

    The loader must expose ``rate: float``, ``__len__() -> int``,
    and ``__getitem__(slice) -> ndarray``.
    """

    def __init__(
        self,
        name: str,
        loader,
        *,
        start_time: float = 0.0,
        channel: int | None = None,
    ):
        self._name = name
        self._loader = loader
        self._start_time = start_time
        self._channel = channel
        self._rate = float(loader.rate)
        self._n_samples = len(loader)

        if channel is not None:
            self._n_channels = 1
        elif hasattr(loader, "n_channels"):
            self._n_channels = loader.n_channels
        else:
            probe = loader[0 : min(2, self._n_samples)]
            self._n_channels = probe.shape[1] if probe.ndim > 1 else 1

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_range(self) -> TimeRange:
        end = self._start_time + self._n_samples / self._rate
        return TimeRange(self._start_time, end)

    @property
    def sampling_rate(self) -> float:
        return self._rate

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def identity(self) -> str:
        loader_id = getattr(self._loader, "filepath", id(self._loader))
        ch = f":{self._channel}" if self._channel is not None else ""
        return f"file:{self._name}:{loader_id}{ch}"

    def get_data(self, t0: float, t1: float) -> SourceData:
        i0 = max(0, int((t0 - self._start_time) * self._rate))
        i1 = min(self._n_samples, int((t1 - self._start_time) * self._rate) + 1)
        if i1 <= i0:
            empty = np.array([], dtype=np.float64)
            return SourceData(empty, empty)

        data = self._loader[i0:i1]
        if self._channel is not None and data.ndim > 1:
            ch = min(self._channel, data.shape[1] - 1)
            data = data[:, ch]

        timestamps = self._start_time + np.arange(i0, i1) / self._rate
        return SourceData(
            timestamps.astype(np.float64),
            np.asarray(data, dtype=np.float64),
        )


class XarraySource:
    """Wraps an xr.Dataset as a ModalitySource.

    Returns time-sliced datasets from ``get_data()``.  Only variables
    sharing the given time coordinate are included.
    """

    def __init__(self, ds: xr.Dataset, time_coord_name: str):
        import xarray as xr  # noqa: F811 — runtime import for non-TYPE_CHECKING

        self._time_coord_name = time_coord_name
        time_vars = [v for v in ds.data_vars if time_coord_name in ds[v].dims]
        self._ds = ds[time_vars] if time_vars else ds
        tc = ds.coords[time_coord_name].values
        self._time_range = TimeRange(float(tc[0]), float(tc[-1]))
        dt = float(tc[1] - tc[0]) if len(tc) > 1 else 1.0
        self._sampling_rate = 1.0 / dt if dt > 0 else 1.0
        self._identity = f"xarray:{id(ds)}:{time_coord_name}"

    @property
    def name(self) -> str:
        return "features"

    @property
    def time_range(self) -> TimeRange:
        return self._time_range

    @property
    def sampling_rate(self) -> float:
        return self._sampling_rate

    @property
    def identity(self) -> str:
        return self._identity

    def get_data(self, t0: float, t1: float) -> xr.Dataset:
        return self._ds.sel({self._time_coord_name: slice(t0, t1)})


class WindowedBuffer:
    """Viewport-aware cache for any ModalitySource.

    Maintains a data window around the current viewport, reloading only
    when the viewport moves outside the buffered region.  Replaces all
    per-plot buffer classes with a single generic implementation.

    The cached object type depends on the source: ``SourceData`` for
    ``FileSource``, ``xr.Dataset`` for ``XarraySource``, etc.

    Parameters
    ----------
    buffer_multiplier
        How much wider than the viewport to load (e.g. 5.0 = load 5x
        the viewport width, centered on the current view).
    coverage_margin
        Fraction of viewport width used as hysteresis margin to avoid
        reloading on tiny pans.
    """

    def __init__(
        self,
        buffer_multiplier: float = 5.0,
        coverage_margin: float = 0.2,
    ):
        self.source: ModalitySource | None = None
        self.buffer_multiplier = buffer_multiplier
        self._coverage_margin = coverage_margin
        self._bounds: TimeRange | None = None

        self._cache: object | None = None
        self._cache_t0 = 0.0
        self._cache_t1 = 0.0
        self._identity: str | None = None

    def set_source(
        self,
        source: ModalitySource | None,
        bounds: TimeRange | None = None,
    ):
        """Attach a new source, invalidating the cache if it changed."""
        if source is None:
            self.source = None
            self._invalidate()
            return

        new_id = source.identity
        if new_id != self._identity:
            self._invalidate()
            self.source = source
            self._identity = new_id

        self._bounds = bounds or source.time_range

    def get(self, t0: float, t1: float):
        """Return cached data covering [t0, t1], loading if necessary."""
        if self.source is None:
            return None
        if not self._covers(t0, t1):
            self._fill(t0, t1)
        return self._cache

    def invalidate(self):
        """Public cache invalidation (e.g. on trial change)."""
        self._invalidate()

    def _covers(self, t0: float, t1: float) -> bool:
        if self._cache is None:
            return False
        margin = (t1 - t0) * self._coverage_margin
        return self._cache_t0 <= t0 - margin and self._cache_t1 >= t1 + margin

    def _fill(self, t0: float, t1: float):
        window = t1 - t0
        half_buf = window * self.buffer_multiplier / 2
        bounds = self._bounds or self.source.time_range
        load_t0 = max(bounds.start_s, t0 - half_buf)
        load_t1 = min(bounds.end_s, t1 + half_buf)
        if load_t1 <= load_t0:
            self._cache = None
            return
        self._cache = self.source.get_data(load_t0, load_t1)
        self._cache_t0 = load_t0
        self._cache_t1 = load_t1

    def _invalidate(self):
        self._cache = None
        self._cache_t0 = 0.0
        self._cache_t1 = 0.0
        self._identity = None
