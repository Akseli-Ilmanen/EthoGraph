from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    pass


def sel_valid(da, sel_kwargs):
    """Select a slice of a DataArray, silently ignoring dimensions it doesn't have.

    Useful when the GUI holds a single ``sel_kwargs`` dict (e.g.
    ``{"keypoints": "nose", "space": "x"}``) but different features have
    different subsets of those dimensions.  Dimensions with labelled
    coordinates are selected with ``.sel()``; dimensions without coordinates
    fall back to ``.isel()``.  The result is squeezed and transposed so time
    is always the first axis.

    Parameters
    ----------
    da : xarray.DataArray
        Source array.  Must contain at least one dimension whose name
        includes ``"time"``.
    sel_kwargs : dict
        Candidate selections.  Keys that don't match any dimension in *da*
        are silently dropped.

    Returns
    -------
    data : numpy.ndarray
        Selected values with shape ``(n_time,)`` or ``(n_time, n_other)``.
    used_kwargs : dict
        The subset of *sel_kwargs* that were actually applied via ``.sel()``
        (i.e. only label-based selections, not integer-based ones).  Handy
        for building plot titles.

    Raises
    ------
    ValueError
        If *da* has no dimension containing ``"time"``.

    Examples
    --------
    >>> import xarray as xr, numpy as np
    >>> da = xr.DataArray(
    ...     np.random.randn(100, 3),
    ...     dims=["time", "space"],
    ...     coords={"time": np.linspace(0, 10, 100), "space": ["x", "y", "z"]},
    ... )
    >>> data, used = eto.sel_valid(da, {"space": "x", "individuals": "mouse1"})
    >>> data.shape
    (100,)
    >>> used
    {'space': 'x'}
    """

    valid_keys = set(da.dims)
    coord_keys = set(da.coords.keys())

    sel_kwargs_filtered = {}
    isel_kwargs = {}

    for k, v in sel_kwargs.items():
        if k not in valid_keys:
            continue
        if k in coord_keys:
            sel_kwargs_filtered[k] = v
        else:
            isel_kwargs[k] = int(v) if isinstance(v, str) else v

    # Only return sel-compatible kwargs (those with coordinates)
    # isel kwargs are applied but not returned since .sel() can't use them
    filt_kwargs = dict(sel_kwargs_filtered)

    if sel_kwargs_filtered:
        da = da.sel(**sel_kwargs_filtered)
    if isel_kwargs:
        da = da.isel(**isel_kwargs)
    da = da.squeeze()

    time_dim = next((dim for dim in da.dims if "time" in dim), None)

    if time_dim is None:
        raise ValueError("No dimension containing 'time' found in the DataArray.")

    da = da.transpose(time_dim, ...)

    data = da.values
    assert data.ndim in [1, 2]  # either (time,) or (time, space)/ (time, RGB), ...

    return data, filt_kwargs


def get_time_coord(da: xr.DataArray) -> xr.DataArray | None:
    """Return the time coordinate of a DataArray, regardless of its name.

    Every feature variable in an ethograph dataset **must** have at least
    one dimension whose name contains ``"time"`` (e.g. ``time``,
    ``time_aux``, ``time_labels``).  Different features can use different
    time dimensions at different sampling rates — this function finds the
    right one for a given DataArray.
    See :ref:`target-data-requirements` for the full specification.

    **Lookup order:** dimension coordinates (``da.dims``) are checked
    before non-dimension coordinates (``da.coords``), so the primary time
    axis that actually indexes the data is returned even when a shorter
    auxiliary coordinate is also attached.

    Parameters
    ----------
    da : xarray.DataArray
        Any DataArray from an ethograph dataset.

    Returns
    -------
    xarray.DataArray or None
        The time coordinate values, or ``None`` if no coordinate name
        contains ``"time"``.

    Examples
    --------
    >>> import ethograph as eto
    >>> dt = eto.open("experiment.nc")
    >>> ds = dt.itrial(0)

    Feature with the default ``time`` dimension:

    >>> eto.get_time_coord(ds["speed"])
    <xarray.DataArray 'time' (time: 9000)> ...

    Audio stored on a higher-rate ``time_aux`` axis:

    >>> eto.get_time_coord(ds["audio_waveform"])
    <xarray.DataArray 'time_aux' (time_aux: 441000)> ...

    Used internally by :func:`add_changepoints_to_ds
    <ethograph.utils.io.add_changepoints_to_ds>` to discover which time
    dimension to vectorise over.
    """
    time_dims = [d for d in da.dims if "time" in d.lower()]
    if time_dims:
        return da.coords[time_dims[0]]
    time_coord = next((c for c in da.coords if "time" in c.lower()), None)
    if time_coord is None:
        return None
    return da.coords[time_coord]


def get_ds_duration(ds: xr.Dataset) -> float | None:
    """Get duration of xarray.Dataset in seconds."""
    durations = []
    for var in ds.data_vars.values():
        time_coord = get_time_coord(var)
        if time_coord is not None:
            durations.append(float(time_coord.max() - time_coord.min()))
    return max(durations) if durations else None
