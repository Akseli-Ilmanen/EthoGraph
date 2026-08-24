from collections.abc import Sequence
from typing import List, Literal

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks

from ethograph.features.preprocessing import z_normalize
from ethograph.io import schema
from ethograph.labels.intervals import (
    purge_short_intervals,
    snap_boundaries,
    stitch_intervals,
)
from ethograph.labels.ml import (
    find_blocks,
    fix_endings,
    purge_small_blocks,
    stitch_gaps,
)
from ethograph.utils.xr_utils import get_time_coord

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _nan_boundary_indices(arr: np.ndarray) -> np.ndarray:
    """Return indices where NaN transitions occur (NaN->valid and valid->NaN)."""
    arr = np.asarray(arr)
    is_valid = ~np.isnan(arr)
    transitions = np.diff(np.concatenate(([0], is_valid.astype(int), [0])))
    nan_to_val = np.where(transitions == 1)[0]
    val_to_nan = np.where(transitions == -1)[0] - 1
    return np.concatenate((nan_to_val, val_to_nan)).astype(int)


def _to_binary(indices: np.ndarray, length: int) -> np.ndarray:
    """Convert sparse index array to dense binary mask.

    If the only marked positions are the first and last sample, returns all zeros
    (boundary-only case treated as empty).
    """
    mask = np.zeros(length, dtype=np.int8)
    if len(indices) == 0:
        return mask
    valid = indices[(indices >= 0) & (indices < length)]
    mask[valid] = 1
    if np.sum(mask) == 2 and mask[0] == 1 and mask[-1] == 1:
        mask[:] = 0
    return mask


# ---------------------------------------------------------------------------
# NaN boundary handling (refactored to use helpers)
# ---------------------------------------------------------------------------


def add_NaN_boundaries(arr, changepoints):
    """Merge NaN-transition boundaries with other changepoints -> binary mask."""
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("add_NaN_boundaries only supports 1D (N,) input arrays.")
    nan_bounds = _nan_boundary_indices(arr)
    merged = np.unique(np.concatenate((np.asarray(changepoints, dtype=int), nan_bounds)))
    return _to_binary(merged, len(arr))


# ---------------------------------------------------------------------------
# Binary detection (dense binary masks for apply_ufunc / dataset storage)
# ---------------------------------------------------------------------------


def find_peaks_binary(x, **kwargs):
    """scipy.signal.find_peaks + NaN boundaries -> binary mask."""
    peaks, _ = find_peaks(np.asarray(x), **kwargs)
    return add_NaN_boundaries(x, peaks)


def find_troughs_binary(x, **kwargs):
    """Find troughs (local minima) + NaN boundaries -> binary mask."""
    troughs, _ = find_peaks(-np.asarray(x), **kwargs)
    return add_NaN_boundaries(x, troughs)


def find_nearest_turning_points_binary(x, threshold=1, max_value=None, prominence=0.5, distance=2, **kwargs):
    """Convert a 1D signal into a binary mask marking boundaries of peak regions.

    Identifies peaks in the signal, then finds the nearest "turning points"
    (where the gradient is near zero) on either side of each peak. These
    turning points define the boundaries of peak regions. The result is a
    binary mask where 1 indicates a turning-point boundary.

    The algorithm works in four steps:
        1. Compute the gradient of x and find indices where |gradient| < threshold,
           treating these as candidate turning points (near-stationary regions).
        2. Find peaks in x using scipy.signal.find_peaks with any additional kwargs.
        3. For each peak, select the closest turning point to its left and right.
        4. Add boundaries at NaN transitions in the original signal.

    Args:
        x: Input 1D signal.
        threshold: Maximum absolute gradient value to qualify as a turning point.
            Lower values select only very flat regions. Default is 1.
        max_value: If set, discard turning points where x exceeds this value.
            Useful for ignoring turning points on high plateaus.
        **kwargs: Passed to scipy.signal.find_peaks (e.g. height, distance,
            prominence).

    Returns:
        Binary array of same length as x, with 1 at turning-point boundaries
        and NaN-transition boundaries, 0 elsewhere.
    """
    x = np.asarray(x, dtype=float)
    grad = np.gradient(x)
    turning_points = np.where((grad > -threshold) & (grad < threshold))[0]

    if max_value is not None:
        turning_points = turning_points[x[turning_points] < max_value]

    peaks, _ = find_peaks(x, prominence, distance, **kwargs)
    turning_points = np.setdiff1d(turning_points, peaks)

    nearest = []
    for peak in peaks:
        left = turning_points[turning_points < peak]
        right = turning_points[turning_points > peak]
        if len(left) > 0:
            nearest.append(left[-1])
        if len(right) > 0:
            nearest.append(right[0])

    return add_NaN_boundaries(x, np.array(nearest, dtype=int))


# ---------------------------------------------------------------------------
# Changepoint time extraction
# ---------------------------------------------------------------------------


def extract_cp_times(ds: xr.Dataset, time: np.ndarray, **cp_kwargs) -> np.ndarray:
    """Extract merged changepoint times from dataset.

    Replaces the inline pattern: merge_changepoints -> binary -> np.where -> times.
    Returns empty array if no CP variables exist.
    """
    filtered = ds.sel(**cp_kwargs) if cp_kwargs else ds
    cp_ds = schema.filter_changepoints(filtered)
    if len(cp_ds.data_vars) == 0:
        return np.array([], dtype=np.float64)

    # TODO: Do I want merging of changepoints
    try:
        ds_merged, _ = merge_changepoints(filtered)
    except (ValueError, KeyError):
        return np.array([], dtype=np.float64)

    cp_binary = ds_merged["changepoints"].values
    cp_indices = np.where(cp_binary)[0]
    if len(cp_indices) == 0:
        return np.array([], dtype=np.float64)

    valid = cp_indices[cp_indices < len(time)]
    return time[valid].astype(np.float64)


def snap_to_nearest_changepoint_time(
    t_clicked: float,
    ds: xr.Dataset,
    feature_sel: str,
    time: np.ndarray,
    **ds_kwargs,
) -> float:
    """Snap a clicked time (seconds) to the nearest changepoint time.

    Works entirely in the time domain — no index conversion needed.
    Combines kinematic, audio, and oscillation changepoints.
    """
    all_cp_times = []

    # Kinematic changepoints (dense binary arrays)
    if feature_sel:
        filtered = ds.sel(**ds_kwargs) if ds_kwargs else ds
        cp_ds = schema.filter_changepoints(filtered).filter_by_attrs(target_feature=feature_sel)
        if len(cp_ds.data_vars) > 0:
            cp_indices = np.concatenate([np.where(cp_ds[var].values)[0] for var in cp_ds.data_vars])
            cp_indices = np.unique(cp_indices)
            valid = cp_indices[cp_indices < len(time)]
            if len(valid) > 0:
                all_cp_times.append(time[valid])

    # Audio changepoints (onset/offset pairs)
    if "audio_cp_onsets" in ds.data_vars and "audio_cp_offsets" in ds.data_vars:
        all_cp_times.append(ds["audio_cp_onsets"].values.astype(np.float64))
        all_cp_times.append(ds["audio_cp_offsets"].values.astype(np.float64))

    # Oscillation event changepoints
    if "osc_event_onsets" in ds.data_vars and "osc_event_offsets" in ds.data_vars:
        all_cp_times.append(ds["osc_event_onsets"].values.astype(np.float64))
        all_cp_times.append(ds["osc_event_offsets"].values.astype(np.float64))

    if not all_cp_times:
        return t_clicked

    cp_times = np.unique(np.concatenate(all_cp_times))
    if len(cp_times) == 0:
        return t_clicked

    nearest_idx = np.argmin(np.abs(cp_times - t_clicked))
    return float(cp_times[nearest_idx])


def correct_changepoints(
    df: pd.DataFrame,
    cp_times: np.ndarray,
    min_duration_s: float,
    stitch_gap_s: float,
    max_expansion_s: float,
    max_shrink_s: float,
    label_thresholds_s: dict[int, float] | None = None,
    do_purge: bool = True,
    do_stitch: bool = True,
    do_snap: bool = True,
    do_purge_after: bool = True,
) -> pd.DataFrame:
    """Full interval-native correction pipeline.

    Steps:
        1. purge_short_intervals — pre-cleanup (do_purge)
        2. stitch_intervals — merge same-label across small gaps (do_stitch)
        3. snap_boundaries — snap to changepoint times (do_snap)
        4. purge_short_intervals — post-cleanup (do_purge_after)
    """
    if df.empty:
        return df.copy()

    result = df
    if do_purge:
        result = purge_short_intervals(result, min_duration_s, label_thresholds_s)
    if do_stitch:
        result = stitch_intervals(result, stitch_gap_s)
    if do_snap:
        result = snap_boundaries(result, cp_times, max_expansion_s, max_shrink_s)
    if do_purge_after:
        result = purge_short_intervals(result, min_duration_s, label_thresholds_s)

    return result


def correct_changepoints_automatic(
    df: pd.DataFrame,
    min_duration_s: float = 1e-3,
    stitch_gap_s: float = 0.0,
) -> pd.DataFrame:
    """Lightweight cleanup used while manually creating labels."""
    if df.empty:
        return df.copy()

    result = purge_short_intervals(df, min_duration_s)
    return stitch_intervals(result, stitch_gap_s)


# ---------------------------------------------------------------------------
# Dense correction (legacy — kept for ML pipeline)
# ---------------------------------------------------------------------------


def correct_changepoints_dense(labels, ds, all_params):
    """Correct dense label arrays using changepoints (legacy ML pipeline).

    Operates on integer label arrays, not interval DataFrames.
    Use :func:`correct_changepoints` for the modern interval-native pipeline.

    Parameters
    ----------
    labels : array-like
        Dense integer label array of shape (T,).
    ds : xr.Dataset
        Trial dataset containing changepoint variables.
    all_params : dict
        Keys:

        - ``cp_kwargs``: Selection kwargs forwarded to ``ds.sel()``.
        - ``min_label_length_s``: Minimum label duration in seconds.
        - ``stitch_gap_len_s``: Maximum gap to stitch in seconds.
        - ``label_thresholds_s``: Per-label minimum durations (dict).
        - ``changepoint_params``: Dict with ``max_expansion_s`` and
          ``max_shrink_s``.
        - ``fps``: Frame rate used to convert seconds to sample counts.

    Returns
    -------
    np.ndarray
        Corrected integer label array of the same shape as ``labels``.
    """
    cp_kwargs = all_params["cp_kwargs"]

    # FIX to work with min_label_length_
    min_label_length = all_params.get("min_label_length_s")
    label_thresholds_s = all_params.get("label_thresholds_s", {})
    stitch_gap_len = all_params.get("stitch_gap_len_s")
    max_expansion = all_params["changepoint_params"]["max_expansion_s"]
    max_shrink = all_params["changepoint_params"]["max_shrink_s"]

    min_label_length = int(min_label_length * all_params["fps"])
    stitch_gap_len = int(stitch_gap_len * all_params["fps"])
    max_expansion = int(max_expansion * all_params["fps"])
    max_shrink = int(max_shrink * all_params["fps"])
    label_thresholds = {}
    for label, thresh in label_thresholds_s.items():
        label_thresholds[label] = int(thresh * all_params["fps"])

    ds = ds.sel(**cp_kwargs)
    ds_merged, _ = merge_changepoints(ds)
    changepoints_binary = ds_merged["changepoints"].values

    assert changepoints_binary.ndim == 1

    changepoint_idxs = np.where(changepoints_binary)[0]
    corrected_labels = np.zeros_like(labels, dtype=np.int8)

    labels = purge_small_blocks(labels, min_label_length, label_thresholds)
    labels = stitch_gaps(labels, stitch_gap_len)

    if len(changepoint_idxs) == 0:
        return labels

    for label in np.unique(labels):
        if label == 0:
            continue

        label_mask = labels == label
        starts, ends = find_blocks(label_mask)

        for block_start, block_end in zip(starts, ends):
            snap_start = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_start))]
            snap_end = changepoint_idxs[np.argmin(np.abs(changepoint_idxs - block_end))]

            start_expansion = block_start - snap_start
            start_shrink = snap_start - block_start

            if start_expansion > max_expansion or start_shrink > max_shrink:
                snap_start = block_start

            end_expansion = snap_end - block_end
            end_shrink = block_end - snap_end

            if end_expansion > max_expansion or end_shrink > max_shrink:
                snap_end = block_end

            if snap_start > snap_end:
                snap_start = block_start
                snap_end = block_end

            if snap_end < len(corrected_labels):
                if corrected_labels[snap_end] != 0 and corrected_labels[snap_end] != label:
                    snap_end = snap_end - 1

            corrected_labels[block_start : block_end + 1] = 0
            if snap_start < snap_end:
                corrected_labels[snap_start : snap_end + 1] = label

    corrected_labels = purge_small_blocks(corrected_labels, min_label_length)
    corrected_labels = fix_endings(corrected_labels, changepoints_binary)

    return corrected_labels


# ---------------------------------------------------------------------------
# Merge changepoints
# ---------------------------------------------------------------------------


def merge_changepoints(ds, vars: Sequence[str] | None = None, keep_dims: Sequence[str] | None = None):
    """Merge changepoint variables in a dataset into a single boolean mask.

    Combines every raw changepoint mask (``attrs["changepoint_mask"]``; see
    :func:`ethograph.io.schema.is_changepoint`) using logical OR across all
    non-time dimensions — the smooth expansions of a mask are ordinary
    features and are not merged. All masks must share the same
    ``target_feature`` attribute.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing one or more changepoint variables.
    vars : sequence of str, optional
        Which changepoint masks to merge; default merges every one
        :func:`~ethograph.io.schema.changepoint_vars` finds. Naming a
        variable that is not a changepoint mask is an error.
    keep_dims : sequence of str, optional
        Dims to leave standing instead of ORing across — typically the
        individual dim, so one animal's changepoints do not leak into
        another's. Default collapses every non-time dim.

    Returns
    -------
    ds : xr.Dataset
        Copy of the input with a new ``"changepoints"`` DataArray
        (float 0/1) replacing the merged changepoint variables.
    target_feature : str
        The shared ``target_feature`` attribute from the input variables.

    Raises
    ------
    ValueError
        If changepoint variables reference different target features.
    """
    ds = ds.copy()
    if vars is None:
        cp_ds = schema.filter_changepoints(ds)
    else:
        unknown = [v for v in vars if v not in ds.data_vars or not schema.is_changepoint(ds[v])]
        if unknown:
            raise ValueError(
                f"{unknown} are not changepoint masks in this dataset "
                f"(available: {schema.changepoint_vars(ds)})"
            )
        cp_ds = ds[list(vars)]

    target_feature = []
    for var in cp_ds.data_vars:
        target_feature.append(cp_ds[var].attrs["target_feature"])

    if np.unique(target_feature).size > 1:
        raise ValueError(
            f"Not allowed to merge changepoints for different target features: {np.unique(target_feature)}"
        )

    keep = set(keep_dims or ())
    dims = [dim for dim in cp_ds.dims if dim not in ["trials", "time"] and dim not in keep]

    ds["changepoints"] = cp_ds.to_array().any(dim=["variable"] + dims).astype(float)
    ds["changepoints"].attrs.update(schema.changepoint_attrs(target_feature=target_feature[0]))

    ds = ds.drop_vars(list(cp_ds.data_vars))

    return ds, target_feature[0]


# ---------------------------------------------------------------------------
# ML feature engineering
# ---------------------------------------------------------------------------


def more_changepoint_features(
    changepoint_binary: np.ndarray,
    targ_feat_vals: np.ndarray,
    sigmas: List[float],
    distribution: Literal["gaussian", "laplacian"] = "laplacian",
) -> np.ndarray:
    """Create changepoint-based features from a binary changepoint array.

    Generates three complementary representations that help a model learn
    changepoint locations:

    - **Binary changepoints** — exact changepoint positions (0/1 mask).
      Example: ``0 0 0 0 1 0 0 0 1 0 0 0 0 0``
    - **Smooth changepoints** — proximity to the nearest changepoint, rendered
      as Laplacian (or Gaussian) peaks centred at each changepoint index.
      Example: ``0 0 0 .3 1 .3 0 .3 1 .3 0 0 0 0``
    - **Segment IDs** — unique identifier for each contiguous region between
      changepoints (normalised to [0, 1]).
      Example: ``0 0 0 0 1 1 1 1 2 2 2 2 2 2``

    Smooth changepoints use a Laplacian kernel by default:

    .. math::

        \\text{smooth}(t) = \\sum_i \\exp\\!\\left(-\\frac{|t - i|}{\\sigma}\\right)

    where :math:`i` are the changepoint indices and :math:`\\sigma` controls
    the peak width. Laplacians have a narrow peak that points directly at the
    changepoint while their long tails remain visible from far away. Passing
    multiple ``sigmas`` (e.g. ``[0.5, 3, 5]``) yields features at several
    scales.

    Additionally, weighted versions emphasise changepoints where the target
    feature (e.g. speed) is low, by multiplying the smooth features with
    :math:`\\exp(-x / (\\bar{x} + \\epsilon))`. This helps the model
    distinguish speed minima before/after movements (low speed) from minima
    occurring within different movements.

    Args:
        changepoint_binary: Binary (0/1) array marking changepoint locations.
        targ_feat_vals: Target feature values (e.g. speed) used to weight the
            smooth changepoint features.
        sigmas: Kernel widths for the smooth changepoint representation.
        distribution: ``"laplacian"`` (default) or ``"gaussian"`` kernel.

    Returns:
        2D array of shape ``(T, 1 + 2 * len(sigmas) + 1)`` stacking the
        binary mask, one smooth feature per sigma, their weighted (speed-
        emphasised, z-normalised) counterparts, and the normalised segment
        IDs.
    """
    features = [changepoint_binary]
    seq_length = len(changepoint_binary)
    changepoint_indices = np.where(changepoint_binary)[0]

    x = np.arange(seq_length)
    for sigma in sigmas:
        peak = np.zeros(seq_length)
        for idx in changepoint_indices:
            if distribution == "gaussian":
                peak += np.exp(-0.5 * ((x - idx) / sigma) ** 2)
            else:
                peak += np.exp(-np.abs(x - idx) / sigma)

        if peak.max() > 0:
            peak /= peak.max()
        features.append(peak)

    cp_binary_peak = np.column_stack(features)

    multiplier = np.exp(-targ_feat_vals / (np.nanmean(targ_feat_vals) + 1e-8))
    weighted_cps = cp_binary_peak * multiplier[:, np.newaxis]
    weighted_cps = np.nan_to_num(weighted_cps, nan=0.0)
    weighted_cps = z_normalize(weighted_cps)

    segment_ids = np.zeros(seq_length)
    if len(changepoint_indices) > 0:
        boundaries = np.unique(np.concatenate([[0], changepoint_indices, [seq_length]]))
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            segment_ids[start:end] = i
        if segment_ids.max() > 0:
            segment_ids /= segment_ids.max()

    return np.column_stack([cp_binary_peak, weighted_cps, segment_ids])


#: The four column groups ``more_changepoint_features`` produces — the
#: vocabulary ``add_changepoint_features``'s ``transforms`` and
#: ``segment.config.ChangepointFeaturesConfig.transforms`` both read.
#: ``proximity``/``proximity_weighted`` name the *shape* generically because
#: which kernel they use is a separate choice (``distribution``).
CP_BINARY = "binary"
CP_PROXIMITY = "proximity"
CP_PROXIMITY_WEIGHTED = "proximity_weighted"
CP_SEGMENT_ID = "segment_id"
CP_TRANSFORMS: tuple[str, ...] = (CP_BINARY, CP_PROXIMITY, CP_PROXIMITY_WEIGHTED, CP_SEGMENT_ID)


def _cp_feature_groups(var: str, sigmas: Sequence[float]) -> list[tuple[str, str]]:
    """``(name, transform_group)`` pairs, in ``more_changepoint_features``'s column order."""
    smooth = [f"{var}_cp_sigma{sigma:g}" for sigma in sigmas]
    pairs = [(f"{var}_cp_binary", CP_BINARY)]
    pairs += [(name, CP_PROXIMITY) for name in smooth]
    pairs.append((f"{var}_cp_binary_weighted", CP_PROXIMITY_WEIGHTED))
    pairs += [(f"{name}_weighted", CP_PROXIMITY_WEIGHTED) for name in smooth]
    pairs.append((f"{var}_cp_segment_id", CP_SEGMENT_ID))
    return pairs


def cp_feature_names(var: str, sigmas: Sequence[float], transforms: Sequence[str] = CP_TRANSFORMS) -> list[str]:
    """Column names ``add_changepoint_features(..., transforms=transforms)`` writes for *var*.

    Use this to name the exact columns a ``changepoint_features`` config will
    produce without materialising anything — e.g. to paste into
    ``features.columns`` by hand, or to check a config's generated layout.
    """
    unknown = set(transforms) - set(CP_TRANSFORMS)
    if unknown:
        raise ValueError(f"transforms must be a subset of {CP_TRANSFORMS}, got {sorted(unknown)}")
    wanted = set(transforms)
    return [name for name, group in _cp_feature_groups(var, sigmas) if group in wanted]


def add_changepoint_features(
    ds: xr.Dataset,
    sigmas: Sequence[float],
    target_feature: str | None = None,
    distribution: Literal["gaussian", "laplacian"] = "laplacian",
    transforms: Sequence[str] = CP_TRANSFORMS,
    vars: Sequence[str] | None = None,
) -> xr.Dataset:
    """Expand changepoint variables into the ML features of ``more_changepoint_features``.

    For each changepoint data_var (see :func:`ethograph.io.schema.is_changepoint`; binary 0/1 over
    ``time`` plus any subset of ``keypoint``/``individual``) the requested
    *transforms* are computed per column and added as data_vars with the
    changepoint var's dims — see :data:`CP_TRANSFORMS` / :func:`cp_feature_names`
    for the exact names:

    - ``binary`` → ``{var}_cp_binary`` — the mask itself
    - ``proximity`` → ``{var}_cp_sigma{sigma}`` — one smooth proximity feature per sigma
    - ``proximity_weighted`` → ``{var}_cp_binary_weighted`` / ``{var}_cp_sigma{sigma}_weighted``
      — the same, weighted by low values of the target feature and z-normalised
    - ``segment_id`` → ``{var}_cp_segment_id`` — normalised id of the segment between changepoints

    All of them carry ``attrs["normalise"] = 0`` (already in ``[0, 1]`` or
    z-normalised) and none is marked as a changepoint variable — neither the
    ``kind`` nor the legacy ``type`` — so only the raw binary masks remain
    changepoints: they are the only ones that can be OR-merged, snapped to or
    drawn as lines.

    Parameters
    ----------
    ds : xr.Dataset
        Trial dataset holding changepoint vars and their target features.
    sigmas : sequence of float
        Kernel widths (samples) for the smooth features.
    target_feature : str, optional
        Feature whose values weight the smooth features; defaults to each
        changepoint var's own ``attrs["target_feature"]``.
    distribution : ``"laplacian"`` or ``"gaussian"``
        Kernel shape, see ``more_changepoint_features``.
    transforms : subset of :data:`CP_TRANSFORMS`
        Which column groups to keep; default is all four.
    vars : sequence of str, optional
        Which changepoint variables to expand; default expands every one
        :func:`~ethograph.io.schema.changepoint_vars` finds. Naming a
        variable that is not a changepoint mask is an error.

    Returns
    -------
    xr.Dataset
        Copy of ``ds`` with the new data_vars.
    """
    sigmas = list(sigmas)
    ds = ds.copy()
    cp_vars = schema.changepoint_vars(ds) if vars is None else list(vars)
    if not cp_vars:
        raise ValueError("Dataset has no changepoint data_var (kind='changepoint_feature' or type='changepoints')")
    for var in cp_vars:
        if var not in ds.data_vars or not schema.is_changepoint(ds[var]):
            raise ValueError(
                f"{var!r} is not a changepoint mask in this dataset "
                f"(available: {schema.changepoint_vars(ds)})"
            )

    for var in cp_vars:
        cp = ds[var]
        time = get_time_coord(cp).name
        target_name = target_feature if target_feature is not None else cp.attrs["target_feature"]
        target = ds[target_name]
        extra = [d for d in target.dims if d not in cp.dims]
        if extra:
            raise ValueError(
                f"Target feature {target_name!r} has dims {extra} that changepoint var {var!r} "
                f"({cp.dims}) lacks; pin them first"
            )
        if time not in target.dims:
            raise ValueError(f"Target feature {target_name!r} is not on the {time!r} axis of {var!r}")

        stacked = xr.apply_ufunc(
            lambda mask, vals: more_changepoint_features(mask, vals, sigmas, distribution),
            cp,
            target,
            input_core_dims=[[time], [time]],
            output_core_dims=[[time, "cp_feature"]],
            vectorize=True,
            output_dtypes=[np.float64],
        )
        all_pairs = _cp_feature_groups(var, sigmas)
        assert stacked.sizes["cp_feature"] == len(all_pairs)
        wanted = set(transforms)
        for i, (name, group) in enumerate(all_pairs):
            if group not in wanted:
                continue
            feature = stacked.isel(cp_feature=i, drop=True).transpose(*cp.dims)
            feature.attrs = {
                "description": f"Changepoint feature of {var} (target {target_name})",
                schema.KIND: schema.CHANGEPOINT_FEATURE,
                "normalise": 0,
            }
            ds[name] = feature

    return ds
