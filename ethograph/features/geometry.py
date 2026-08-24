"""Geometric features derived from pose data (movement convention).

Every function takes a ``movement``-style DataArray — ``position`` / ``velocity``
with dims ``(time, space, keypoint, individual)`` — and returns a DataArray
that is a plain feature: the GUI can plot it and a segmentation model can
select it by name. All functions work for 2-D (``x, y``) and 3-D (``x, y, z``)
space, propagate NaN, and keep the ``individual`` dim even when it has length 1.

Every output is stamped with the variable schema of :mod:`ethograph.io.schema`:
``kind="kinematic_feature"`` plus ``is_egocentric``. Outputs that are unit
vectors or angles also carry ``attrs["normalise"] = 0`` so a later
preprocessing step knows not to z-score them.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import xarray as xr
from movement.kinematics import compute_forward_vector_angle
from movement.utils.vector import compute_signed_angle_2d

from ethograph.io import schema

SPACE_DIM = "space"
KEYPOINT_DIM = "keypoint"
INDIVIDUAL_DIM = "individual"
PAIR_DIM = "pair"
OTHER_DIM = "other"
ANGLE_DIM = "angle"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _time_dim(da: xr.DataArray) -> str:
    """Name of the dim that indexes time (any dim whose name contains ``time``)."""
    for dim in da.dims:
        if "time" in str(dim).lower():
            return str(dim)
    raise ValueError(f"No time dimension found in dims {da.dims}")


def _canonical(da: xr.DataArray) -> tuple[xr.DataArray, str]:
    """Return ``da`` transposed to ``(time, space, keypoint, individual)`` plus the time dim name."""
    time = _time_dim(da)
    required = [time, SPACE_DIM, KEYPOINT_DIM, INDIVIDUAL_DIM]
    missing = [d for d in required if d not in da.dims]
    if missing:
        raise ValueError(f"Expected dims {required}, missing {missing} in {da.dims}")
    extra = [d for d in da.dims if d not in required]
    if extra:
        raise ValueError(f"Unexpected extra dims {extra} in {da.dims}")
    space = [str(s) for s in da[SPACE_DIM].values]
    if space[:2] != ["x", "y"] or len(space) not in (2, 3):
        raise ValueError(f"space coord must be ['x', 'y'] or ['x', 'y', 'z'], got {space}")
    return da.transpose(*required), time


#: Sentinel for "leave the source's ``units`` attr alone" — ``None`` means drop it.
_KEEP_UNITS = "<keep>"


def _feature(
    da: xr.DataArray,
    source: xr.DataArray,
    description: str,
    *,
    is_egocentric: bool = False,
    normalise: bool = True,
    units: str | None = _KEEP_UNITS,
) -> xr.DataArray:
    """Carry ``source``'s attrs onto ``da`` and stamp the schema — the one place this module describes an output.

    ``units`` defaults to the source's; ``None`` drops it (the output is not in
    the input's units), any other string replaces it. The schema attrs are the
    output's own, never inherited.
    """
    attrs = schema.clear(dict(source.attrs))
    attrs["description"] = description
    if units != _KEEP_UNITS:
        attrs.pop("units", None)
        if units is not None:
            attrs["units"] = units
    da.attrs = attrs
    return schema.describe(da, schema.KINEMATIC_FEATURE, is_egocentric=is_egocentric, normalise=normalise)


def _keypoint_index(da: xr.DataArray, name: str) -> int:
    names = [str(k) for k in da[KEYPOINT_DIM].values]
    if name not in names:
        raise ValueError(f"Keypoint {name!r} not in {names}")
    return names.index(name)


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` with NaN wherever ``den`` is 0 or NaN (no warnings)."""
    out = np.full(np.broadcast(num, den).shape, np.nan)
    ok = np.isfinite(den) & (den != 0)
    return np.divide(num, den, out=out, where=ok)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def egocentric_position(
    position: xr.DataArray,
    centre_keypoint: str | None = None,
    *,
    centre_on_centroid: bool = False,
    heading_keypoint: str | None = None,
    left_keypoint: str | None = None,
    right_keypoint: str | None = None,
) -> xr.DataArray:
    """Position in an egocentric frame: centred on a keypoint (or the centroid), heading along +x.

    The frame is translated so the centre sits at the origin and rotated in the
    x-y plane so the heading direction points along +x. In 3-D the z coordinate
    is only translated.

    The centre is either one named keypoint (``centre_keypoint``) or the mean
    of all keypoints (``centre_on_centroid=True``); exactly one must be given.

    Heading comes from either a body-axis pair (``heading_keypoint``, e.g. a
    beak tip: the vector centre → heading) or a symmetric left-right pair
    (``left_keypoint``/``right_keypoint``, e.g. two ears: the vector
    perpendicular to the line between them, assuming a top-down view).
    Exactly one of the two must be given.

    Parameters
    ----------
    position : xarray.DataArray
        Dims ``(time, space, keypoint, individual)``.
    centre_keypoint : str, optional
        Keypoint name defining the origin. Mutually exclusive with
        ``centre_on_centroid``.
    centre_on_centroid : bool, optional
        Use the mean of all keypoints as the origin instead of one named
        keypoint. Mutually exclusive with ``centre_keypoint``.
    heading_keypoint : str, optional
        Keypoint whose direction from the centre defines +x.
        Mutually exclusive with ``left_keypoint``/``right_keypoint``.
    left_keypoint, right_keypoint : str, optional
        Symmetric keypoint pair; heading is the vector perpendicular to the
        line between them. Mutually exclusive with ``heading_keypoint``.

    Returns
    -------
    xarray.DataArray
        Dims ``(time, space, keypoint, individual)`` in the egocentric frame.
    """
    pos, _ = _canonical(position)
    if (centre_keypoint is not None) == centre_on_centroid:
        raise ValueError("Provide either centre_keypoint or centre_on_centroid=True")
    if centre_on_centroid:
        centre = pos.mean(dim=KEYPOINT_DIM, skipna=False)
        centre_label = "centroid"
    else:
        centre = pos.sel(keypoint=centre_keypoint)
        centre_label = centre_keypoint

    centred = pos - centre
    xy = pos.sel(space=["x", "y"])
    xy_centre = centre.sel(space=["x", "y"])
    reference = np.array([1.0, 0.0])

    if heading_keypoint is not None:
        direction = (xy - xy_centre).sel(keypoint=heading_keypoint, drop=True)
        theta = compute_signed_angle_2d(direction, reference)
        description = f"heading is {centre_label}->{heading_keypoint}"
    elif left_keypoint is not None and right_keypoint is not None:
        # compute_forward_vector_angle returns the forward vector's own angle relative to `reference`
        # (v_as_left_operand=True internally); negate for the rotation that brings it onto `reference`.
        theta = -compute_forward_vector_angle(xy, left_keypoint, right_keypoint, reference_vector=reference)
        description = f"heading is perpendicular to {left_keypoint}-{right_keypoint}"
    else:
        raise ValueError("Provide either heading_keypoint or both left_keypoint and right_keypoint")

    cos, sin = np.cos(theta), np.sin(theta)
    x, y = centred.sel(space="x"), centred.sel(space="y")
    rotated = xr.concat([x * cos - y * sin, x * sin + y * cos], dim=SPACE_DIM)
    if "z" in pos[SPACE_DIM].values:
        rotated = xr.concat([rotated, centred.sel(space="z", drop=True)], dim=SPACE_DIM)
    rotated = rotated.assign_coords({SPACE_DIM: pos[SPACE_DIM]}).transpose(*pos.dims)

    return _feature(
        rotated.rename("egocentric_position"),
        position,
        f"Position centred on {centre_label}, rotated so {description} is +x",
        is_egocentric=True,
    )


def intra_distances(position: xr.DataArray, keypoints: Sequence[str] | None = None) -> xr.DataArray:
    """Euclidean distance between every unordered pair of keypoints.

    Parameters
    ----------
    position : xarray.DataArray
        Dims ``(time, space, keypoint, individual)``.
    keypoints : sequence of str, optional
        Keypoints to pair up; defaults to all, in coordinate order.

    Returns
    -------
    xarray.DataArray
        Dims ``(time, pair, individual)``; ``pair`` coords are ``"a-b"`` strings.
    """
    pos, time = _canonical(position)
    names = [str(k) for k in pos[KEYPOINT_DIM].values] if keypoints is None else [str(k) for k in keypoints]
    if len(names) < 2:
        raise ValueError(f"Need at least two keypoints, got {names}")
    arr = pos.values.astype(np.float64)
    pairs = list(combinations(names, 2))
    stack = np.empty((arr.shape[0], len(pairs), arr.shape[3]))
    for p, (a, b) in enumerate(pairs):
        diff = arr[:, :, _keypoint_index(pos, a), :] - arr[:, :, _keypoint_index(pos, b), :]
        stack[:, p, :] = np.linalg.norm(diff, axis=1)

    return _feature(
        xr.DataArray(
            stack,
            dims=(time, PAIR_DIM, INDIVIDUAL_DIM),
            coords={
                time: pos[time],
                PAIR_DIM: [f"{a}-{b}" for a, b in pairs],
                INDIVIDUAL_DIM: pos[INDIVIDUAL_DIM],
            },
            name="intra_distances",
        ),
        position,
        "Euclidean distance between keypoint pairs of one individual",
    )


def inter_distances(position: xr.DataArray, keypoint: str) -> xr.DataArray:
    """Distance between one keypoint of each individual and that keypoint of every other.

    Parameters
    ----------
    position : xarray.DataArray
        Dims ``(time, space, keypoint, individual)`` with at least two individuals.
    keypoint : str
        Keypoint compared across individuals.

    Returns
    -------
    xarray.DataArray
        Dims ``(time, individual, other)``; ``other`` shares the ``individual``
        coord values and the diagonal (self) is NaN.
    """
    pos, time = _canonical(position)
    if pos.sizes[INDIVIDUAL_DIM] < 2:
        raise ValueError("inter_distances needs at least two individuals")
    arr = pos.values[:, :, _keypoint_index(pos, keypoint), :].astype(np.float64)
    diff = arr[:, :, :, None] - arr[:, :, None, :]
    dist = np.linalg.norm(diff, axis=1)
    idx = np.arange(arr.shape[2])
    dist[:, idx, idx] = np.nan

    return _feature(
        xr.DataArray(
            dist,
            dims=(time, INDIVIDUAL_DIM, OTHER_DIM),
            coords={
                time: pos[time],
                INDIVIDUAL_DIM: pos[INDIVIDUAL_DIM],
                OTHER_DIM: pos[INDIVIDUAL_DIM].values,
            },
            name="inter_distances",
        ),
        position,
        f"Distance between {keypoint} of each individual and {keypoint} of every other",
    )


def heading(position: xr.DataArray, from_keypoint: str, to_keypoint: str) -> xr.DataArray:
    """Unit vector pointing from one keypoint to another.

    Parameters
    ----------
    position : xarray.DataArray
        Dims ``(time, space, keypoint, individual)``.
    from_keypoint, to_keypoint : str
        Tail and head of the vector.

    Returns
    -------
    xarray.DataArray
        Dims ``(time, space, individual)``; NaN where the two keypoints coincide.
    """
    pos, time = _canonical(position)
    arr = pos.values.astype(np.float64)
    vec = arr[:, :, _keypoint_index(pos, to_keypoint), :] - arr[:, :, _keypoint_index(pos, from_keypoint), :]
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    unit = _safe_divide(vec, norm)

    return _feature(
        xr.DataArray(
            unit,
            dims=(time, SPACE_DIM, INDIVIDUAL_DIM),
            coords={time: pos[time], SPACE_DIM: pos[SPACE_DIM], INDIVIDUAL_DIM: pos[INDIVIDUAL_DIM]},
            name="heading",
        ),
        position,
        f"Unit vector from {from_keypoint} to {to_keypoint}",
        normalise=False,
        units=None,
    )


def heading_angle(position: xr.DataArray, from_keypoint: str, to_keypoint: str) -> xr.DataArray:
    """Angle of the from → to vector in the x–y plane, radians in ``(-pi, pi]``.

    Parameters
    ----------
    position : xarray.DataArray
        Dims ``(time, space, keypoint, individual)``.
    from_keypoint, to_keypoint : str
        Tail and head of the vector.

    Returns
    -------
    xarray.DataArray
        Dims ``(time, individual)``.
    """
    pos, _ = _canonical(position)
    xy = pos.sel(space=["x", "y"])
    vec = xy.sel(keypoint=to_keypoint, drop=True) - xy.sel(keypoint=from_keypoint, drop=True)
    angle = compute_signed_angle_2d(vec, np.array([1.0, 0.0]), v_as_left_operand=True)

    return _feature(
        angle.rename("heading_angle"),
        position,
        f"Angle of {from_keypoint}->{to_keypoint} in the x-y plane",
        normalise=False,
        units="rad",
    )


def joint_angles(position: xr.DataArray, triplets: Sequence[tuple[str, str, str]]) -> xr.DataArray:
    """Angle at the middle keypoint of each ``(a, b, c)`` triplet, radians in ``[0, pi]``.

    Parameters
    ----------
    position : xarray.DataArray
        Dims ``(time, space, keypoint, individual)``.
    triplets : sequence of (str, str, str)
        Keypoint triplets; the angle is measured at ``b`` between ``b->a`` and ``b->c``.

    Returns
    -------
    xarray.DataArray
        Dims ``(time, angle, individual)``; ``angle`` coords are ``"a-b-c"`` strings.
        NaN where either arm has zero length.
    """
    pos, time = _canonical(position)
    if len(triplets) == 0:
        raise ValueError("triplets must not be empty")
    arr = pos.values.astype(np.float64)
    out = np.empty((arr.shape[0], len(triplets), arr.shape[3]))
    for t, (a, b, c) in enumerate(triplets):
        pb = arr[:, :, _keypoint_index(pos, b), :]
        ba = arr[:, :, _keypoint_index(pos, a), :] - pb
        bc = arr[:, :, _keypoint_index(pos, c), :] - pb
        cosine = _safe_divide(np.sum(ba * bc, axis=1), np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
        out[:, t, :] = np.arccos(np.clip(cosine, -1.0, 1.0))

    return _feature(
        xr.DataArray(
            out,
            dims=(time, ANGLE_DIM, INDIVIDUAL_DIM),
            coords={
                time: pos[time],
                ANGLE_DIM: [f"{a}-{b}-{c}" for a, b, c in triplets],
                INDIVIDUAL_DIM: pos[INDIVIDUAL_DIM],
            },
            name="joint_angles",
        ),
        position,
        "Angle at the middle keypoint of each triplet",
        normalise=False,
        units="rad",
    )


def polygon_area(position: xr.DataArray, keypoints: Sequence[str]) -> xr.DataArray:
    """Shoelace area of the polygon through the listed keypoints, in the x–y plane.

    Parameters
    ----------
    position : xarray.DataArray
        Dims ``(time, space, keypoint, individual)``.
    keypoints : sequence of str
        Polygon vertices in order (at least three).

    Returns
    -------
    xarray.DataArray
        Dims ``(time, individual)``; absolute area.
    """
    pos, time = _canonical(position)
    if len(keypoints) < 3:
        raise ValueError(f"A polygon needs at least three keypoints, got {list(keypoints)}")
    idx = [_keypoint_index(pos, k) for k in keypoints]
    arr = pos.values.astype(np.float64)
    x = arr[:, 0, idx, :]
    y = arr[:, 1, idx, :]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    area = 0.5 * np.abs(np.sum(x * y_next - x_next * y, axis=1))

    return _feature(
        xr.DataArray(
            area,
            dims=(time, INDIVIDUAL_DIM),
            coords={time: pos[time], INDIVIDUAL_DIM: pos[INDIVIDUAL_DIM]},
            name="polygon_area",
        ),
        position,
        f"Shoelace area of the polygon through {'-'.join(keypoints)}",
    )


def speed_direction(velocity: xr.DataArray) -> xr.DataArray:
    """Velocity divided by its norm: the unit direction of motion per keypoint.

    Parameters
    ----------
    velocity : xarray.DataArray
        Dims ``(time, space, keypoint, individual)``.

    Returns
    -------
    xarray.DataArray
        Dims ``(time, space, keypoint, individual)``; NaN where the speed is 0.
    """
    vel, _ = _canonical(velocity)
    arr = vel.values.astype(np.float64)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    unit = _safe_divide(arr, norm)

    return _feature(
        xr.DataArray(unit, dims=vel.dims, coords=vel.coords, name="speed_direction"),
        velocity,
        "Unit direction of the velocity vector",
        normalise=False,
        units=None,
    )
