"""Anchored shapes that deform to follow pose keypoints.

A *shape* is a canonical template (square, triangle, circle …) with named
control points. The user binds two or more control points to keypoints
(*anchors*); each frame a transform is fit from the template's anchor points to
the live keypoint positions, and the whole outline is transformed by it.

Transform family (adaptive to the number of anchors):

- **2 anchors** → a *similarity* transform (translation + rotation + uniform
  scale). Angles are preserved, so e.g. a triangle anchored at its base
  midpoint and apex keeps its base perpendicular to the median.
- **3+ anchors** → an *affine* transform (least-squares). The shape may
  stretch/shear to satisfy the anchors; angles are not preserved.

This mirrors ``PrecomputedRenderer``: geometry is precomputed per frame so the
result drops straight into a napari Shapes layer whose first vertex coordinate
is the frame index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ShapeTemplate:
    """A canonical shape with named control points and an outline.

    control_points
        Mapping ``name -> (x, y)`` in canonical (template) coordinates.
    outline
        Control-point names (or extra ``(x, y)`` vertices) traced, in order, to
        draw the shape.
    """

    name: str
    control_points: dict[str, tuple[float, float]]
    outline: list = field(default_factory=list)

    def anchor_point_names(self) -> list[str]:
        return list(self.control_points.keys())

    def outline_array(self) -> np.ndarray:
        pts = []
        for v in self.outline:
            pts.append(self.control_points[v] if isinstance(v, str) else v)
        return np.asarray(pts, dtype=float)


def _circle_outline(n: int = 48) -> list[tuple[float, float]]:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(float(np.cos(a)), float(np.sin(a))) for a in ang]


SHAPE_TEMPLATES: dict[str, ShapeTemplate] = {
    "square": ShapeTemplate(
        name="square",
        control_points={
            "corner_tl": (-1.0, -1.0),
            "corner_tr": (1.0, -1.0),
            "corner_br": (1.0, 1.0),
            "corner_bl": (-1.0, 1.0),
            "mid_top": (0.0, -1.0),
            "mid_right": (1.0, 0.0),
            "mid_bottom": (0.0, 1.0),
            "mid_left": (-1.0, 0.0),
            "center": (0.0, 0.0),
        },
        outline=["corner_tl", "corner_tr", "corner_br", "corner_bl"],
    ),
    "triangle": ShapeTemplate(
        name="triangle",
        control_points={
            "apex": (0.0, -1.0),
            "base_left": (-1.0, 1.0),
            "base_right": (1.0, 1.0),
            "base_mid": (0.0, 1.0),
            "centroid": (0.0, 1.0 / 3.0),
        },
        outline=["apex", "base_left", "base_right"],
    ),
    "circle": ShapeTemplate(
        name="circle",
        control_points={
            "center": (0.0, 0.0),
            "right": (1.0, 0.0),
            "top": (0.0, -1.0),
            "left": (-1.0, 0.0),
            "bottom": (0.0, 1.0),
        },
        outline=_circle_outline(),
    ),
}


def fit_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``x' = R @ x + t`` mapping template points *src* to targets *dst*.

    Two points → similarity (rotation + uniform scale). Three or more → affine
    (least squares). Returns ``(R, t)`` with ``R`` shape ``(2, 2)`` and ``t``
    shape ``(2,)``. Raises ``ValueError`` for degenerate input.
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) != len(dst) or len(src) < 2:
        raise ValueError("Need >=2 matching anchor points")

    if len(src) == 2:
        sv = src[1] - src[0]
        dv = dst[1] - dst[0]
        src_len = float(np.hypot(*sv))
        if src_len == 0 or np.hypot(*dv) == 0:
            raise ValueError("Degenerate anchors")
        scale = float(np.hypot(*dv)) / src_len
        angle = np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0])
        c, s = np.cos(angle) * scale, np.sin(angle) * scale
        r_mat = np.array([[c, -s], [s, c]])
        return r_mat, dst[0] - r_mat @ src[0]

    a = np.column_stack([src, np.ones(len(src))])  # (n, 3)
    m, *_ = np.linalg.lstsq(a, dst, rcond=None)  # (3, 2)
    return m[:2].T, m[2]


def transform_points(r_mat: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return np.asarray(pts, dtype=float) @ r_mat.T + t


def shape_outline_for_frame(
    template: ShapeTemplate,
    anchor_names: list[str],
    anchor_target_pts: np.ndarray,
    scale: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray | None:
    """Return the transformed outline ``(K, 2)`` in image coords, or ``None``.

    ``scale`` reshapes the canonical template (``(width, height)`` factors)
    before fitting, so e.g. a triangle's length-to-width ratio can be tuned.
    Both the anchor points and the outline are scaled consistently. ``None`` is
    returned when any target anchor is NaN or the fit is degenerate.
    """
    if np.any(np.isnan(anchor_target_pts)):
        return None
    s = np.asarray(scale, dtype=float)
    src = np.array([template.control_points[n] for n in anchor_names]) * s
    try:
        r_mat, t = fit_transform(src, anchor_target_pts)
    except ValueError:
        return None
    return transform_points(r_mat, t, template.outline_array() * s)
