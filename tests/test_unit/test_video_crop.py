"""Display crop geometry for camera views.

A crop is selected as a free-hand rectangle on the video canvas and must snap
to whole pixel edges inside the image (``snap_crop_rect``); it is applied as
pygfx world-space clipping planes on the y-flipped video image
(``crop_clip_planes``). The kept/discarded rule below mirrors pygfx's
``clipping_planes.wgsl`` (mode "ANY") verbatim: a fragment is discarded where
``dot(world_pos, plane.xyz) < plane.w`` — ``w`` is a dot-product threshold,
not the ``+d`` of the plane-equation convention, and getting that sign wrong
blanks the whole video.
"""

from __future__ import annotations

import numpy as np

from ethograph.gui.pygfx_video import MIN_CROP_SIZE_PX, crop_clip_planes, snap_crop_rect


class TestSnapCropRect:
    def test_snaps_to_pixel_edges(self):
        assert snap_crop_rect(10.4, 5.6, 99.5, 50.2, 640, 480) == (10, 6, 100, 50)

    def test_normalizes_corner_order(self):
        # Dragged from bottom-right to top-left — same rect either way.
        assert snap_crop_rect(100, 50, 10, 6, 640, 480) == (10, 6, 100, 50)

    def test_clamps_to_image_bounds(self):
        assert snap_crop_rect(-20.0, -5.0, 700.0, 500.0, 640, 480) == (0, 0, 640, 480)

    def test_degenerate_rect_is_none(self):
        assert snap_crop_rect(50.0, 50.0, 50.4, 200.0, 640, 480) is None
        assert snap_crop_rect(50.0, 50.0, 200.0, 50.4, 640, 480) is None
        # A click with no drag at all.
        assert snap_crop_rect(50.0, 50.0, 50.0, 50.0, 640, 480) is None

    def test_minimum_size_boundary(self):
        rect = snap_crop_rect(0, 0, MIN_CROP_SIZE_PX, MIN_CROP_SIZE_PX, 640, 480)
        assert rect == (0, 0, MIN_CROP_SIZE_PX, MIN_CROP_SIZE_PX)
        assert snap_crop_rect(0, 0, MIN_CROP_SIZE_PX - 1, MIN_CROP_SIZE_PX, 640, 480) is None


def _kept_by_shader(planes, x: float, y: float) -> bool:
    """The exact fragment test from pygfx's clipping_planes.wgsl (mode ANY)."""
    return not any(a * x + b * y + c * 0.0 < w for a, b, c, w in planes)


class TestCropClipPlanes:
    def test_four_planes(self):
        assert len(crop_clip_planes((10, 6, 100, 50), 480.0)) == 4

    def test_inside_survives_outside_clipped(self):
        # Image rect (x0, y0, x1, y1) = (10, 6, 100, 50), y down; the image is
        # rendered y-flipped, so image y maps to world y = 480 - y.
        planes = crop_clip_planes((10, 6, 100, 50), 480.0)

        def kept(x_img: float, y_img: float) -> bool:
            return _kept_by_shader(planes, x_img, 480.0 - y_img)

        assert kept(50, 25)  # centre
        assert kept(10, 6) and kept(100, 50)  # both corners, inclusive edges
        assert not kept(9, 25)  # left of the crop
        assert not kept(101, 25)  # right of the crop
        assert not kept(50, 5)  # above the crop (image space)
        assert not kept(50, 51)  # below the crop (image space)

    def test_crop_never_blanks_its_own_interior(self):
        # Regression: with the w sign flipped, the right/top planes exclude
        # every image pixel and the whole video disappears on crop.
        planes = crop_clip_planes((100, 100, 467, 339), 480.0)
        assert _kept_by_shader(planes, 283.0, 480.0 - 220.0)

    def test_full_frame_keeps_everything(self):
        planes = crop_clip_planes((0, 0, 640, 480), 480.0)
        xs, ys = np.meshgrid(np.linspace(0, 640, 9), np.linspace(0, 480, 9))
        for x, y_world in zip(xs.ravel(), ys.ravel()):
            assert _kept_by_shader(planes, float(x), float(y_world))
