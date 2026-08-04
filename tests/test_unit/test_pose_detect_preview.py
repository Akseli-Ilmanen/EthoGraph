"""The detector tuning preview.

Painting is checked by rendering into a pixmap rather than by comparing pixels —
what matters is that a widget handed real detector output draws without crashing
and shows *something*, since the alternative failure (a blank panel) is exactly
what the preview exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication  # noqa: E402

from ethograph.gui.pose_detect import DetectionPreview, PreviewShape  # noqa: E402
from ethograph.gui.pose_detect_preview import PREVIEW_HEIGHT, PreviewPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _frame(value: int = 40, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _quad(x: float, y: float, side: float = 16.0) -> np.ndarray:
    return np.array([[x, y], [x + side, y], [x + side, y + side], [x, y + side]], dtype=np.float64)


def _preview() -> DetectionPreview:
    return DetectionPreview(
        shapes=[
            PreviewShape(xy=np.array([16.0, 16.0]), label=24, accepted=True, outline=_quad(8, 8), quality=0.9),
            PreviewShape(
                xy=np.array([44.0, 44.0]),
                label=None,
                accepted=False,
                reason="seen but not decoded",
                outline=_quad(38, 38, 12),
            ),
        ],
        size=(64, 64),
    )


COLORS = {24: (0.2, 0.9, 0.3)}


def test_the_panel_draws_at_the_fixed_height(qapp):
    panel = PreviewPanel()
    panel.show_preview(_frame(), _preview(), COLORS, {24: "tag 3"})

    assert panel.pixmap() is not None
    assert panel.pixmap().height() == PREVIEW_HEIGHT
    assert panel.text() == ""


def test_the_panel_says_why_it_is_empty(qapp):
    """A blank panel is the failure this whole widget exists to prevent."""
    panel = PreviewPanel()
    panel.show_preview(_frame(), _preview(), COLORS)
    panel.show_message("Load a video to preview the detector.")

    assert panel.text() == "Load a video to preview the detector."
    assert panel.pixmap().isNull()


def test_the_panel_draws_a_shape_with_no_outline(qapp):
    """A detector without ``diagnose`` reports centres, not quads."""
    preview = DetectionPreview(
        shapes=[PreviewShape(xy=np.array([32.0, 32.0]), label=24, accepted=True)],
        size=(64, 64),
    )
    panel = PreviewPanel()
    panel.show_preview(_frame(), preview, COLORS, {24: "tag 3"})

    assert panel.pixmap().height() == PREVIEW_HEIGHT


def test_the_panel_handles_a_mono_frame(qapp):
    """Infrared and machine-vision cameras hand over ``(H, W)``, not RGB."""
    panel = PreviewPanel()
    panel.show_preview(np.full((64, 64), 60, np.uint8), _preview(), COLORS)

    assert panel.pixmap().height() == PREVIEW_HEIGHT
