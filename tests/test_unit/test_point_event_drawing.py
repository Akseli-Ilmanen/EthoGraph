"""Point events draw, in every branch position.

``_draw_single_point`` styled its pen with ``pg.QtCore.Qt.SolidLine``.
pyqtgraph imports PyQt6 directly, where the unscoped enum aliases do not
exist (qtpy promotes them, pyqtgraph does not), so every point event raised
AttributeError mid-draw — taking the rest of that panel's labels with it.
The branch a point sits in was never the gate: ``_draw_single_point``
ignores ``position`` and draws a full-height line wherever the class lives.
"""

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest

pytest.importorskip("qtpy")

from ethograph.gui.app_constants import DEFAULT_LABEL_OVERLAY_MODES  # noqa: E402
from ethograph.gui.label_drawing_mixin import LabelDrawingMixin  # noqa: E402


class _State:
    label_overlay_modes = dict(DEFAULT_LABEL_OVERLAY_MODES)


class _Container(LabelDrawingMixin):
    """Minimal host: one audio panel, one point class and one state class."""

    def __init__(self, plot):
        self.app_state = _State()
        self.label_mappings = {
            1: {"name": "state", "color": np.array([0.0, 1.0, 0.0]), "branch": 0, "event_type": "state"},
            30: {"name": "contact", "color": np.array([1.0, 0.0, 0.0]), "branch": 1, "event_type": "point"},
        }
        self.audio_trace_plots = [plot]
        self.spectrogram_plots = []
        self.heatmap_plots = []
        self.neo_trace_plots = []
        self.ephys_trace_plot = None


@pytest.fixture
def container(qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.plot_item = widget.getPlotItem()
    widget.vb = widget.getPlotItem().vb
    widget.setXRange(0, 10)
    widget.setYRange(0, 1)
    return _Container(widget)


def _intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "onset_s": [2.5, 4.0],
            "offset_s": [np.nan, 5.0],
            "labels": [30, 1],
            "individual": ["Poppy", "Poppy"],
            "individual_rec": ["", ""],
            "event_type": ["point", "state"],
        }
    )


@pytest.mark.parametrize("position", ["main", "top1", "top2"])
def test_point_event_draws_in_every_branch_position(container, position):
    plot = container.audio_trace_plots[0]

    container.draw_all_labels([{"df": _intervals(), "label_ids": {1, 30}, "position": position}])

    lines = [i for i in plot.label_items if isinstance(i, pg.InfiniteLine)]
    assert len(lines) == 1, "the point event is one vertical line, whatever the branch"
    assert lines[0].value() == pytest.approx(2.5)
    assert len(plot.label_items) == 2, "the state label is drawn alongside it"
