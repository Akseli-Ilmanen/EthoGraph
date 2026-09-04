"""The label timeline is a panel that exists only to show labels.

So the per-plot-type overlay setting — which can hide labels on every other
type — never applies to it: whatever ``label_overlay_modes`` says, a ribbon
draws every label in full. A video-only session's labels have no other home.
"""

import pandas as pd
import pytest

pytest.importorskip("qtpy")

from ethograph.gui.app_constants import LABEL_OVERLAY_MODE_FULL, LABEL_OVERLAY_MODE_NONE  # noqa: E402
from ethograph.gui.label_drawing_mixin import LabelDrawingMixin  # noqa: E402
from ethograph.gui.plots_labelribbon import LabelRibbonPlot  # noqa: E402
from ethograph.labels.intervals import EVENT_TYPE_POINT, EVENT_TYPE_STATE  # noqa: E402

MAPPINGS = {
    1: {"name": "hop", "color": (1.0, 0.0, 0.0)},
    2: {"name": "peck", "color": (0.0, 1.0, 0.0)},
}


class _Container(LabelDrawingMixin):
    def __init__(self, app_state, plots):
        self.app_state = app_state
        self._plots = plots
        self.label_mappings = MAPPINGS

    def _get_all_plots(self) -> list:
        return list(self._plots)


@pytest.fixture
def ribbon(qtbot, app_state):
    plot = LabelRibbonPlot(app_state)
    qtbot.addWidget(plot)
    return plot


def _slots():
    df = pd.DataFrame(
        {
            "onset_s": [1.0, 2.5],
            "offset_s": [2.0, 2.5],
            "labels": [1, 2],
            "event_type": [EVENT_TYPE_STATE, EVENT_TYPE_POINT],
            "labeling_method": ["manual", "manual"],
        }
    )
    return [{"df": df, "label_ids": None, "position": "main"}]


def test_ribbon_has_no_value_axis(ribbon):
    assert ribbon.panel_type == "labels"
    assert not ribbon.plot_item.getAxis("left").isVisible()
    assert ribbon.vb.viewRange()[1] == [0.0, 1.0]


def test_ribbon_draws_labels_whatever_the_overlay_modes_say(ribbon, app_state):
    app_state.label_overlay_modes = {"lineplot": LABEL_OVERLAY_MODE_NONE, "labels": LABEL_OVERLAY_MODE_NONE}
    container = _Container(app_state, [ribbon])

    assert container._label_overlay_mode(ribbon) == LABEL_OVERLAY_MODE_FULL
    container.draw_all_labels(_slots())

    assert len(ribbon.label_items) == 2, "one state rectangle and one point line"
