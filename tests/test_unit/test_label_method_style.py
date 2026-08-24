"""A label's outline says who vouches for it.

An automated label — a model's output nobody has looked at — is drawn dotted;
a manual or curated one solid. That is the only thing on the plot that tells a
reviewer what still needs a look, so it has to survive both draw paths (point
events and state intervals), the in-place restyle a single curation performs,
and a labels frame that carries no ``labeling_method`` column at all.
"""

import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtCore import Qt

pytest.importorskip("qtpy")

from ethograph.gui.label_drawing_mixin import LabelDrawingMixin, draw_key  # noqa: E402
from ethograph.labels.intervals import (  # noqa: E402
    EVENT_TYPE_POINT,
    LABELING_AUTOMATED,
    LABELING_CURATED,
    LABELING_MANUAL,
)

LABEL_ID = 3


class _Container(LabelDrawingMixin):
    """Minimal host: the mixin needs the plot list and the label mappings."""

    def __init__(self, plot):
        self._plots = [plot]
        self.label_mappings = {LABEL_ID: {"name": "peck", "color": (1.0, 0.0, 0.0)}}

    def _get_all_plots(self) -> list:
        return list(self._plots)


@pytest.fixture
def plot(qtbot):
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.plot_item = widget.getPlotItem()
    return widget


@pytest.fixture
def container(plot):
    return _Container(plot)


def _row(method: str, *, point: bool = True, onset: float = 1.0) -> dict:
    return {
        "labels": LABEL_ID,
        "onset_s": onset,
        "offset_s": float("nan") if point else onset + 0.5,
        "event_type": EVENT_TYPE_POINT if point else "state",
        "individual": "ind0",
        "individual_rec": "",
        "labeling_method": method,
    }


def _styles(plot) -> set:
    """The pen styles of every outline drawn: a point event is one
    ``InfiniteLine``, a state interval a region whose two edge lines carry the
    boundary pen."""
    styles = set()
    for item in plot.plot_item.items:
        if isinstance(item, pg.LinearRegionItem):
            styles.update(line.pen.style() for line in item.lines)
        elif isinstance(item, pg.InfiniteLine):
            styles.add(item.pen.style())
    return styles


def _draw(container, plot, rows: list[dict]) -> None:
    container._draw_intervals_on_plot(plot, pd.DataFrame(rows))


class TestPointEvents:
    def test_automated_is_dotted_and_manual_is_solid(self, container, plot):
        _draw(container, plot, [_row(LABELING_AUTOMATED, onset=1.0), _row(LABELING_MANUAL, onset=2.0)])
        assert _styles(plot) == {Qt.PenStyle.DotLine, Qt.PenStyle.SolidLine}

    def test_curated_draws_like_manual(self, container, plot):
        """Curated means a human vouched for it — nothing left to flag."""
        _draw(container, plot, [_row(LABELING_CURATED)])
        assert _styles(plot) == {Qt.PenStyle.SolidLine}

    def test_a_frame_without_the_column_draws_solid(self, container, plot):
        """A labels file written before the column existed must still draw."""
        row = _row(LABELING_MANUAL)
        del row["labeling_method"]
        _draw(container, plot, [row])
        assert _styles(plot) == {Qt.PenStyle.SolidLine}


class TestStateIntervals:
    def test_boundaries_follow_the_method(self, container, plot):
        _draw(container, plot, [_row(LABELING_AUTOMATED, point=False)])
        assert _styles(plot) == {Qt.PenStyle.DotLine}

        plot.plot_item.clear()
        container._label_item_index = {}
        _draw(container, plot, [_row(LABELING_MANUAL, point=False)])
        assert _styles(plot) == {Qt.PenStyle.SolidLine}


class TestRestyle:
    def test_curating_one_label_swaps_its_pens_in_place(self, container, plot):
        """The automated → curated transition the GUI performs on one label
        must reach the drawn item without a full redraw."""
        _draw(container, plot, [_row(LABELING_AUTOMATED)])
        assert _styles(plot) == {Qt.PenStyle.DotLine}

        key = draw_key(LABEL_ID, 1.0, "ind0", "")
        assert container.restyle_label(key, automated=False) > 0
        assert _styles(plot) == {Qt.PenStyle.SolidLine}

    def test_an_unknown_key_restyles_nothing(self, container, plot):
        """0 is what tells the caller to fall back to a redraw."""
        _draw(container, plot, [_row(LABELING_AUTOMATED)])
        assert container.restyle_label(draw_key(LABEL_ID, 99.0, "ind0", ""), automated=False) == 0
