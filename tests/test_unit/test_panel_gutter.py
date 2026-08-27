"""Every stacked panel ends its plotting rectangle on the same pixel.

A heatmap carries a colorbar on its right; a line plot does not. Rather than
measure and match, every panel reserves the colorbar's footprint as a fixed
right gutter (``PANEL_RIGHT_GUTTER_PX``), where a line plot parks its legend.
"""

import numpy as np
import pyqtgraph as pg
import pytest

from ethograph.gui.app_constants import (
    COLORBAR_WIDTH_PX,
    PANEL_RIGHT_GUTTER_PX,
    PANEL_RIGHT_SPACER_PX,
)
from ethograph.gui.plots_base import right_gutter_width
from ethograph.gui.plots_heatmap import HeatmapPlot
from ethograph.gui.plots_lineplot import LinePlot, PlotData, add_legend, drop_legend, render_plot_data

WIDTH = 800


def _settle(qapp, *widgets):
    for w in widgets:
        w.resize(WIDTH, 300)
        w.show()
    for _ in range(3):
        qapp.processEvents()


def _vb_right(plot) -> float:
    return plot.plot_item.vb.geometry().right()


def test_gutter_is_the_colorbar_footprint(qapp):
    """Contract with pyqtgraph: a ColorBarItem inserted into a PlotItem takes
    exactly the spacer + gutter we reserve — whatever its tick labels say."""
    widget = pg.PlotWidget()
    image = pg.ImageItem()
    widget.plotItem.addItem(image)
    bar = pg.ColorBarItem(
        values=(-1, 1), colorMap=pg.colormap.get("viridis"), interactive=False, width=COLORBAR_WIDTH_PX
    )
    bar.setImageItem(image, insert_in=widget.plotItem)
    _settle(qapp, widget)

    layout = widget.plotItem.layout
    assert layout.columnMinimumWidth(4) == PANEL_RIGHT_SPACER_PX
    assert bar.geometry().width() == PANEL_RIGHT_GUTTER_PX

    bar.setLevels(values=(-0.123456, 0.98765))
    _settle(qapp, widget)
    assert bar.geometry().width() == PANEL_RIGHT_GUTTER_PX


def test_heatmap_and_lineplot_end_on_the_same_pixel(qapp, app_state):
    line = LinePlot(app_state)
    heat = HeatmapPlot(app_state)
    _settle(qapp, line, heat)
    assert _vb_right(line) == _vb_right(heat)
    assert _vb_right(line) == pytest.approx(WIDTH - PANEL_RIGHT_GUTTER_PX - PANEL_RIGHT_SPACER_PX, abs=2)


def test_visible_right_axis_trims_the_gutter(qapp, app_state):
    plain = LinePlot(app_state)
    scaled = LinePlot(app_state)
    scaled.plot_item.showAxis("right")
    _settle(qapp, plain, scaled)
    assert right_gutter_width(plain) == PANEL_RIGHT_GUTTER_PX
    assert right_gutter_width(scaled) < PANEL_RIGHT_GUTTER_PX

    scaled.reserve_right_gutter(right_gutter_width(scaled))
    _settle(qapp, plain, scaled)
    assert _vb_right(plain) == _vb_right(scaled)


def test_colorbar_panel_offers_no_legend_host(qapp, app_state):
    assert HeatmapPlot(app_state).legend_host() is None
    line = LinePlot(app_state)
    host = line.legend_host()
    assert host is not None
    assert line.legend_host() is host


def test_legend_lives_in_the_gutter_and_never_stacks(qapp, app_state):
    line = LinePlot(app_state)
    _settle(qapp, line)
    t = np.linspace(0.0, 1.0, 50)
    data = PlotData(time=t, data=np.stack([t, t * 2], axis=1), dim_labels=["a", "b"], title="f", ylabel="y")
    render_plot_data(line.plot_item, data, legend_host=line.legend_host())
    legend = line.plot_item.legend
    assert legend.parentItem() is line.legend_host()

    render_plot_data(line.plot_item, data, legend_host=line.legend_host())
    assert line.plot_item.legend is not legend
    assert legend.scene() is None  # the old one left the scene


def test_legend_without_host_overlays_the_viewbox(qapp, app_state):
    line = LinePlot(app_state)
    add_legend(line.plot_item)
    assert line.plot_item.legend.parentItem() is line.plot_item.vb
    drop_legend(line.plot_item)
    assert line.plot_item.legend is None
