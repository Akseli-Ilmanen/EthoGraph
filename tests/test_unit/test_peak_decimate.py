"""Regression guard for the lineplot zoom-out invisibility bug.

pyqtgraph's ``method="peak"`` downsampling takes max/min per bin, so any raw
NaN in a bin blanks the whole bin — NaN-gapped curves vanished when zoomed
out. The line plot therefore creates curves via ``_nan_safe_curve_args``:
NaNs are forward-filled (min/max stay within real data values) and a
``connect`` mask hides the gap segments; pyqtgraph bins the mask during
downsampling, so gaps survive at any zoom.
"""

import numpy as np
import pytest

pg = pytest.importorskip("pyqtgraph")

from ethograph.gui.plots_lineplot import _nan_safe_curve_args, plot_multidim, plot_singledim  # noqa: E402


@pytest.fixture
def nan_gapped_signal():
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, 0.01)
    y = np.sin(t)
    y[rng.choice(len(y), size=len(y) // 20, replace=False)] = np.nan
    return t, y


def test_nan_safe_curve_args_fills_and_masks(nan_gapped_signal):
    _, y = nan_gapped_signal
    filled, connect = _nan_safe_curve_args(y)
    assert np.isfinite(filled).all()
    assert np.nanmax(filled) <= np.nanmax(y)
    assert np.nanmin(filled) >= np.nanmin(y)
    finite = np.isfinite(y)
    np.testing.assert_array_equal(connect[:-1], finite[:-1] & finite[1:])


def test_nan_safe_curve_args_leading_gap():
    y = np.array([np.nan, np.nan, 3.0, 4.0])
    filled, connect = _nan_safe_curve_args(y)
    np.testing.assert_array_equal(filled, [3.0, 3.0, 3.0, 4.0])
    assert not connect[0] and not connect[1] and connect[2]


def test_nan_safe_curve_args_passthrough():
    y = np.arange(5.0)
    filled, connect = _nan_safe_curve_args(y)
    assert connect is None
    np.testing.assert_array_equal(filled, y)


def _zoomed_out_display(qtbot, w, item, view):
    w.setXRange(*view, padding=0)
    qtbot.wait(10)
    return item._getDisplayDataset()


@pytest.mark.parametrize("view", [(4, 6), (0, 10), (-50, 60), (-500, 600), (-5000, 6000)])
def test_singledim_curve_stays_visible_when_zoomed_out(qtbot, nan_gapped_signal, view):
    t, y = nan_gapped_signal
    w = pg.PlotWidget()
    qtbot.addWidget(w)
    w.resize(800, 400)
    w.show()

    items = plot_singledim(w.getPlotItem(), t, y)
    for item in items:
        item.setDownsampling(auto=True, method="peak")

    dsp = _zoomed_out_display(qtbot, w, items[0], view)
    assert dsp.x is not None and len(dsp.x) > 1
    assert np.isfinite(dsp.y).sum() > 1


def test_multidim_curves_stay_visible_when_zoomed_out(qtbot, nan_gapped_signal):
    t, y = nan_gapped_signal
    data = np.stack([y, y * 2], axis=1)
    w = pg.PlotWidget()
    qtbot.addWidget(w)
    w.resize(800, 400)
    w.show()

    items = plot_multidim(w.getPlotItem(), t, data, coord_labels=["a", "b"])
    for item in items:
        item.setDownsampling(auto=True, method="peak")

    for item in items:
        dsp = _zoomed_out_display(qtbot, w, item, (-500, 600))
        assert np.isfinite(dsp.y).sum() > 1
