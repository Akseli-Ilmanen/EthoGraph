"""Ad-hoc repro: 'All' checkbox on a multi-column feature (pynapple vs xarray)."""

import pyqtgraph as pg
from qtpy.QtWidgets import QApplication

from ethograph.utils.qt import set_combo_to_value


def _curve_count(lp):
    return sum(1 for it in lp.plot_items if isinstance(it, (pg.PlotDataItem, pg.PlotCurveItem)))


def test_all_checkbox_pynapple(moll2025_pynapple_gui):
    _, meta = moll2025_pynapple_gui
    dw = meta.data_widget
    lp = meta.plot_container.line_plots[0]
    lp.plot_clicked.emit(lp)
    QApplication.processEvents()

    set_combo_to_value(dw.combos["features"], "beakTip_position")
    QApplication.processEvents()
    print("combos:", list(dw.combos))
    print("all_checkboxes:", list(dw.all_checkboxes))
    print("feature_dims:", meta.app_state.data_loader.feature_dims("beakTip_position"))
    print("selections before:", lp._effective_selections())
    print("curves before:", _curve_count(lp))

    cb = dw.all_checkboxes["columns"]
    cb.setChecked(True)
    QApplication.processEvents()
    print("selections after:", lp._effective_selections())
    print("curves after:", _curve_count(lp))
    assert _curve_count(lp) == 3
