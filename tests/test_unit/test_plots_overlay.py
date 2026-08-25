"""An overlay outliving the panel it hangs on.

Closing a feature panel while an onset-curve overlay is registered leaves the
manager holding a host whose Qt object is gone; every later access raised
``RuntimeError: wrapped C/C++ object of type AxisItem has been deleted``, which
escaped ``CurationPanel._teardown`` and aborted the whole review stop.
"""

from __future__ import annotations

import numpy as np
import pytest

from ethograph.gui.plots_overlay import OverlayManager


class _DeadAxis:
    def hide(self):
        raise RuntimeError("wrapped C/C++ object of type AxisItem has been deleted")

    def show(self):
        raise RuntimeError("wrapped C/C++ object of type AxisItem has been deleted")


class _DeadPlotItem:
    def getAxis(self, _name):
        raise RuntimeError("wrapped C/C++ object of type PlotItem has been deleted")

    def viewRange(self):
        raise RuntimeError("wrapped C/C++ object of type PlotItem has been deleted")

    def hideAxis(self, _name):
        raise RuntimeError("wrapped C/C++ object of type PlotItem has been deleted")

    def scene(self):
        raise RuntimeError("wrapped C/C++ object of type PlotItem has been deleted")


class _DeadViewBox:
    def removeItem(self, _item):
        raise RuntimeError("wrapped C/C++ object of type ViewBox has been deleted")


class _DeadHost:
    """A closed panel: every Qt access raises, exactly as PyQt does."""

    plot_item = _DeadPlotItem()
    vb = _DeadViewBox()


class _LiveAxis:
    def __init__(self):
        self.hidden = False

    def hide(self):
        self.hidden = True

    def show(self):
        self.hidden = False

    def setStyle(self, **_kw):
        pass

    def setTicks(self, _ticks):
        pass


class _LivePlotItem:
    def __init__(self):
        self.axis = _LiveAxis()

    def getAxis(self, _name):
        return self.axis

    def viewRange(self):
        return [[0.0, 1.0], [0.0, 1.0]]


class _LiveViewBox:
    def __init__(self):
        self.items = []

    def addItem(self, item, **_kw):
        self.items.append(item)

    def removeItem(self, item):
        self.items.remove(item)


class _LiveHost:
    def __init__(self):
        self.plot_item = _LivePlotItem()
        self.vb = _LiveViewBox()


class _Curve:
    def setData(self, *_a, **_kw):
        pass


@pytest.fixture()
def manager_with_overlay():
    mgr = OverlayManager()
    host = _LiveHost()
    mgr.add_scaled_overlay(
        "onset_curve_31", host, _Curve(), np.arange(4.0), np.linspace(0.0, 1.0, 4), data_min=0.0, data_max=1.0
    )
    return mgr, host


class TestClosedHost:
    def test_removing_an_overlay_whose_panel_is_gone_does_not_raise(self, manager_with_overlay):
        mgr, _ = manager_with_overlay
        mgr._entries["onset_curve_31"].host_plot = _DeadHost()
        mgr.remove_overlay("onset_curve_31")
        assert not mgr.has_overlay("onset_curve_31")

    def test_rescaling_a_dead_host_forgets_it_instead_of_raising(self, manager_with_overlay):
        mgr, _ = manager_with_overlay
        dead = _DeadHost()
        mgr._entries["onset_curve_31"].host_plot = dead
        mgr.rescale_for_plot(dead)
        assert not mgr.has_overlay("onset_curve_31")


class TestLiveHost:
    def test_a_live_host_still_gets_its_right_axis_managed(self, manager_with_overlay):
        mgr, host = manager_with_overlay
        assert host.plot_item.axis.hidden is False
        mgr.remove_overlay("onset_curve_31")
        assert host.plot_item.axis.hidden is True
        assert host.vb.items == []
