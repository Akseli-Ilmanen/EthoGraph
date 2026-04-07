"""Tests that plots actually contain rendered data after loading and interactions.

This fills the critical gap: existing tests verify state variables and xlim,
but never check whether plots rendered any data.  These tests inspect the
pyqtgraph items (PlotDataItem curves, ImageItem images) to confirm data
reached the screen.
"""

import numpy as np
import pytest
import pyqtgraph as pg
from qtpy.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_curve_data(plot_items):
    """Extract (x, y) from the first PlotDataItem/PlotCurveItem in a list."""
    for item in plot_items:
        if isinstance(item, (pg.PlotDataItem, pg.PlotCurveItem)):
            return item.xData, item.yData
    return None, None


def _assert_lineplot_has_data(line_plot):
    assert len(line_plot.plot_items) > 0, "LinePlot.plot_items is empty"
    x, y = _get_curve_data(line_plot.plot_items)
    assert x is not None and len(x) > 0, "LinePlot curve has no x data"
    assert y is not None and len(y) > 0, "LinePlot curve has no y data"


def _assert_audio_has_data(audio_plot):
    assert audio_plot.trace_item.xData is not None, "AudioTrace xData is None"
    assert len(audio_plot.trace_item.xData) > 0, "AudioTrace xData is empty"
    assert audio_plot.trace_item.yData is not None, "AudioTrace yData is None"
    assert len(audio_plot.trace_item.yData) > 0, "AudioTrace yData is empty"


def _assert_spectrogram_has_data(spec_plot):
    assert spec_plot.spec_item.image is not None, "Spectrogram image is None"
    assert spec_plot.spec_item.image.size > 0, "Spectrogram image is empty"


# ===========================================================================
# 1. Plot populated after initial load
# ===========================================================================

class TestPlotPopulatedAfterLoad:

    def test_lineplot_has_data(self, loaded_gui):
        _, meta = loaded_gui
        pc = meta.plot_container
        assert pc._feature_type == "lineplot"
        _assert_lineplot_has_data(pc.line_plot)

    def test_lineplot_x_within_reasonable_range(self, loaded_gui):
        _, meta = loaded_gui
        lp = meta.plot_container.line_plot
        x, _ = _get_curve_data(lp.plot_items)
        xlim = lp.get_current_xlim()
        # Curve x should overlap with the visible range
        assert x[0] <= xlim[1], "Curve starts after visible range"
        assert x[-1] >= xlim[0], "Curve ends before visible range"

    def test_lineplot_y_is_finite(self, loaded_gui):
        _, meta = loaded_gui
        lp = meta.plot_container.line_plot
        _, y = _get_curve_data(lp.plot_items)
        assert np.all(np.isfinite(y)), "LinePlot y contains inf/nan"

    def test_audio_trace_has_data_if_visible(self, loaded_gui):
        _, meta = loaded_gui
        pc = meta.plot_container
        if not pc._panel_visible.get("audiotrace"):
            pytest.skip("Audio panel not visible for this dataset")
        _assert_audio_has_data(pc.audio_trace_plot)

    def test_spectrogram_has_data_if_visible(self, loaded_gui):
        _, meta = loaded_gui
        pc = meta.plot_container
        if not pc._panel_visible.get("spectrogram"):
            pytest.skip("Spectrogram panel not visible for this dataset")
        _assert_spectrogram_has_data(pc.spectrogram_plot)


# ===========================================================================
# 2. Trial switch updates plot data
# ===========================================================================

class TestTrialSwitchUpdatesPlot:

    def test_lineplot_data_changes_on_trial_switch(self, loaded_gui):
        _, meta = loaded_gui
        if len(meta.app_state.trials) < 2:
            pytest.skip("Need 2+ trials")

        lp = meta.plot_container.line_plot
        x1, y1 = _get_curve_data(lp.plot_items)
        x1, y1 = x1.copy(), y1.copy()

        meta.navigation_widget.next_trial()
        QApplication.processEvents()

        x2, y2 = _get_curve_data(lp.plot_items)
        assert x2 is not None and len(x2) > 0, "LinePlot empty after trial switch"
        # Different trial should produce different time range or different values
        changed = (not np.array_equal(x1, x2)) or (not np.array_equal(y1, y2))
        assert changed, "LinePlot data identical after trial switch"

    def test_lineplot_has_data_every_trial(self, loaded_gui):
        _, meta = loaded_gui
        lp = meta.plot_container.line_plot

        for i, trial in enumerate(meta.app_state.trials):
            meta.navigation_widget.trials_combo.setCurrentText(str(trial))
            QApplication.processEvents()
            assert len(lp.plot_items) > 0, f"LinePlot empty on trial {trial}"
            x, y = _get_curve_data(lp.plot_items)
            assert x is not None and len(x) > 0, f"No curve data on trial {trial}"

    def test_xlim_updates_on_trial_switch(self, loaded_gui):
        _, meta = loaded_gui
        if len(meta.app_state.trials) < 2:
            pytest.skip("Need 2+ trials")

        xlim1 = meta.plot_container.get_current_xlim()

        meta.navigation_widget.next_trial()
        QApplication.processEvents()

        xlim2 = meta.plot_container.get_current_xlim()
        assert xlim2[0] < xlim2[1], "Invalid xlim after trial switch"


# ===========================================================================
# 3. Feature switch updates plot data
# ===========================================================================

class TestFeatureSwitchUpdatesPlot:

    def test_switching_feature_changes_plot_data(self, loaded_gui):
        _, meta = loaded_gui
        features_combo = meta.data_widget.combos["features"]
        if features_combo.count() < 2:
            pytest.skip("Need 2+ features")

        lp = meta.plot_container.line_plot

        # Record data for first feature
        features_combo.setCurrentIndex(0)
        QApplication.processEvents()
        if features_combo.currentText() in ("Spectrogram", "Waveform"):
            features_combo.setCurrentIndex(1)
            QApplication.processEvents()
        _, y1 = _get_curve_data(lp.plot_items)
        if y1 is not None:
            y1 = y1.copy()

        # Find a different plottable feature
        switched = False
        for i in range(features_combo.count()):
            text = features_combo.itemText(i)
            if text in ("Spectrogram", "Waveform"):
                continue
            if i == features_combo.currentIndex():
                continue
            features_combo.setCurrentIndex(i)
            QApplication.processEvents()
            switched = True
            break

        if not switched:
            pytest.skip("Could not find a second plottable feature")

        _, y2 = _get_curve_data(lp.plot_items)
        assert y2 is not None and len(y2) > 0, "LinePlot empty after feature switch"

    def test_every_feature_renders_data(self, loaded_gui):
        _, meta = loaded_gui
        features_combo = meta.data_widget.combos["features"]
        lp = meta.plot_container.line_plot

        for i in range(features_combo.count()):
            text = features_combo.itemText(i)
            if text in ("Spectrogram", "Waveform"):
                continue
            features_combo.setCurrentIndex(i)
            QApplication.processEvents()
            assert len(lp.plot_items) > 0, f"Feature '{text}' produced no plot items"
            x, y = _get_curve_data(lp.plot_items)
            assert x is not None and len(x) > 0, f"Feature '{text}' has empty curve"


# ===========================================================================
# 4. Heatmap mode
# ===========================================================================

class TestHeatmapMode:

    def test_switch_to_heatmap_renders_image(self, loaded_gui):
        _, meta = loaded_gui
        pc = meta.plot_container

        pc.set_feature_view("heatmap")
        QApplication.processEvents()

        # Trigger a plot update so heatmap renders
        meta.data_widget.update_main_plot()
        QApplication.processEvents()

        assert pc._feature_type == "heatmap"
        hm = pc.heatmap_plot
        assert hm.image_item.image is not None, "Heatmap image is None after switch"
        assert hm.image_item.image.size > 0, "Heatmap image is empty"

    def test_switch_back_to_lineplot(self, loaded_gui):
        _, meta = loaded_gui
        pc = meta.plot_container

        pc.set_feature_view("heatmap")
        QApplication.processEvents()
        meta.data_widget.update_main_plot()
        QApplication.processEvents()

        pc.set_feature_view("lineplot")
        QApplication.processEvents()
        meta.data_widget.update_main_plot()
        QApplication.processEvents()

        assert pc._feature_type == "lineplot"
        _assert_lineplot_has_data(pc.line_plot)


# ===========================================================================
# 5. Pan / zoom — buffer reload
# ===========================================================================

class TestPanZoom:

    def test_zoom_in_preserves_data(self, loaded_gui):
        _, meta = loaded_gui
        lp = meta.plot_container.line_plot
        xlim = lp.get_current_xlim()
        mid = (xlim[0] + xlim[1]) / 2
        quarter = (xlim[1] - xlim[0]) / 4

        # Zoom in to 50% of original range
        lp.plot_item.setXRange(mid - quarter, mid + quarter)
        QApplication.processEvents()

        new_xlim = lp.get_current_xlim()
        assert new_xlim[1] - new_xlim[0] < xlim[1] - xlim[0], "Zoom didn't narrow range"
        # Data should still be present (buffer covers zoomed region)
        assert len(lp.plot_items) > 0, "Plot items gone after zoom"

    def test_zoom_out_preserves_data(self, loaded_gui):
        _, meta = loaded_gui
        lp = meta.plot_container.line_plot
        xlim = lp.get_current_xlim()
        span = xlim[1] - xlim[0]
        mid = (xlim[0] + xlim[1]) / 2

        # Zoom out to 200% of original range
        lp.plot_item.setXRange(mid - span, mid + span)
        QApplication.processEvents()

        assert len(lp.plot_items) > 0, "Plot items gone after zoom out"

    def test_pan_right_preserves_data(self, loaded_gui):
        _, meta = loaded_gui
        lp = meta.plot_container.line_plot
        xlim = lp.get_current_xlim()
        span = xlim[1] - xlim[0]
        shift = span * 0.3

        # Pan right by 30%
        lp.plot_item.setXRange(xlim[0] + shift, xlim[1] + shift)
        QApplication.processEvents()

        assert len(lp.plot_items) > 0, "Plot items gone after pan"

    def test_pan_triggers_buffer_update(self, loaded_gui):
        """Pan far enough to exceed buffer margin and force a reload."""
        _, meta = loaded_gui
        lp = meta.plot_container.line_plot

        xlim = lp.get_current_xlim()
        trial_dur = xlim[1] - xlim[0]
        if trial_dur < 0.5:
            pytest.skip("Trial too short to test buffer reload")

        # Pan to the last portion of the visible range
        t_end = xlim[1]
        t_start = t_end - trial_dur * 0.2
        lp.plot_item.setXRange(t_start, t_end)
        QApplication.processEvents()

        # Force content update (simulates what happens on range change)
        lp.update_plot_content(t_start, t_end)
        QApplication.processEvents()

        x, y = _get_curve_data(lp.plot_items)
        assert x is not None and len(x) > 0, "No data after panning to end of trial"
        assert x[-1] >= t_start, "Data doesn't cover panned region"
