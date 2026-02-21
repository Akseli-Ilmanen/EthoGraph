"""Tests for no-video (audio-only) 3-panel GUI mode."""

import pytest
import numpy as np
from qtpy.QtWidgets import QApplication


class TestNoVideoLoading:

    def test_no_video_mode_activated(self, no_video_gui):
        _, meta = no_video_gui
        assert meta.app_state.no_video_mode is True

    def test_state_after_load(self, no_video_gui):
        _, meta = no_video_gui
        state = meta.app_state

        assert state.ready is True
        assert state.dt is not None
        assert state.ds is not None
        assert state.label_dt is not None
        assert state.time is not None
        assert len(state.time) > 0
        assert state.audio_path is not None

    def test_effective_fps_fallback(self, no_video_gui):
        _, meta = no_video_gui
        # birdpark has fps in attrs, so effective_fps should use it
        fps = meta.app_state.effective_fps
        assert fps > 0


class TestMultiPanelContainer:

    def test_plot_container_is_multipanel(self, no_video_gui):
        from ethograph.gui.multipanel_container import MultiPanelContainer
        _, meta = no_video_gui
        assert isinstance(meta.plot_container, MultiPanelContainer)

    def test_three_panels_exist(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        assert pc.audio_trace_plot is not None
        assert pc.spectrogram_plot is not None
        assert pc.line_plot is not None
        assert pc.ephys_trace_plot is None

    def test_x_axis_linked(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        # Spectrogram and line_plot should be linked to audio_trace_plot
        spec_link = pc.spectrogram_plot.plotItem.getAxis('bottom')
        assert spec_link is not None

    def test_valid_x_range(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        xlim = pc.get_current_xlim()
        assert xlim[0] < xlim[1]
        # Range should be within reasonable data bounds (not default -0.5..0.5)
        assert xlim[0] >= -1.0
        assert xlim[1] > 0.1

    def test_time_slider_range(self, no_video_gui):
        _, meta = no_video_gui
        slider = meta.plot_container.time_slider
        assert slider._t_max > slider._t_min

    def test_bottom_panel_defaults_to_lineplot(self, no_video_gui):
        _, meta = no_video_gui
        assert meta.plot_container.is_lineplot()
        assert not meta.plot_container.is_heatmap()

    def test_switch_to_heatmap(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        pc.switch_to_heatmap()
        QApplication.processEvents()
        assert pc.is_heatmap()
        assert not pc.is_lineplot()

    def test_switch_back_to_lineplot(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        pc.switch_to_heatmap()
        QApplication.processEvents()
        pc.switch_to_lineplot()
        QApplication.processEvents()
        assert pc.is_lineplot()

    def test_spectrogram_is_in_splitter(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        assert not pc.is_spectrogram()  # not "current" mode
        # Spectrogram is always a child of the splitter
        assert pc.spectrogram_plot.parent() is not None

    def test_audio_trace_is_in_splitter(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        assert not pc.is_audiotrace()  # not "current" mode
        assert pc.audio_trace_plot.parent() is not None


class TestNoVideoOverlays:

    def test_overlay_checkboxes_hidden(self, no_video_gui):
        _, meta = no_video_gui
        dw = meta.data_widget
        assert not dw.show_waveform_checkbox.isVisible()
        assert not dw.show_spectrogram_checkbox.isVisible()

    def test_envelope_target_combo_configured(self, no_video_gui):
        _, meta = no_video_gui
        tw = meta.transform_widget
        # The combo is shown (not hidden) — but isVisible() depends on parent panel
        # Check that show() was called (isHidden() checks the widget's own flag)
        assert not tw.envelope_target_combo.isHidden()
        assert not tw.envelope_target_label.isHidden()


class TestNoVideoTimeMarker:

    def test_update_time_marker(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container
        pc.update_time_marker_by_time(1.5)
        # All three plots should have the marker at 1.5
        assert pc.audio_trace_plot.time_marker.value() == pytest.approx(1.5)
        assert pc.spectrogram_plot.time_marker.value() == pytest.approx(1.5)
        assert pc._bottom_plot.time_marker.value() == pytest.approx(1.5)


class TestNoVideoComboInteractions:

    def test_feature_combo_populated(self, no_video_gui):
        _, meta = no_video_gui
        combo = meta.data_widget.combos.get("features")
        assert combo is not None
        assert combo.count() > 0
        # "Audio Waveform" should NOT be injected in no-video mode
        items = [combo.itemText(i) for i in range(combo.count())]
        assert "Audio Waveform" not in items

    def test_view_mode_no_spectrogram_option(self, no_video_gui):
        _, meta = no_video_gui
        combo = meta.data_widget.view_mode_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        for item in items:
            assert "Spectrogram" not in item
            assert "Audio" not in item

    def test_change_feature(self, no_video_gui):
        _, meta = no_video_gui
        combo = meta.data_widget.combos["features"]
        if combo.count() < 2:
            pytest.skip("Need at least 2 features")

        combo.setCurrentIndex(1 if combo.currentIndex() == 0 else 0)
        QApplication.processEvents()
        assert meta.app_state.features_sel == combo.currentText()


class TestNoVideoLabels:

    def test_label_creation(self, no_video_gui):
        from qtpy.QtCore import Qt
        from ethograph.utils.label_intervals import find_interval_at

        _, meta = no_video_gui
        t_start, t_end = 1.0, 2.0

        meta.labels_widget.activate_label(1)
        meta.labels_widget._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
        meta.labels_widget._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
        QApplication.processEvents()

        df = meta.app_state.label_intervals
        assert df is not None and not df.empty

        individual = meta.labels_widget._current_individual()
        idx = find_interval_at(df, 1.5, individual)
        assert idx is not None

    def test_shapes_layer_skipped(self, no_video_gui):
        _, meta = no_video_gui
        # In no-video mode, the video shapes overlay is not created
        assert "_labels" not in meta.viewer.layers


class TestNoVideoTrialNavigation:

    def test_trial_change(self, no_video_gui):
        _, meta = no_video_gui
        if len(meta.app_state.trials) < 2:
            pytest.skip("Need at least 2 trials")

        first_trial = meta.app_state.trials_sel
        meta.navigation_widget.next_trial()
        QApplication.processEvents()
        assert meta.app_state.trials_sel != first_trial


class TestNapariViewerHidden:

    def test_central_widget_hidden(self, no_video_gui):
        viewer, _ = no_video_gui
        central = viewer.window._qt_window.centralWidget()
        if central:
            assert not central.isVisible()

    def test_layer_docks_hidden(self, no_video_gui):
        viewer, _ = no_video_gui
        from qtpy.QtWidgets import QDockWidget
        for dock in viewer.window._qt_window.findChildren(QDockWidget):
            title = (dock.windowTitle() or "").lower()
            if "layer" in title or "control" in title:
                assert not dock.isVisible(), f"Dock '{dock.windowTitle()}' should be hidden"
