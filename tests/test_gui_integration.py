import pytest
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication


class TestMetaWidgetCreation:

    def test_widget_initialization(self, gui):
        _, meta = gui
        for attr in (
            "app_state", "plot_container", "data_widget", "io_widget",
            "labels_widget", "navigation_widget", "changepoints_widget",
            "plot_settings_widget", "ephys_widget",
        ):
            assert getattr(meta, attr) is not None

        assert meta.app_state.ready is False
        assert meta.app_state.trial_changed is not None


class TestDataLoading:

    def test_state_after_load(self, loaded_gui):
        _, meta = loaded_gui
        state = meta.app_state

        assert state.ready is True
        assert state.dt is not None
        assert state.ds is not None
        assert len(state.trials) > 0
        assert state.trials_sel in state.trials
        assert state.time_coord is not None
        assert len(state.time_coord) > 0

        cat = meta.data_widget.catalog
        assert cat is not None
        assert len(cat.features) > 0

    def test_combos_populated_after_load(self, loaded_gui):
        _, meta = loaded_gui

        features_combo = meta.data_widget.combos.get("features")
        assert features_combo is not None
        assert features_combo.count() > 0

        ind_combo = meta.data_widget.combos.get("individuals")
        if ind_combo is None:
            ind_combo = meta.io_widget.combos.get("individuals")
        assert ind_combo is not None
        assert ind_combo.count() > 0

        assert meta.navigation_widget.trials_combo.count() > 0


class TestComboInteractions:

    def test_change_feature_selection(self, loaded_gui):
        _, meta = loaded_gui
        features_combo = meta.data_widget.combos["features"]
        if features_combo.count() < 2:
            pytest.skip("Need at least 2 features to test switching")

        features_combo.setCurrentIndex(1)
        QApplication.processEvents()

        expected = features_combo.currentText()
        assert meta.app_state.features_sel == expected

    def test_change_individual_selection(self, loaded_gui):
        _, meta = loaded_gui
        combo = meta.data_widget.combos.get("individuals")
        if combo is None or combo.count() < 2:
            pytest.skip("Need at least 2 individuals in data_widget.combos")

        combo.setCurrentIndex(1)
        QApplication.processEvents()

        expected = combo.currentText()
        assert meta.app_state.individuals_sel == expected

    def test_cycle_all_features(self, loaded_gui):
        _, meta = loaded_gui
        features_combo = meta.data_widget.combos["features"]

        for i in range(features_combo.count()):
            text = features_combo.itemText(i)
            if text in ("Spectrogram", "Waveform"):
                continue
            features_combo.setCurrentIndex(i)
            QApplication.processEvents()
            assert meta.app_state.features_sel == text

    def test_all_checkbox_sets_sel_to_none(self, loaded_gui):
        _, meta = loaded_gui
        if not meta.data_widget.all_checkboxes:
            pytest.skip("No 'All' checkboxes available")

        key = next(iter(meta.data_widget.all_checkboxes))
        checkbox = meta.data_widget.all_checkboxes[key]
        checkbox.setChecked(True)
        QApplication.processEvents()

        assert meta.app_state.get_key_sel(key) is None

    def test_uncheck_all_restores_combo_value(self, loaded_gui):
        _, meta = loaded_gui
        if not meta.data_widget.all_checkboxes:
            pytest.skip("No 'All' checkboxes available")

        key = next(iter(meta.data_widget.all_checkboxes))
        combo = meta.data_widget.combos[key]
        checkbox = meta.data_widget.all_checkboxes[key]

        original_text = combo.currentText()

        checkbox.setChecked(True)
        QApplication.processEvents()
        checkbox.setChecked(False)
        QApplication.processEvents()

        restored = meta.app_state.get_key_sel(key)
        assert restored == original_text


class TestTrialNavigation:

    def test_next_trial(self, loaded_gui):
        _, meta = loaded_gui
        if len(meta.app_state.trials) < 2:
            pytest.skip("Need at least 2 trials")

        meta.navigation_widget.mode_combo.setCurrentText("Trial")
        QApplication.processEvents()
        trials = meta.app_state.trials
        meta.navigation_widget.trials_combo.setCurrentText(str(trials[0]))
        QApplication.processEvents()

        first_trial = meta.app_state.trials_sel
        meta.navigation_widget.next_trial()
        QApplication.processEvents()

        assert meta.app_state.trials_sel != first_trial

    def test_prev_trial_at_start_stays(self, loaded_gui):
        _, meta = loaded_gui
        meta.navigation_widget.mode_combo.setCurrentText("Trial")
        QApplication.processEvents()
        first_trial = meta.app_state.trials[0]

        meta.navigation_widget.trials_combo.setCurrentText(str(first_trial))
        QApplication.processEvents()

        meta.navigation_widget.prev_trial()
        QApplication.processEvents()

        assert meta.app_state.trials_sel == first_trial

    def test_trial_combo_change_loads_correct_ds(self, loaded_gui):
        _, meta = loaded_gui
        if len(meta.app_state.trials) < 2:
            pytest.skip("Need at least 2 trials")

        second_trial = meta.app_state.trials[1]
        meta.navigation_widget.trials_combo.setCurrentText(str(second_trial))
        QApplication.processEvents()

        assert meta.app_state.trials_sel == second_trial
        assert meta.app_state.ds is not None


class TestLabelsWidget:

    def test_activate_label(self, loaded_gui):
        _, meta = loaded_gui
        meta.labels_widget.activate_label(1)
        assert meta.labels_widget.ready_for_label_click is True
        assert meta.labels_widget.selected_labels == 1

    def test_label_creation_via_two_clicks(self, loaded_gui):
        _, meta = loaded_gui
        from qtpy.QtCore import Qt
        from ethograph.labels.intervals import find_interval_at

        labels = 1
        t_start = 1.0
        t_end = 2.0

        meta.labels_widget.activate_label(labels)
        meta.labels_widget._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
        assert meta.labels_widget.first_click == pytest.approx(t_start)

        meta.labels_widget._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
        QApplication.processEvents()

        df = meta.app_state.label_intervals
        assert df is not None and not df.empty, "No intervals after label creation"

        individual = meta.labels_widget._current_individual()
        idx = find_interval_at(df, (t_start + t_end) / 2, individual)
        assert idx is not None, "Interval not found at midpoint"

        row = df.loc[idx]
        assert row["labels"] == labels
        assert row["onset_s"] == pytest.approx(t_start, abs=0.01)
        assert row["offset_s"] == pytest.approx(t_end, abs=0.01)

    def test_human_verification_single_trial(self, loaded_gui):
        _, meta = loaded_gui
        meta.io_widget._human_verification_true(mode="single_trial")
        QApplication.processEvents()

        df = meta.app_state._all_labels_df
        trial = meta.app_state.trials_sel
        if df is not None and not df.empty and "human_verified" in df.columns:
            trial_rows = df[df["trial"] == trial]
            assert (trial_rows["human_verified"] == 1).all()

    def test_changes_saved_initially_true(self, loaded_gui):
        _, meta = loaded_gui
        assert meta.app_state.changes_saved is True


class TestDownsampledData:

    DOWNSAMPLE_FACTOR = 100

    def test_downsample_by_100(self, loaded_gui_downsampled):
        _, meta = loaded_gui_downsampled

        # State and attributes
        assert meta.app_state.ready is True
        assert meta.app_state.downsample_factor_used == self.DOWNSAMPLE_FACTOR
        assert meta.app_state.dt is not None

        attrs = meta.app_state.ds.attrs
        assert attrs["downsample_factor"] == self.DOWNSAMPLE_FACTOR

        # min-max envelope: output_len = (original // factor) * 2
        # so original = output_len / 2 * factor, which must be >> output_len
        n_time = len(meta.app_state.time_coord.values)
        original_approx = (n_time // 2) * self.DOWNSAMPLE_FACTOR
        assert n_time < original_approx

        assert not meta.io_widget.downsample_checkbox.isEnabled()
        assert not meta.io_widget.downsample_spin.isEnabled()

        # Combo interactions
        features_combo = meta.data_widget.combos.get("features")
        assert features_combo is not None
        assert features_combo.count() > 0

        for i in range(features_combo.count()):
            text = features_combo.itemText(i)
            if text in ("Spectrogram", "Waveform"):
                continue
            features_combo.setCurrentIndex(i)
            QApplication.processEvents()
            assert meta.app_state.features_sel == text

        # Plot
        assert meta.plot_container.is_lineplot()
        xlim = meta.plot_container.get_current_xlim()
        assert xlim[0] < xlim[1]


class TestPlotContainer:

    def test_default_lineplot_with_valid_range(self, loaded_gui):
        _, meta = loaded_gui
        assert meta.plot_container.is_lineplot()

        xlim = meta.plot_container.get_current_xlim()
        assert len(xlim) == 2
        assert xlim[0] < xlim[1]


class TestHiddenPanelsNoData:
    """Hidden panels must not load data or hold sources/loaders."""

    def _toggle_panel(self, meta, name, checked):
        cb = getattr(meta.data_widget, f"{name}_checkbox")
        cb.setChecked(checked)
        QApplication.processEvents()

    # -- Audio trace --

    def test_audiotrace_hidden_clears_source(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container

        assert pc._panel_visible["audiotrace"]
        assert pc.audio_trace_plot.source is not None

        self._toggle_panel(meta, "audiotrace", False)

        assert not pc._panel_visible["audiotrace"]
        assert pc.audio_trace_plot.source is None

    def test_audiotrace_show_restores_source(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container

        self._toggle_panel(meta, "audiotrace", False)
        assert pc.audio_trace_plot.source is None

        self._toggle_panel(meta, "audiotrace", True)
        assert pc.audio_trace_plot.source is not None

    def test_audiotrace_hidden_no_update_on_xrange(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container

        self._toggle_panel(meta, "audiotrace", False)
        assert pc.audio_trace_plot.source is None

        # Simulate x-range change — should be a no-op
        pc.audio_trace_plot._on_view_range_changed()
        assert pc.audio_trace_plot.source is None

    # -- Spectrogram --

    def test_spectrogram_hidden_clears_source(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container

        assert pc._panel_visible["spectrogram"]
        assert pc.spectrogram_plot.source is not None

        self._toggle_panel(meta, "spectrogram", False)

        assert not pc._panel_visible["spectrogram"]
        assert pc.spectrogram_plot.source is None

    def test_spectrogram_show_restores_source(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container

        self._toggle_panel(meta, "spectrogram", False)
        assert pc.spectrogram_plot.source is None

        self._toggle_panel(meta, "spectrogram", True)
        assert pc.spectrogram_plot.source is not None

    def test_spectrogram_hidden_no_update_on_xrange(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container

        self._toggle_panel(meta, "spectrogram", False)
        pc.spectrogram_plot._on_view_range_changed()
        assert pc.spectrogram_plot.source is None

    # -- Neo (ephys) --

    def test_neo_hidden_clears_loader(self, loaded_gui):
        _, meta = loaded_gui
        pc = meta.plot_container

        from unittest.mock import MagicMock
        fake_loader = MagicMock()
        fake_loader.rate = 30000.0
        fake_loader.n_channels = 4
        fake_loader.__len__ = lambda self: 1000

        pc.neo_trace_plot.buffer.loader = fake_loader
        pc._panel_visible["neo"] = True

        pc.set_neo_visible(False)

        assert not pc._panel_visible["neo"]
        assert pc.neo_trace_plot.buffer.loader is None
        assert pc.neo_trace_plot._source is None

    def test_neo_hidden_no_update_on_xrange(self, loaded_gui):
        _, meta = loaded_gui
        pc = meta.plot_container

        pc._panel_visible["neo"] = False
        pc.neo_trace_plot.hide()

        pc.neo_trace_plot._on_view_range_changed()

    # -- Ephys (Phy) --

    def test_ephys_hidden_clears_loader(self, loaded_gui):
        _, meta = loaded_gui
        pc = meta.plot_container

        from unittest.mock import MagicMock
        fake_loader = MagicMock()
        fake_loader.rate = 30000.0
        fake_loader.n_channels = 4
        fake_loader.__len__ = lambda self: 1000

        pc.ephys_trace_plot.buffer.loader = fake_loader
        pc._panel_visible["ephys"] = True

        pc.set_ephys_visible(False)

        assert not pc._panel_visible["ephys"]
        assert pc.ephys_trace_plot.buffer.loader is None
        assert pc.ephys_trace_plot._source is None

    # -- Feature plot --

    def test_featureplot_hidden_no_update_on_xrange(self, loaded_gui):
        _, meta = loaded_gui
        pc = meta.plot_container

        self._toggle_panel(meta, "featureplot", False)
        pc.line_plot._on_view_range_changed()

    # -- update_audio_panels respects visibility --

    def test_update_audio_panels_skips_hidden(self, no_video_gui):
        _, meta = no_video_gui
        pc = meta.plot_container

        self._toggle_panel(meta, "audiotrace", False)
        self._toggle_panel(meta, "spectrogram", False)

        assert pc.audio_trace_plot.source is None
        assert pc.spectrogram_plot.source is None

        pc.update_audio_panels()

        assert pc.audio_trace_plot.source is None
        assert pc.spectrogram_plot.source is None
