import pytest
import numpy as np
from qtpy.QtWidgets import QApplication


class TestMetaWidgetCreation:

    def test_widget_initialization(self, gui):
        _, meta = gui
        for attr in (
            "app_state", "plot_container", "data_widget", "io_widget",
            "labels_widget", "navigation_widget", "changepoints_widget",
            "axes_widget", "audio_widget",
        ):
            assert getattr(meta, attr) is not None

        assert meta.app_state.ready is False
        assert meta.app_state.labels_modified is not None
        assert meta.app_state.trial_changed is not None
        assert meta.app_state.verification_changed is not None


class TestDataLoading:

    def test_state_after_load(self, loaded_gui):
        _, meta = loaded_gui
        state = meta.app_state

        assert state.ready is True
        assert state.dt is not None
        assert state.ds is not None
        assert state.label_dt is not None
        assert state.label_ds is not None
        assert len(state.trials) > 0
        assert state.trials_sel in state.trials
        assert state.time is not None
        assert len(state.time) > 0
        assert state.label_sr is not None
        assert state.label_sr > 0

        tvd = meta.data_widget.type_vars_dict
        assert "features" in tvd
        assert len(tvd["features"]) > 0

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
        assert features_combo.count() >= 2, "Need at least 2 features to test switching"

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

        first_trial = meta.app_state.trials_sel
        meta.navigation_widget.next_trial()
        QApplication.processEvents()

        assert meta.app_state.trials_sel != first_trial

    def test_prev_trial_at_start_stays(self, loaded_gui):
        _, meta = loaded_gui
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
        assert meta.labels_widget.selected_labels_id == 1

    def test_label_creation_via_two_clicks(self, loaded_gui):
        _, meta = loaded_gui
        from qtpy.QtCore import Qt
        from ethograph.utils.data_utils import sel_valid, get_time_coord

        label_id = 1
        label_sr = meta.app_state.label_sr
        ds_kwargs = meta.app_state.get_ds_kwargs()

        labels_before, _ = sel_valid(meta.app_state.label_ds.labels, ds_kwargs)
        n = len(labels_before)

        start_idx = max(10, n // 10)
        end_idx = min(n - 10, n // 5)
        assert end_idx - start_idx > 10, "Not enough samples for label creation test"

        # Zero out the target region to ensure a clean slate
        labels_before[start_idx:end_idx + 1] = 0

        # Convert label indices to time (simulates where user clicks on plot)
        t_start = start_idx / label_sr
        t_end = end_idx / label_sr

        # Activate label and simulate two clicks
        meta.labels_widget.activate_label(label_id)
        meta.labels_widget._on_plot_clicked({"x": t_start, "button": Qt.LeftButton})
        assert meta.labels_widget.first_click == start_idx

        meta.labels_widget._on_plot_clicked({"x": t_end, "button": Qt.LeftButton})
        QApplication.processEvents()

        # Verify label was written into label_dt
        labels_after, _ = sel_valid(meta.app_state.label_ds.labels, ds_kwargs)
        assert np.all(labels_after[start_idx:end_idx + 1] == label_id), (
            f"Expected label {label_id} in range [{start_idx}:{end_idx}]"
        )

        # Boundaries must remain untouched
        if start_idx > 0:
            assert labels_after[start_idx - 1] == 0
        if end_idx + 1 < n:
            assert labels_after[end_idx + 1] == 0

        # Verify time-to-index alignment via label time coordinate
        time_coord = get_time_coord(meta.app_state.label_ds.labels).values
        labeled_mask = labels_after == label_id
        labeled_times = time_coord[labeled_mask]
        assert labeled_times[0] == pytest.approx(time_coord[start_idx], abs=1e-6)
        assert labeled_times[-1] == pytest.approx(time_coord[end_idx], abs=1e-6)

    def test_human_verification_single_trial(self, loaded_gui):
        _, meta = loaded_gui
        meta.labels_widget._human_verification_true(mode="single_trial")
        QApplication.processEvents()

        trial = meta.app_state.trials_sel
        attrs = meta.app_state.label_dt.trial(trial).attrs
        assert attrs.get("human_verified") == np.int8(1)

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
        assert meta.app_state.label_dt is not None

        attrs = meta.app_state.ds.attrs
        assert attrs["downsample_factor"] == self.DOWNSAMPLE_FACTOR

        # min-max envelope: output_len = (original // factor) * 2
        # so original = output_len / 2 * factor, which must be >> output_len
        n_time = len(meta.app_state.time)
        original_approx = (n_time // 2) * self.DOWNSAMPLE_FACTOR
        assert n_time < original_approx

        assert meta.app_state.label_sr is not None
        assert meta.app_state.label_sr > 0

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
