import pytest
import numpy as np
from qtpy.QtWidgets import QApplication


class TestMetaWidgetCreation:

    def test_all_widgets_exist(self, gui):
        _, meta = gui
        assert meta.app_state is not None
        assert meta.plot_container is not None
        assert meta.data_widget is not None
        assert meta.io_widget is not None
        assert meta.labels_widget is not None
        assert meta.navigation_widget is not None
        assert meta.changepoints_widget is not None
        assert meta.axes_widget is not None
        assert meta.audio_widget is not None

    def test_app_state_not_ready_initially(self, gui):
        _, meta = gui
        assert meta.app_state.ready is False

    def test_signals_exist(self, gui):
        _, meta = gui
        assert meta.app_state.labels_modified is not None
        assert meta.app_state.trial_changed is not None
        assert meta.app_state.verification_changed is not None


class TestDataLoading:

    def test_app_state_ready_after_load(self, loaded_gui):
        _, meta = loaded_gui
        assert meta.app_state.ready is True

    def test_datatree_loaded(self, loaded_gui):
        _, meta = loaded_gui
        assert meta.app_state.dt is not None
        assert meta.app_state.ds is not None

    def test_label_dt_loaded(self, loaded_gui):
        _, meta = loaded_gui
        assert meta.app_state.label_dt is not None
        assert meta.app_state.label_ds is not None

    def test_trials_populated(self, loaded_gui):
        _, meta = loaded_gui
        assert len(meta.app_state.trials) > 0

    def test_trials_sel_set(self, loaded_gui):
        _, meta = loaded_gui
        assert meta.app_state.trials_sel is not None
        assert meta.app_state.trials_sel in meta.app_state.trials

    def test_time_coord_set(self, loaded_gui):
        _, meta = loaded_gui
        assert meta.app_state.time is not None
        assert len(meta.app_state.time) > 0

    def test_label_sr_set(self, loaded_gui):
        _, meta = loaded_gui
        assert meta.app_state.label_sr is not None
        assert meta.app_state.label_sr > 0

    def test_features_combo_populated(self, loaded_gui):
        _, meta = loaded_gui
        features_combo = meta.data_widget.combos.get("features")
        assert features_combo is not None
        assert features_combo.count() > 0

    def test_individuals_combo_populated(self, loaded_gui):
        _, meta = loaded_gui
        combo = meta.data_widget.combos.get("individuals")
        if combo is None:
            combo = meta.io_widget.combos.get("individuals")
        assert combo is not None
        assert combo.count() > 0

    def test_trials_combo_populated(self, loaded_gui):
        _, meta = loaded_gui
        trials_combo = meta.navigation_widget.trials_combo
        assert trials_combo.count() > 0

    def test_type_vars_dict_populated(self, loaded_gui):
        _, meta = loaded_gui
        tvd = meta.data_widget.type_vars_dict
        assert "features" in tvd
        assert len(tvd["features"]) > 0


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

    def test_cycle_all_trials(self, loaded_gui):
        _, meta = loaded_gui
        for trial in meta.app_state.trials:
            meta.navigation_widget.trials_combo.setCurrentText(str(trial))
            QApplication.processEvents()
            assert meta.app_state.trials_sel == trial
            assert meta.app_state.ds is not None


class TestLabelsWidget:

    def test_activate_label(self, loaded_gui):
        _, meta = loaded_gui
        meta.labels_widget.activate_label(1)
        assert meta.labels_widget.ready_for_label_click is True
        assert meta.labels_widget.selected_labels_id == 1

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


class TestPlotContainer:

    def test_lineplot_is_default(self, loaded_gui):
        _, meta = loaded_gui
        assert meta.plot_container.is_lineplot()

    def test_get_current_xlim_returns_valid_range(self, loaded_gui):
        _, meta = loaded_gui
        xlim = meta.plot_container.get_current_xlim()
        assert len(xlim) == 2
        assert xlim[0] < xlim[1]
