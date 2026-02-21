"""Widget for selecting start/stop times and playing a segment in napari."""

import os
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr
from movement.napari.loader_widgets import DataLoader
from napari.utils.notifications import show_error, show_warning
from napari.viewer import Viewer
from napari_pyav._reader import FastVideoReader
from qtpy.QtCore import QSortFilterProxyModel, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QWidget,
)

from ethograph import downsample_trialtree
from ethograph.utils.data_utils import get_time_coord, sel_valid

from .app_constants import DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_SPACING
from .data_loader import load_dataset
from .plots_space import SpacePlot
from .plots_spectrogram import SharedAudioCache
from .video_sync import NapariVideoSync


def make_searchable(combo_box: QComboBox) -> None:
    combo_box.setFocusPolicy(Qt.StrongFocus)
    combo_box.setEditable(True)
    combo_box.setInsertPolicy(QComboBox.NoInsert)

    filter_model = QSortFilterProxyModel(combo_box)
    filter_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
    filter_model.setSourceModel(combo_box.model())

    completer = QCompleter(filter_model, combo_box)
    completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
    combo_box.setCompleter(completer)
    combo_box.lineEdit().textEdited.connect(filter_model.setFilterFixedString)


class DataWidget(DataLoader, QWidget):
    """Orchestrator widget — loads data, manages selections, updates plots."""

    def __init__(
        self,
        napari_viewer: Viewer,
        app_state,
        meta_widget,
        io_widget,
        parent=None,
    ):
        DataLoader.__init__(self, napari_viewer)
        QWidget.__init__(self, parent=parent)
        self.viewer = napari_viewer
        layout = QFormLayout()
        layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        layout.setContentsMargins(DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN)
        self.setLayout(layout)
        self.app_state = app_state
        self.meta_widget = meta_widget
        self.io_widget = io_widget
        self.video = None
        self.plot_container = None
        self.labels_widget = None
        self.plot_settings_widget = None
        self.transform_widget = None
        self.ephys_widget = None
        self.audio_player = None
        self.video_path = None
        self.audio_path = None
        self.space_plot = None

        self.combos = {}
        self.all_checkboxes = {}
        self.controls = []

        self.fps = None
        self.source_software = None
        self.file_path = None
        self.file_name = None

        self.app_state.audio_video_sync = None
        self.type_vars_dict = {}

    def set_references(
        self, plot_container, labels_widget, plot_settings_widget,
        navigation_widget, transform_widget=None, changepoints_widget=None,
        ephys_widget=None,
    ):
        self.plot_container = plot_container
        self.labels_widget = labels_widget
        self.plot_settings_widget = plot_settings_widget
        self.navigation_widget = navigation_widget
        self.transform_widget = transform_widget
        self.changepoints_widget = changepoints_widget
        self.ephys_widget = ephys_widget

        if changepoints_widget is not None:
            changepoints_widget.request_plot_update.connect(self._on_plot_update_request)

    def _on_plot_update_request(self):
        if not self.app_state.ready or not self.plot_container:
            return
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _cleanup_load_state(self):
        dt = getattr(self.app_state, 'dt', None)
        if dt is not None:
            dt.close()
            self.app_state.dt = None
        self.app_state.ds = None
        self.app_state.label_dt = None
        self.app_state.label_ds = None
        self.type_vars_dict = {}
        self.app_state.ready = False

    def _cancel_load(self, reason: str):
        QMessageBox.warning(self, "Load cancelled", reason)
        self._cleanup_load_state()

    def on_load_clicked(self):
        if not self.app_state.nc_file_path:
            QMessageBox.warning(self, "Load cancelled", "Please select a path ending with .nc")
            return

        nc_file_path = self.io_widget.get_nc_file_path()

        has_video = bool(self.app_state.video_folder)
        has_pose = bool(self.app_state.pose_folder)
        require_fps = has_video or has_pose
        require_cameras = has_video

        try:
            self.app_state.dt, label_dt, self.type_vars_dict = load_dataset(
                nc_file_path,
                require_fps=require_fps,
                require_cameras=require_cameras,
            )
        except Exception:
            self._cleanup_load_state()
            return

        self.app_state.trial_conditions = self.type_vars_dict["trial_conditions"]

        has_audio = "mics" in self.type_vars_dict and self.app_state.audio_folder
        no_video = bool(has_audio and not has_video)
        self.app_state.no_video_mode = no_video

        if has_audio and "features" in self.type_vars_dict:
            features_list = list(self.type_vars_dict["features"])
            self.type_vars_dict["features"] = features_list + ["Audio Waveform"]

        if self.app_state.ephys_folder and "features" in self.type_vars_dict:
            try:
                ephys_features = self.io_widget._expand_ephys_with_streams(
                    self.app_state.ephys_folder, self.app_state.ds,
                )
            except Exception as e:
                self._cancel_load(f"Failed to load ephys features: {e}")
                return
            if ephys_features:
                features_list = list(self.type_vars_dict["features"])
                self.type_vars_dict["features"] = features_list + ephys_features

        downsample_factor = self.io_widget.get_downsample_factor()
        if downsample_factor is not None:
            self.app_state.dt = downsample_trialtree(self.app_state.dt, downsample_factor)
            self.app_state.downsample_factor_used = downsample_factor
            print(f"Downsampled data by factor {downsample_factor}")
        else:
            self.app_state.downsample_factor_used = None

        self.io_widget.disable_downsample_controls()

        trials = self.app_state.dt.trials

        if self.app_state.import_labels_nc_data:
            self.app_state.label_dt = label_dt
        else:
            self.app_state.label_dt = self.app_state.dt.get_label_dt(empty=True)

        self.app_state.ds = self.app_state.dt.trial(trials[0])
        self.app_state.label_ds = self.app_state.label_dt.trial(trials[0])
        self.app_state.trials = sorted(trials)

        missing = self._validate_media_files(self.app_state.ds)
        if missing:
            self._cancel_load(
                "Missing media files for first trial:\n" + "\n".join(missing)
            )
            return

        self._create_trial_controls()

        self._restore_or_set_defaults()
        self._set_controls_enabled(True)
        self.app_state.ready = True

        load_btn = self.io_widget.load_button
        load_btn.setEnabled(False)
        self.io_widget.create_nc_button.setEnabled(False)
        self.io_widget.template_button.setEnabled(False)
        self.changepoints_widget.setEnabled(True)
        self.plot_settings_widget.set_enabled_state(has_audio=False)
        if self.transform_widget:
            self.transform_widget.setEnabled(True)
            if no_video:
                self.transform_widget.show_envelope_target_combo()
        if self.ephys_widget:
            self.ephys_widget.setEnabled(True)
            self.ephys_widget.populate_ephys_default_path()

        # Switch to 3-panel layout when no video is provided
        if no_video:
            self.meta_widget.switch_to_no_video_layout()

        trial = self.app_state.trials_sel
        try:
            is_nan = np.isnan(trial)
        except (TypeError, ValueError):
            is_nan = False
        if not trial or is_nan:
            self.app_state.trials_sel = self.app_state.trials[0]

        self.update_trials_combo()
        self._load_trial_with_fallback()

        self.view_mode_combo.show()
        load_btn.setText("Restart app to load new data")

    # ------------------------------------------------------------------
    # Trials combo
    # ------------------------------------------------------------------

    def update_trials_combo(self) -> None:
        if not self.app_state.ready:
            return

        combo = self.navigation_widget.trials_combo
        combo.blockSignals(True)
        combo.clear()

        trial_status = self._collect_trial_status()

        for trial in self.app_state.trials:
            combo.addItem(str(trial))
            index = combo.count() - 1
            is_verified = trial_status.get(trial)
            bg_color = QColor(144, 238, 144) if is_verified else QColor(255, 182, 193)
            combo.setItemData(index, bg_color, Qt.BackgroundRole)
            text_color = QColor(0, 100, 0) if is_verified else QColor(139, 0, 0)
            combo.setItemData(index, text_color, Qt.ForegroundRole)

        combo.setCurrentText(str(self.app_state.trials_sel))
        combo.blockSignals(False)

    def _collect_trial_status(self) -> Dict[int, int]:
        trial_status = {}
        for trial in self.app_state.trials:
            is_verified = self.app_state.label_dt.trial(trial).attrs.get('human_verified', 0)
            trial_status[trial] = bool(is_verified)
        return trial_status

    # ------------------------------------------------------------------
    # Create controls (placed inside transform_widget Main tab)
    # ------------------------------------------------------------------

    def _create_trial_controls(self):
        self.io_widget.create_device_controls(self.type_vars_dict)
        self.navigation_widget.setup_trial_conditions(self.type_vars_dict)
        self.navigation_widget.set_data_widget(self)

        # Use transform_widget's coords groupbox for xarray combos
        tw = self.transform_widget
        self.coords_groupbox = tw.coords_groupbox
        self.coords_groupbox_layout = tw.coords_groupbox_layout

        non_data_type_vars = ["cameras", "mics", "pose", "trial_conditions", "changepoints", "rgb"]

        for type_var in self.type_vars_dict.keys():
            if type_var.lower() not in non_data_type_vars:
                self._create_combo_widget(type_var, self.type_vars_dict[type_var])

        # View mode combo — placed in transform_widget's view_space_layout
        view_space_layout = tw.view_space_layout

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.setObjectName("view_mode_combo")
        if self.app_state.no_video_mode:
            # Spectrogram is always visible in its own panel
            self.view_mode_combo.addItems([
                "LinePlot (N-dim)",
                "Heatmap (N-dim)",
            ])
        else:
            self.view_mode_combo.addItems([
                "LinePlot (N-dim)",
                "Spectrogram (1-dim)",
                "Heatmap (N-dim)",
            ])
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        self.view_mode_combo.hide()
        self.controls.append(self.view_mode_combo)
        view_space_layout.addWidget(self.view_mode_combo)

        self.space_plot_combo = QComboBox()
        self.space_plot_combo.setObjectName("space_plot_combo")
        if 'position' in self.app_state.ds.data_vars:
            self.space_plot_combo.addItems(["Layer controls", "space_2D", "space_3D"])
            self.space_plot_combo.currentTextChanged.connect(self._on_space_plot_changed)
        else:
            self.space_plot_combo.addItems(["Layer controls"])
        self.controls.append(self.space_plot_combo)
        view_space_layout.addWidget(self.space_plot_combo)

        if 'pose' in self.type_vars_dict:
            hide_label = QLabel("Hide markers:")
            hide_label.setToolTip("Hide pose markers with confidence below this value (0.0-1.0)")
            self.pose_hide_threshold_spin = QDoubleSpinBox()
            self.pose_hide_threshold_spin.setObjectName("pose_hide_threshold_spin")
            self.pose_hide_threshold_spin.setRange(0.0, 1.0)
            self.pose_hide_threshold_spin.setSingleStep(0.1)
            self.pose_hide_threshold_spin.setDecimals(1)
            self.pose_hide_threshold_spin.setFixedWidth(60)
            self.pose_hide_threshold_spin.setToolTip("Hide pose markers with confidence below this value (0.0-1.0)")
            self.pose_hide_threshold_spin.setValue(self.app_state.pose_hide_threshold)
            self.pose_hide_threshold_spin.valueChanged.connect(self._on_pose_hide_threshold_changed)
            view_space_layout.addWidget(hide_label)
            view_space_layout.addWidget(self.pose_hide_threshold_spin)

        view_space_layout.addStretch()

        # Overlay checkboxes — placed in transform_widget's overlay_layout
        overlay_layout = tw.overlay_layout

        self.show_confidence_checkbox = QCheckBox("Confidence")
        self.show_confidence_checkbox.setChecked(False)
        self.show_confidence_checkbox.stateChanged.connect(self.refresh_lineplot)
        overlay_layout.addWidget(self.show_confidence_checkbox)

        self.show_envelope_checkbox = QCheckBox("Envelope")
        self.show_envelope_checkbox.setChecked(False)
        self.show_envelope_checkbox.stateChanged.connect(self._on_envelope_overlay_changed)
        self.show_envelope_checkbox.hide()
        overlay_layout.addWidget(self.show_envelope_checkbox)

        self.show_waveform_checkbox = QCheckBox("Waveform (audio)")
        self.show_waveform_checkbox.setChecked(False)
        self.show_waveform_checkbox.stateChanged.connect(self._on_audio_overlay_changed)
        self.show_waveform_checkbox.hide()
        overlay_layout.addWidget(self.show_waveform_checkbox)

        self.show_spectrogram_checkbox = QCheckBox("Spectrogram (audio)")
        self.show_spectrogram_checkbox.setChecked(False)
        self.show_spectrogram_checkbox.stateChanged.connect(self._on_audio_overlay_changed)
        self.show_spectrogram_checkbox.hide()
        overlay_layout.addWidget(self.show_spectrogram_checkbox)

        overlay_layout.addStretch()

        # In no-video mode, audio waveform/spectrogram have dedicated panels
        # so overlay checkboxes are not needed.
        if self.app_state.no_video_mode:
            self.show_waveform_checkbox.setVisible(False)
            self.show_spectrogram_checkbox.setVisible(False)

        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    # Combo / checkbox handlers
    # ------------------------------------------------------------------

    def refresh_lineplot(self):
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def cycle_view_mode(self):
        if not hasattr(self, 'view_mode_combo') or not self.view_mode_combo.isVisible():
            return
        next_index = (self.view_mode_combo.currentIndex() + 1) % self.view_mode_combo.count()
        self.view_mode_combo.setCurrentIndex(next_index)

    def _on_audio_overlay_changed(self):
        sender = self.sender()
        if sender == self.show_waveform_checkbox and self.show_waveform_checkbox.isChecked():
            self.show_spectrogram_checkbox.blockSignals(True)
            self.show_spectrogram_checkbox.setChecked(False)
            self.show_spectrogram_checkbox.blockSignals(False)
        elif sender == self.show_spectrogram_checkbox and self.show_spectrogram_checkbox.isChecked():
            self.show_waveform_checkbox.blockSignals(True)
            self.show_waveform_checkbox.setChecked(False)
            self.show_waveform_checkbox.blockSignals(False)
        self._update_audio_overlay()

    def _update_audio_overlay(self):
        if not self.plot_container:
            return
        if self.show_waveform_checkbox.isChecked():
            self.plot_container.show_audio_overlay('waveform')
        elif self.show_spectrogram_checkbox.isChecked():
            self.plot_container.show_audio_overlay('spectrogram')
        else:
            self.plot_container.hide_audio_overlay()
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _update_view_mode_items(self, feature_sel: str):
        """Update view_mode_combo items based on selected feature.

        When Audio Waveform is selected, LinePlot is not applicable — the
        dedicated AudioTracePlot panel already shows the waveform.  Only
        Heatmap (and Spectrogram in video mode) remain as options.
        """
        is_audio = feature_sel == "Audio Waveform"
        current_text = self.view_mode_combo.currentText()

        self.view_mode_combo.blockSignals(True)
        self.view_mode_combo.clear()

        if self.app_state.no_video_mode:
            if is_audio:
                self.view_mode_combo.addItems(["Heatmap (N-dim)"])
            else:
                self.view_mode_combo.addItems(["LinePlot (N-dim)", "Heatmap (N-dim)"])
        else:
            if is_audio:
                self.view_mode_combo.addItems(["Spectrogram (1-dim)", "Heatmap (N-dim)"])
            else:
                self.view_mode_combo.addItems([
                    "LinePlot (N-dim)", "Spectrogram (1-dim)", "Heatmap (N-dim)",
                ])

        idx = self.view_mode_combo.findText(current_text)
        if idx >= 0:
            self.view_mode_combo.setCurrentIndex(idx)
        self.view_mode_combo.blockSignals(False)

    def _on_view_mode_changed(self, mode: str):
        if not self.app_state.ready or not self.plot_container:
            return

        feature_sel = getattr(self.app_state, 'features_sel', None)
        is_ephys = feature_sel in getattr(self.app_state, 'ephys_source_map', {})
        if feature_sel == "Audio Waveform":
            self._apply_view_mode_for_waveform()
        elif is_ephys:
            self._apply_view_mode_for_ephys_waveform()
        else:
            self._apply_view_mode_for_feature()

        self._update_audio_overlay_checkboxes()
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _apply_view_mode_for_waveform(self):
        mode = self.view_mode_combo.currentText()
        if mode.startswith("Spectrogram"):
            self.plot_container.spectrogram_plot.set_source(None)
            self.plot_container.switch_to_spectrogram()
        elif mode.startswith("Heatmap"):
            self.plot_container.spectrogram_plot.set_source(None)
            self.plot_container.switch_to_heatmap()
        else:
            self.plot_container.spectrogram_plot.set_source(None)
            self.plot_container.switch_to_audiotrace()

    def _apply_view_mode_for_ephys_waveform(self):
        from .data_sources import build_ephys_source

        mode = self.view_mode_combo.currentText()
        if mode.startswith("Spectrogram"):
            source = build_ephys_source(self.app_state)
            if source is None:
                self.view_mode_combo.blockSignals(True)
                self.view_mode_combo.setCurrentText("Line (N-dim)")
                self.view_mode_combo.blockSignals(False)
                self._configure_ephys_trace_plot()
                self.plot_container.switch_to_ephystrace()
                return
            self.plot_container.spectrogram_plot.set_source(source)
            self.plot_container.switch_to_spectrogram()
        elif mode.startswith("Heatmap"):
            self.plot_container.spectrogram_plot.set_source(None)
            self.plot_container.switch_to_heatmap()
        else:
            self.plot_container.spectrogram_plot.set_source(None)
            self._configure_ephys_trace_plot()
            self.plot_container.switch_to_ephystrace()

    def _configure_ephys_trace_plot(self):
        if self.ephys_widget:
            self.ephys_widget.configure_ephys_trace_plot()

    def _hide_ephys_channel_controls(self):
        if self.ephys_widget:
            self.ephys_widget.hide_ephys_channel_controls()

    def _apply_view_mode_for_feature(self):
        mode = self.view_mode_combo.currentText()
        if mode.startswith("Spectrogram"):
            source = self._build_xarray_source()
            if source is None:
                self.view_mode_combo.blockSignals(True)
                self.view_mode_combo.setCurrentText("Line")
                self.view_mode_combo.blockSignals(False)
                self.plot_container.switch_to_lineplot()
                self._update_audio_overlay()
                return
            self.plot_container.spectrogram_plot.set_source(source)
            self.plot_container.switch_to_spectrogram()
        elif mode.startswith("Heatmap"):
            self.plot_container.spectrogram_plot.set_source(None)
            self.plot_container.switch_to_heatmap()
        else:
            self.plot_container.spectrogram_plot.set_source(None)
            self.plot_container.switch_to_lineplot()
            self._update_audio_overlay()

    def _build_xarray_source(self):
        from .data_sources import build_xarray_source

        feature_sel = getattr(self.app_state, 'features_sel', None)
        if not feature_sel:
            return None

        try:
            da = self.app_state.ds[feature_sel]
            ds_kwargs = self.app_state.get_ds_kwargs()
            data, _ = sel_valid(da, ds_kwargs)
            time_coord = get_time_coord(da)
            time_dim_name = time_coord.dims[0]
            time_values = np.asarray(time_coord)
            if data.ndim > 1:
                data = np.nanmean(data, axis=tuple(range(1, data.ndim)))
            data = np.asarray(data, dtype=np.float64).ravel()
            np.nan_to_num(data, copy=False, nan=0.0)
            return build_xarray_source(
                xr.DataArray(data, dims=[time_dim_name], coords={time_dim_name: time_values}),
                time_values,
                feature_sel,
                ds_kwargs,
            )
        except (KeyError, ValueError, IndexError) as e:
            print(f"Cannot build xarray spectrogram source: {e}")
            return None

    def _on_envelope_overlay_changed(self):
        if not self.plot_container:
            return
        can_show = self.plot_container.is_lineplot() or self.plot_container.is_audiotrace() or self.plot_container.is_ephystrace()
        if self.show_envelope_checkbox.isChecked() and can_show:
            self.plot_container.show_envelope_overlay()
        else:
            self.plot_container.hide_envelope_overlay()

    def _update_audio_overlay_checkboxes(self):
        if not hasattr(self.app_state, 'features_sel'):
            return

        feature = self.app_state.features_sel
        view_mode = self.view_mode_combo.currentText() if hasattr(self, 'view_mode_combo') else "Line"
        is_audio_waveform = feature == "Audio Waveform"
        is_ephys_waveform = feature in getattr(self.app_state, 'ephys_source_map', {})
        is_waveform = is_audio_waveform or is_ephys_waveform
        is_line_view = view_mode.startswith("Line")

        if is_waveform or not is_line_view:
            self.show_waveform_checkbox.blockSignals(True)
            self.show_waveform_checkbox.setEnabled(False)
            self.show_waveform_checkbox.setChecked(False)
            self.show_waveform_checkbox.blockSignals(False)
            self.show_spectrogram_checkbox.blockSignals(True)
            self.show_spectrogram_checkbox.setEnabled(False)
            self.show_spectrogram_checkbox.setChecked(False)
            self.show_spectrogram_checkbox.blockSignals(False)

        if is_waveform and is_line_view:
            self.show_envelope_checkbox.setEnabled(True)
        elif not is_line_view:
            self.show_envelope_checkbox.blockSignals(True)
            self.show_envelope_checkbox.setEnabled(False)
            self.show_envelope_checkbox.setChecked(False)
            self.show_envelope_checkbox.blockSignals(False)
            if self.plot_container:
                self.plot_container.hide_envelope_overlay()
        else:
            self.show_waveform_checkbox.setEnabled(True)
            self.show_spectrogram_checkbox.setEnabled(True)
            self.show_envelope_checkbox.setEnabled(True)

    def _set_controls_enabled(self, enabled: bool):
        for control in self.controls:
            control.setEnabled(enabled)
        self.io_widget.set_controls_enabled(enabled)
        self.app_state.ready = enabled

    def _create_combo_widget(self, key, vars):
        excluded_from_all = {"individuals", "features", "colors", "cameras", "mics", "pose"}
        show_all_checkbox = key not in excluded_from_all

        combo = QComboBox()
        combo.setObjectName(f"{key}_combo")
        combo.currentIndexChanged.connect(self._on_combo_changed)
        if key == "colors":
            colour_variables = ["None"] + [str(var) for var in vars]
            combo.addItems(colour_variables)
        else:
            combo.addItems([str(var) for var in vars])

        make_searchable(combo)

        target_layout = self.coords_groupbox_layout

        if show_all_checkbox:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)

            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout.addWidget(combo)

            all_checkbox = QCheckBox("All")
            all_checkbox.setObjectName(f"{key}_all_checkbox")
            all_checkbox.setToolTip(f"Show all {key} traces on the plot")
            all_checkbox.stateChanged.connect(lambda state, k=key: self._on_all_checkbox_changed(k, state))
            row_layout.addWidget(all_checkbox)

            self.all_checkboxes[key] = all_checkbox
            self.controls.append(all_checkbox)

            target_layout.addRow(f"{key.capitalize()}:", row_widget)
        else:
            target_layout.addRow(f"{key.capitalize()}:", combo)

        self.combos[key] = combo
        self.controls.append(combo)

        return combo

    def _on_combo_changed(self):
        if not self.app_state.ready:
            return

        combo = self.sender()
        key = None

        for io_key, io_value in self.io_widget.combos.items():
            if io_value is combo:
                key = io_key
                break

        if key is None:
            for data_key, data_value in self.combos.items():
                if data_value is combo:
                    key = data_key
                    break

        if key:
            selected_value = combo.currentText()
            self.app_state.set_key_sel(key, selected_value)

            if key == "features":
                self._update_view_mode_items(selected_value)
                self.view_mode_combo.show()
                is_ephys = selected_value in getattr(self.app_state, 'ephys_source_map', {})
                if selected_value == "Audio Waveform":
                    self._apply_view_mode_for_waveform()
                    self._hide_ephys_channel_controls()
                elif is_ephys:
                    self._apply_view_mode_for_ephys_waveform()
                else:
                    self._apply_view_mode_for_feature()
                    self._hide_ephys_channel_controls()

                self._update_audio_overlay_checkboxes()
                self._update_audio_overlay()

                current_plot = self.plot_container.get_current_plot()
                xmin, xmax = current_plot.get_current_xlim()

            if key in ["cameras", "mics"]:
                if self.app_state.no_video_mode and key == "mics":
                    self._update_audio_only()
                else:
                    self.update_video_audio()

            if key not in ["cameras", "pose"]:
                current_plot = self.plot_container.get_current_plot()
                xmin, xmax = current_plot.get_current_xlim()
                self.update_main_plot(t0=xmin, t1=xmax)

            if key in ["individuals", "keypoints", "colors"]:
                self.update_space_plot()

            if key == "individuals":
                self.labels_widget.refresh_labels_shapes_layer()

            if key == "pose":
                self.update_pose()

    def _on_all_checkbox_changed(self, key: str, state: int):
        if not self.app_state.ready:
            return

        combo = self.combos.get(key)
        if combo is None:
            return

        is_checked = state == Qt.Checked

        if is_checked:
            for other_key, other_checkbox in self.all_checkboxes.items():
                if other_key != key and other_checkbox.isChecked():
                    other_checkbox.blockSignals(True)
                    other_checkbox.setChecked(False)
                    other_checkbox.blockSignals(False)
                    other_combo = self.combos.get(other_key)
                    if other_combo:
                        other_combo.setEnabled(True)
                        self.app_state.set_key_sel(other_key, other_combo.currentText())
                    self._update_all_checkbox_state(other_key, False)

        combo.setEnabled(not is_checked)
        self._update_all_checkbox_state(key, is_checked)

        if is_checked:
            self.app_state.set_key_sel(key, None)
        else:
            current_text = combo.currentText()
            self.app_state.set_key_sel(key, current_text)

        current_plot = self.plot_container.get_current_plot()
        xmin, xmax = current_plot.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)
        self.update_space_plot()

    def _update_all_checkbox_state(self, key: str, is_checked: bool):
        states = self.app_state.all_checkbox_states.copy()
        if is_checked:
            states[key] = True
        else:
            states.pop(key, None)
        self.app_state.all_checkbox_states = states

    def _restore_or_set_defaults(self):
        for key, vars in self.type_vars_dict.items():
            combo = self.io_widget.combos.get(key) or self.combos.get(key)

            if combo is not None:
                saved_value = self.app_state.get_key_sel(key) if self.app_state.key_sel_exists(key) else None
                vars_str = [str(var) for var in vars]

                if saved_value in vars_str:
                    combo.setCurrentText(str(saved_value))
                elif saved_value and key == "mics":
                    match = next((v for v in vars_str if v.startswith(str(saved_value))), None)
                    if match:
                        combo.setCurrentText(match)
                        self.app_state.set_key_sel(key, match)
                    else:
                        combo.setCurrentText(str(vars[0]))
                        self.app_state.set_key_sel(key, str(vars[0]))
                else:
                    if key == "features" and "speed" in vars:
                        combo.setCurrentText("speed")
                        self.app_state.set_key_sel(key, "speed")
                    else:
                        combo.setCurrentText(str(vars[0]))
                        self.app_state.set_key_sel(key, str(vars[0]))

                previous = getattr(self.app_state, f"{key}_sel_previous", None)
                if previous not in vars_str:
                    fallback = vars_str[1] if len(vars_str) > 1 else vars_str[0] if vars_str else None
                    self.app_state.set_key_sel_previous(key, fallback)

        if self.app_state.key_sel_exists("trials"):
            saved_trial = self.app_state.get_key_sel("trials")
            self.app_state.set_key_sel("trials", saved_trial)
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials_sel))
        else:
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials[0]))
            self.app_state.trials_sel = self.app_state.trials[0]

        space_plot_type = getattr(self.app_state, 'space_plot_type', 'None')
        if hasattr(self, 'space_plot_combo'):
            self.space_plot_combo.setCurrentText(space_plot_type)

        checkbox_states = self.app_state.all_checkbox_states or {}
        for key, is_checked in checkbox_states.items():
            checkbox = self.all_checkboxes.get(key)
            combo = self.combos.get(key)
            if checkbox and is_checked:
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
                if combo:
                    combo.setEnabled(False)
                self.app_state.set_key_sel(key, None)

    # ------------------------------------------------------------------
    # Trial change
    # ------------------------------------------------------------------

    def _load_trial_with_fallback(self) -> None:
        first_trial = self.app_state.trials[0]
        current_trial = self.app_state.trials_sel

        try:
            is_nan = np.isnan(current_trial)
        except (TypeError, ValueError):
            is_nan = False
        if not current_trial or is_nan:
            self.app_state.trials_sel = first_trial
            current_trial = first_trial

        try:
            self.on_trial_changed()
        except Exception as e:
            if current_trial == first_trial:
                raise RuntimeError(f"Failed to load first trial: {e}") from e
            print(f"Error loading trial {current_trial}: {e}\nReverting to first trial.")
            self.app_state.trials_sel = first_trial
            self.on_trial_changed()

    def _validate_media_files(self, ds) -> list[str]:
        missing = []
        attrs = ds.attrs

        video_folder = self.app_state.video_folder
        if video_folder:
            for cam in np.atleast_1d(attrs.get("cameras", [])):
                path = os.path.join(video_folder, str(cam))
                if not os.path.isfile(path):
                    missing.append(f"Video: {path}")

        audio_folder = self.app_state.audio_folder
        if audio_folder:
            mics = np.atleast_1d(attrs.get("mics", []))
            if mics.size == 0:
                show_warning(
                "You selected an audio folder, although the .nc "
                "contains no media attrs of audio data."
            )
            else:      
                for mic in mics:
                    path = os.path.join(audio_folder, str(mic))
                    if not os.path.isfile(path):
                        missing.append(f"Audio: {path}")

        pose_folder = self.app_state.pose_folder
        if pose_folder:
            poses = np.atleast_1d(attrs.get("pose", []))
            if poses.size == 0:
                show_warning(
                    "You selected a pose folder, although the .nc"
                    "contains no pose data.") 
            else:  
                for pose_file in poses:
                    path = os.path.join(pose_folder, str(pose_file))
                    if not os.path.isfile(path):
                        missing.append(f"Pose: {path}")

        return missing

    def on_trial_changed(self):
        trials_sel = self.app_state.trials_sel

        self.app_state.ds = self.app_state.dt.trial(trials_sel)
        self.app_state.label_ds = self.app_state.label_dt.trial(trials_sel)

        if self.app_state.pred_dt is not None:
            self.app_state.pred_ds = self.app_state.pred_dt.trial(trials_sel)

        self.io_widget.update_device_combos_for_trial(self.app_state.ds)

        fallback_feature = self.combos["features"].itemText(0)
        feature_sel = getattr(self.app_state, 'features_sel', fallback_feature)
        self.app_state.set_time(feature_sel=feature_sel)

        if hasattr(self, 'view_mode_combo'):
            self._update_view_mode_items(feature_sel)
        is_ephys = feature_sel in getattr(self.app_state, 'ephys_source_map', {})
        if feature_sel == "Audio Waveform":
            self._apply_view_mode_for_waveform()
        elif is_ephys:
            self._apply_view_mode_for_ephys_waveform()
        elif hasattr(self, 'view_mode_combo'):
            view_mode = self.view_mode_combo.currentText()
            if view_mode.startswith("Spectrogram"):
                source = self._build_xarray_source()
                if source is not None:
                    self.plot_container.spectrogram_plot.set_source(source)
            elif view_mode.startswith("Heatmap"):
                self.plot_container.heatmap_plot._clear_buffer()

        self.app_state.label_intervals = self.app_state.get_trial_intervals(trials_sel)

        self.app_state.current_frame = 0
        self.update_video_audio()
        if not self.app_state.no_video_mode:
            self.update_pose()
        self.update_label()
        self.update_main_plot()
        self.update_space_plot()

        from .multipanel_container import MultiPanelContainer
        if isinstance(self.plot_container, MultiPanelContainer):
            self.plot_container.update_time_range_from_data()

        self.app_state.verification_changed.emit()

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def update_main_plot(self, **kwargs):
        if not self.app_state.ready:
            return

        ds_kwargs = self.app_state.get_ds_kwargs()
        current_plot = self.plot_container.get_current_plot()

        self.plot_container.clear_amplitude_envelope()

        try:
            current_plot.update_plot(**kwargs)
            self.update_label_plot(ds_kwargs)

            if self.show_confidence_checkbox.isChecked():
                try:
                    label_confidence, _ = sel_valid(self.app_state.label_ds.labels_confidence, ds_kwargs)
                    self.plot_container.show_confidence_plot(label_confidence)
                except (KeyError, AttributeError):
                    pass
            else:
                self.plot_container.hide_confidence_plot()

            if self.plot_container.is_lineplot():
                self.plot_container.update_audio_overlay()

            if hasattr(self, 'show_envelope_checkbox') and self.show_envelope_checkbox.isChecked():
                if self.plot_container.is_lineplot() or self.plot_container.is_audiotrace() or self.plot_container.is_ephystrace():
                    self.plot_container.show_envelope_overlay()

        except (KeyError, AttributeError, ValueError) as e:
            show_error(f"Error updating plot: {e}")

    def update_label_plot(self, ds_kwargs):
        intervals_df = self.app_state.label_intervals

        if intervals_df is not None and not intervals_df.empty and "individuals" in ds_kwargs:
            selected_ind = str(ds_kwargs["individuals"])
            intervals_df = intervals_df[intervals_df["individual"] == selected_ind]

        predictions_df = None

        if (
            self.labels_widget.pred_show_predictions.isChecked()
            and hasattr(self.app_state, 'pred_ds')
            and self.app_state.pred_ds is not None
        ):
            pred_ds = self.app_state.pred_ds
            predictions, _ = sel_valid(pred_ds.labels, ds_kwargs)
            from ethograph.utils.label_intervals import dense_to_intervals
            pred_time = get_time_coord(pred_ds.labels).values
            individuals = (
                list(pred_ds.coords['individuals'].values)
                if 'individuals' in pred_ds.coords
                else ["default"]
            )
            predictions_df = dense_to_intervals(
                np.asarray(predictions).reshape(-1, 1) if np.asarray(predictions).ndim == 1 else np.asarray(predictions),
                pred_time,
                individuals,
            )

        self.labels_widget.plot_all_labels(intervals_df, predictions_df)

    # ------------------------------------------------------------------
    # Video / audio / pose / space
    # ------------------------------------------------------------------

    def _update_audio_only(self):
        """Update audio path and refresh multi-panel audio panels (no-video mode)."""
        if not self.app_state.ready:
            return
        if self.app_state.audio_folder and hasattr(self.app_state, 'mics_sel'):
            audio_path, _ = self.app_state.get_audio_source()
            if audio_path:
                self.app_state.audio_path = audio_path
            else:
                self.app_state.audio_path = None

        if self.transform_widget:
            self.transform_widget.set_enabled_state(has_audio=True)
        if self.changepoints_widget:
            self.changepoints_widget.set_enabled_state(True)

        from .multipanel_container import MultiPanelContainer
        if isinstance(self.plot_container, MultiPanelContainer):
            self.plot_container.update_audio_panels()

    def update_video_audio(self):
        if not self.app_state.ready:
            return
        if self.app_state.no_video_mode:
            self._update_audio_only()
            return
        if not self.app_state.video_folder:
            return

        current_plot = self.plot_container.get_current_plot()
        marker_time = current_plot.time_marker.value() if current_plot else 0.0
        restore_frame = max(0, int(marker_time * self.app_state.ds.fps))

        if self.app_state.video_folder and hasattr(self.app_state, 'cameras_sel'):
            video_file = self.app_state.cameras_sel
            video_path = os.path.join(self.app_state.video_folder, video_file)
            self.app_state.video_path = os.path.normpath(video_path)

        has_audio = False
        if self.app_state.audio_folder and hasattr(self.app_state, 'mics_sel'):
            audio_path, _ = self.app_state.get_audio_source()
            if audio_path:
                self.app_state.audio_path = audio_path
                has_audio = True
            else:
                self.app_state.audio_path = None

        if self.transform_widget:
            self.transform_widget.set_enabled_state(has_audio=has_audio)

        if self.changepoints_widget:
            self.changepoints_widget.set_enabled_state(has_audio)

        self.show_envelope_checkbox.show()
        if has_audio:
            
            self.show_waveform_checkbox.show()
            self.show_spectrogram_checkbox.show()
        else:
            self.show_waveform_checkbox.hide()
            self.show_spectrogram_checkbox.hide()

        video_ext = Path(self.app_state.video_path).suffix.lower()
        if video_ext in ('.avi', '.mov') and not getattr(self, '_video_format_warned', False):
            self._video_format_warned = True
            show_warning(
                f"Video format '{video_ext}' may have inaccurate frame seeking. "
                f"See https://ethograph.readthedocs.io/en/latest/troubleshooting/"
            )

        new_video_data = FastVideoReader(
            self.app_state.video_path, read_format='rgb24',
        )
        _ = new_video_data.shape

        if self.video:
            try:
                self.video.frame_changed.disconnect(self._on_sync_frame_changed)
                self.video.cleanup()
            except (RuntimeError, TypeError):
                pass
            self.video = None
            self.app_state.video = None

        video_layer = self.viewer.add_image(new_video_data, name="video_new", rgb=True)

        for layer in list(self.viewer.layers):
            if layer.name in ["video", "Video Stream"]:
                self.viewer.layers.remove(layer)

        video_layer.name = "video"
        video_index = self.viewer.layers.index(video_layer)
        self.viewer.layers.move(video_index, 0)

        try:
            self.video = NapariVideoSync(
                viewer=self.viewer,
                app_state=self.app_state,
                video_source=self.app_state.video_path,
                audio_source=self.app_state.audio_path
            )
            self.app_state.video = self.video
        except Exception as e:
            print(f"Error initializing NapariVideoSync: {e}")
            return

        self.viewer.dims.events.current_step.disconnect(self.video._on_napari_step_change)
        self.video.seek_to_frame(restore_frame)
        self.app_state.current_frame = restore_frame
        self.viewer.dims.events.current_step.connect(self.video._on_napari_step_change)
        self.video.frame_changed.connect(self._on_sync_frame_changed)

    def update_label(self):
        self.labels_widget.refresh_labels_shapes_layer()

    def toggle_pause_resume(self):
        from .multipanel_container import MultiPanelContainer
        if isinstance(self.plot_container, MultiPanelContainer):
            self.plot_container.toggle_pause_resume()
            return
        if not self.video:
            return
        self.video.toggle_pause_resume()

    def _on_sync_frame_changed(self, frame_number: int):
        self.app_state.current_frame = frame_number
        self.plot_container.update_time_marker_and_window(frame_number)

        current_time = frame_number / self.app_state.effective_fps
        xlim = self.plot_container.get_current_xlim()
        if current_time < xlim[0] or current_time > xlim[1]:
            self.plot_container.set_x_range(mode='center', center_on_frame=frame_number)

    def update_pose(self):
        if not self.app_state.pose_folder or not hasattr(self.app_state, 'pose_sel'):
            return

        for layer in list(self.viewer.layers):
            if self.file_name and layer.name in [
                f"tracks: {self.file_name}",
                f"points: {self.file_name}",
                f"boxes: {self.file_name}",
                f"skeleton: {self.file_name}"
            ]:
                self.viewer.layers.remove(layer)

        self.fps = self.app_state.ds.fps
        self.source_software = self.app_state.ds.source_software

        pose_file = self.app_state.pose_sel
        self.file_path = os.path.join(self.app_state.pose_folder, pose_file)
        self.file_name = Path(self.file_path).name

        self._format_data_for_layers()
        self._apply_pose_confidence_filter()
        if not np.any(self.data_not_nan):
            return
        self._set_common_color_property()
        self._set_text_property()
        self._add_points_layer()
        self._set_initial_state()

    def _apply_pose_confidence_filter(self):
        threshold = self.app_state.pose_hide_threshold
        if threshold <= 0.0 or self.properties is None:
            return
        if "confidence" not in self.properties.columns:
            return
        low_confidence = self.properties["confidence"].values < threshold
        self.data_not_nan[low_confidence] = False

    def closeEvent(self, event):
        from .plots_ephystrace import SharedEphysCache
        SharedAudioCache.clear_cache()
        SharedEphysCache.clear_cache()
        self.video.stop()
        super().closeEvent(event)

    def _on_pose_hide_threshold_changed(self, value: float):
        self.app_state.pose_hide_threshold = value
        self.update_pose()

    def _on_space_plot_changed(self):
        if not self.app_state.ready:
            return
        plot_type = self.space_plot_combo.currentText()
        self.app_state.space_plot_type = plot_type
        self.update_space_plot()

    def update_space_plot(self):
        if not self.app_state.ready:
            return

        plot_type = self.app_state.get_with_default('space_plot_type')

        if plot_type == "Layer controls":
            if self.space_plot:
                self.space_plot.hide()
        else:
            if not self.space_plot:
                self.space_plot = SpacePlot(self.viewer, self.app_state)
                if self.labels_widget:
                    self.labels_widget.highlight_spaceplot.connect(self._highlight_positions_in_space_plot)

            individual = self.combos.get('individuals', None)
            individual_text = individual.currentText() if individual else None
            keypoints = self.combos.get('keypoints', None)
            keypoints_text = keypoints.currentText() if keypoints else None
            color_variable = self.combos.get('colors', None)
            color_variable = color_variable.currentText() if color_variable else None

            if plot_type == "space_3D":
                view_3d = True
            elif plot_type == "space_2D":
                view_3d = False

            self.space_plot.update_plot(individual_text, keypoints_text, color_variable, view_3d)
            self.space_plot.show()

    def _highlight_positions_in_space_plot(self, start_time: float, end_time: float):
        start_frame = int(start_time * self.app_state.ds.fps)
        end_frame = int(end_time * self.app_state.ds.fps)
        if self.space_plot and self.space_plot.dock_widget.isVisible():
            self.space_plot.highlight_positions(start_frame, end_frame)

    def reset_widget_state(self):
        for combo in self.combos.values():
            combo.clear()
            combo.addItems(["None"])
            combo.setCurrentText("None")

        if hasattr(self, 'plot_spec_checkbox'):
            self.plot_spec_checkbox.setChecked(False)
        if hasattr(self, 'clear_audio_checkbox'):
            self.clear_audio_checkbox.setChecked(False)

        if hasattr(self, 'space_plot_combo'):
            self.space_plot_combo.clear()
            self.space_plot_combo.addItems(["Layer controls"])
            self.space_plot_combo.setCurrentText("Layer controls")

        if self.navigation_widget:
            self.navigation_widget.trial_conditions_combo.clear()
            self.navigation_widget.trial_conditions_combo.addItem("None")
            self.navigation_widget.trial_conditions_value_combo.clear()
            self.navigation_widget.trial_conditions_value_combo.addItem("None")

        if self.navigation_widget and hasattr(self.navigation_widget, 'trials_combo'):
            self.navigation_widget.trials_combo.clear()

        self.type_vars_dict = {}
        self.video_path = None
        self.audio_path = None
        self.fps = None
        self.source_software = None
        self.file_path = None
        self.file_name = None

        if self.space_plot:
            self.space_plot.hide()

        print("DataWidget state reset to default")
