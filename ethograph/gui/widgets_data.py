"""Widget for selecting start/stop times and playing a segment in napari."""

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr
from movement.napari.loader_widgets import DataLoader
from napari.utils.notifications import show_error
from napari.viewer import Viewer
from napari_pyav._reader import FastVideoReader
from qtpy.QtCore import QSortFilterProxyModel, Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
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
    """Widget to control which data is loaded, displayed and stored for next time."""

    def __init__(
        self,
        napari_viewer: Viewer,
        app_state,
        meta_widget,
        io_widget,
        parent=None,
    ):
        DataLoader.__init__(self, napari_viewer)  # Pass required args for DataLoader
        QWidget.__init__(self, parent=parent)
        self.parent = parent
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
        self.axes_widget = None
        self.audio_widget = None
        self.audio_player = None 
        self.video_path = None
        self.audio_path = None
        self.space_plot = None  

        # Dictionary to store all combo boxes
        self.combos = {}
        # Dictionary to store "All" checkboxes for combo widgets
        self.all_checkboxes = {}
        # Dictionary to store all controls for enabling/disabling
        self.controls = []

        # Tracking stuff
        self.fps = None
        self.source_software = None
        self.file_path = None
        self.file_name = None

        self.app_state.audio_video_sync = None
        # E.g. {keypoints = ["beakTip, StickTip"], trials=[1, 2, 3, 4], ...}
        self.type_vars_dict = {}  # Gets filled by load_dataset
        
        

    def set_references(self, plot_container, labels_widget, axes_widget, navigation_widget, audio_widget=None, changepoints_widget=None):
        """Set references to other widgets after creation."""
        self.plot_container = plot_container
        self.labels_widget = labels_widget
        self.axes_widget = axes_widget
        self.navigation_widget = navigation_widget
        self.audio_widget = audio_widget
        self.changepoints_widget = changepoints_widget

        if changepoints_widget is not None:
            changepoints_widget.request_plot_update.connect(self._on_plot_update_request)

    def _on_plot_update_request(self):
        """Handle request to update the main plot from changepoints widget."""
        if not self.app_state.ready or not self.plot_container:
            return
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _get_sampling_rate(self, time_coords: np.ndarray) -> float:
        """Calculate sampling rate from time coordinates."""
        dt = np.median(np.diff(time_coords))
        return 1.0 / dt

    def _set_audio_time(self):
        """Set app_state.time to an audio-rate time coordinate for Audio Waveform."""
        ds = self.app_state.ds
        audio_sr = ds.attrs.get("audio_sr")
        if audio_sr is None:
            return

        audio_sr = float(audio_sr)

        # Determine duration from audio file (preferred) or feature time extent
        n_samples = None
        audio_path = getattr(self.app_state, 'audio_path', None)
        if audio_path:
            loader = SharedAudioCache.get_loader(audio_path)
            if loader:
                n_samples = len(loader)
                audio_sr = float(loader.rate)

        if n_samples is None:
            features = [v for v in self.type_vars_dict.get("features", [])
                        if v != "Audio Waveform"]
            if features:
                feature_time = get_time_coord(ds[features[0]])
                duration = float(feature_time.values[-1])
                n_samples = int(duration * audio_sr) + 1
            else:
                return

        time_values = np.arange(n_samples) / audio_sr
        self.app_state.time = xr.DataArray(
            time_values, dims=["time_audio"],
            coords={"time_audio": time_values}, name="time_audio",
        )
        self.app_state.label_sr = audio_sr




    def on_load_clicked(self):
        """Load the file and show line plot in napari dock."""
        
        if not self.app_state.nc_file_path:
            show_error("Please select a path ending with .nc")
            return
        
        
        self.setVisible(False)

        # Load ds
        nc_file_path = self.io_widget.get_nc_file_path()
        
        
        self.app_state.dt, label_dt, self.type_vars_dict = load_dataset(nc_file_path)

        # Add audio features to type_vars_dict if audio is available
        has_audio = "mics" in self.type_vars_dict or self.app_state.audio_folder
        if has_audio and "features" in self.type_vars_dict:
            features_list = list(self.type_vars_dict["features"])
            self.type_vars_dict["features"] = features_list + ["Audio Waveform"]



        


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

        
    
        features_list = list(self.type_vars_dict["features"])
        
        # TODO: Change later
        if 'speed' in features_list:
            setattr(self.app_state, "features_sel", "speed")
        else:
            setattr(self.app_state, "features_sel", features_list[0])
        
        
        self.app_state.time = get_time_coord(self.app_state.ds[features_list[0]])
        

        # Derive label_sr from feature time coord (labels are now interval-based)
        feature_time = self.app_state.time
        if feature_time is not None and len(feature_time) > 1:
            self.app_state.label_sr = self._get_sampling_rate(np.asarray(feature_time))


        self.app_state.trials = sorted(trials)
        
        
        

        self._create_trial_controls()

        if self.changepoints_widget:
            self.changepoints_widget.setEnabled(True)

        if hasattr(self.app_state, 'features_sel_changed'):
            self.app_state.features_sel_changed.connect(self._on_external_feature_change)

        self._restore_or_set_defaults()
        self._set_controls_enabled(True)
        self.app_state.ready = True

        if self.audio_widget:
            self.audio_widget.set_enabled_state(has_audio=False)

        load_btn = self.io_widget.load_button
        load_btn.setEnabled(False)

        # Set initial trial selection before calling on_trial_changed
        if not self.app_state.trials_sel:
            self.app_state.trials_sel = self.app_state.trials[0]

        # Update trials combo after ready=True so it populates correctly
        self.update_trials_combo()

        self._load_trial_with_fallback()

        self.view_mode_combo.show()

        self.setVisible(True)
        load_btn.setText("Restart app to load new data")

        self._force_layout_update()
        


            
    def update_trials_combo(self) -> None:
        """Update trials combo box with verification status color coding in dropdown."""
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
            
      
            bg_color = QColor(144, 238, 144) if is_verified else QColor(255, 182, 193)  # 
            combo.setItemData(index, bg_color, Qt.BackgroundRole)
            

            text_color = QColor(0, 100, 0) if is_verified else QColor(139, 0, 0)  
            combo.setItemData(index, text_color, Qt.ForegroundRole)
        
        
        combo.setCurrentText(str(self.app_state.trials_sel))
        combo.blockSignals(False)


    def _collect_trial_status(self) -> Dict[int, int]:
        """Extract verification status for each trial from data tree."""
        trial_status = {}
        
        for trial in self.app_state.trials:
            is_verified = self.app_state.label_dt.trial(trial).attrs.get('human_verified', 0)
            trial_status[trial] = bool(is_verified)

        return trial_status


 

    def _create_trial_controls(self):
        """Create all trial-related controls based on info configuration."""

        # Create device combos in IOWidget
        self.io_widget.create_device_controls(self.type_vars_dict)

        self.navigation_widget.setup_trial_conditions(self.type_vars_dict)
        self.navigation_widget.set_data_widget(self)

        # Create QGroupBox for xarray coordinates
        self.coords_groupbox = QGroupBox("Xarray coords")
        self.coords_groupbox_layout = QFormLayout()
        self.coords_groupbox_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        self.coords_groupbox_layout.setContentsMargins(DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN)
        self.coords_groupbox.setLayout(self.coords_groupbox_layout)

        non_data_type_vars = ["cameras", "mics", "pose", "trial_conditions", "changepoints", "rgb"]

        for type_var in self.type_vars_dict.keys():
            if type_var.lower() not in non_data_type_vars:
                self._create_combo_widget(type_var, self.type_vars_dict[type_var])

        self.layout().addRow(self.coords_groupbox)

        # View mode + space plot row
        view_space_row = QHBoxLayout()
        view_space_row.setSpacing(15)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.setObjectName("view_mode_combo")
        self.view_mode_combo.addItems([
            "Line (N-dim)",
            "Spectrogram (1-dim)",
            "Heatmap (N-dim)",
        ])
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        self.view_mode_combo.hide()
        self.controls.append(self.view_mode_combo)
        view_space_row.addWidget(self.view_mode_combo)

        
        self.space_plot_combo = QComboBox()
        self.space_plot_combo.setObjectName("space_plot_combo")
        
        # TODO: Make more generalized to handle other types of 2D/3D spatial input (e.g. PCA space)
        if 'position' in self.app_state.ds.data_vars:
            self.space_plot_combo.addItems(["Layer controls", "space_2D", "space_3D"])
            self.space_plot_combo.currentTextChanged.connect(self._on_space_plot_changed)
        else:
            self.space_plot_combo.addItems(["Layer controls"])
        self.controls.append(self.space_plot_combo)
        view_space_row.addWidget(self.space_plot_combo)

        view_space_row.addStretch()
        self.layout().addRow("Views:", view_space_row)


        overlay_row = QHBoxLayout()
        overlay_row.setSpacing(15)

        self.show_confidence_checkbox = QCheckBox("Confidence")
        self.show_confidence_checkbox.setChecked(False)
        self.show_confidence_checkbox.stateChanged.connect(self.refresh_lineplot)
        overlay_row.addWidget(self.show_confidence_checkbox)

        self.show_waveform_checkbox = QCheckBox("Waveform (audio)")
        self.show_waveform_checkbox.setChecked(False)
        self.show_waveform_checkbox.stateChanged.connect(self._on_audio_overlay_changed)
        self.show_waveform_checkbox.hide()
        overlay_row.addWidget(self.show_waveform_checkbox)

        self.show_spectrogram_checkbox = QCheckBox("Spectrogram (audio)")
        self.show_spectrogram_checkbox.setChecked(False)
        self.show_spectrogram_checkbox.stateChanged.connect(self._on_audio_overlay_changed)
        self.show_spectrogram_checkbox.hide()
        overlay_row.addWidget(self.show_spectrogram_checkbox)

        overlay_row.addStretch()
        self.layout().addRow("Overlays:", overlay_row)

        # Initially disable trial controls until data is loaded
        self._set_controls_enabled(False)
        
    
    def refresh_lineplot(self):
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)
        

    def cycle_view_mode(self):
        """Cycle through view modes: Line -> Spectrogram -> Heatmap -> Line."""
        if not hasattr(self, 'view_mode_combo') or not self.view_mode_combo.isVisible():
            return
        next_index = (self.view_mode_combo.currentIndex() + 1) % self.view_mode_combo.count()
        self.view_mode_combo.setCurrentIndex(next_index)

    def _on_audio_overlay_changed(self):
        """Handle audio overlay checkbox changes with mutual exclusion."""
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
        """Update audio overlay visibility in plot container."""
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

    def _on_view_mode_changed(self, mode: str):
        """Switch between Line / Spectrogram / Heatmap view modes."""
        if not self.app_state.ready or not self.plot_container:
            return

        feature_sel = getattr(self.app_state, 'features_sel', None)
        if feature_sel == "Audio Waveform":
            self._apply_view_mode_for_waveform()
        else:
            self._apply_view_mode_for_feature()

        self._update_audio_overlay_checkboxes()
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _apply_view_mode_for_waveform(self):
        """Apply current view mode when feature is Audio Waveform (audio data)."""
        mode = self.view_mode_combo.currentText()
        if mode.startswith("Spectrogram"):
            self.plot_container.spectrogram_plot.set_source(None)
            self.plot_container.switch_to_spectrogram()
        else:
            self.plot_container.spectrogram_plot.set_source(None)
            self.plot_container.switch_to_audiotrace()

    def _apply_view_mode_for_feature(self):
        """Apply current view mode when feature is a regular xarray variable."""
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
        """Build an XarraySource from the current feature selection."""
        from .spectrogram_sources import build_xarray_source

        feature_sel = getattr(self.app_state, 'features_sel', None)
        if not feature_sel or feature_sel == "Audio Waveform":
            return None

        try:
            da = self.app_state.ds[feature_sel]
            ds_kwargs = self.app_state.get_ds_kwargs()
            data_1d, _ = sel_valid(da, ds_kwargs)
            time_values = np.asarray(get_time_coord(da))
            return build_xarray_source(
                xr.DataArray(data_1d, dims=["time"], coords={"time": time_values}),
                time_values,
                feature_sel,
                ds_kwargs,
            )
        except (KeyError, ValueError, IndexError) as e:
            print(f"Cannot build xarray spectrogram source: {e}")
            return None

    def _update_audio_overlay_checkboxes(self):
        """Enable/disable audio overlay checkboxes based on feature selection.

        Overlays only make sense when viewing a line plot feature, not when
        viewing Audio Waveform or Spectrogram directly.
        """
        if not hasattr(self.app_state, 'features_sel'):
            return

        feature = self.app_state.features_sel
        view_mode = self.view_mode_combo.currentText() if hasattr(self, 'view_mode_combo') else "Line"

        if feature == "Audio Waveform" or not view_mode.startswith("Line"):
            self.show_waveform_checkbox.blockSignals(True)
            self.show_waveform_checkbox.setEnabled(False)
            self.show_waveform_checkbox.setChecked(False)
            self.show_waveform_checkbox.blockSignals(False)

            self.show_spectrogram_checkbox.blockSignals(True)
            self.show_spectrogram_checkbox.setEnabled(False)
            self.show_spectrogram_checkbox.setChecked(False)
            self.show_spectrogram_checkbox.blockSignals(False)
        else:
            self.show_waveform_checkbox.setEnabled(True)
            self.show_spectrogram_checkbox.setEnabled(True)

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all trial-related controls."""
        for control in self.controls:
            control.setEnabled(enabled)
        # Also enable/disable IOWidget device controls
        self.io_widget.set_controls_enabled(enabled)
        self.app_state.ready = enabled

    def _force_layout_update(self):
        """Force Qt to recalculate layout by toggling collapsible widget."""
        if self.meta_widget and hasattr(self.meta_widget, 'collapsible_widgets'):
            io_collapsible = self.meta_widget.collapsible_widgets[1]
            io_collapsible.collapse()
            QApplication.processEvents()
            io_collapsible.expand()

    def _create_combo_widget(self, key, vars):
        """Create a combo box widget for a given info key.

        For qualifying type variables (not features, colors, cameras, mics,
        pose), adds an 'All' checkbox that shows all traces.
        Adds widgets to the coords_groupbox_layout.
        """
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

        # Add to groupbox layout instead of main layout
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

        # Figure out which key this belongs to
        combo = self.sender()
        key = None

        # Check IOWidget combos first
        for io_key, io_value in self.io_widget.combos.items():
            if io_value is combo:
                key = io_key
                break

        # Check DataWidget combos if not found in IOWidget
        if key is None:
            for data_key, data_value in self.combos.items():
                if data_value is combo:
                    key = data_key
                    break

        if key:
            selected_value = combo.currentText()
            self.app_state.set_key_sel(key, selected_value)

            if key == "features":
                if selected_value == "Audio Waveform":
                    self._set_audio_time()
                    self.view_mode_combo.show()
                    self._apply_view_mode_for_waveform()
                else:
                    self.app_state.time = get_time_coord(self.app_state.ds[selected_value])
                    self.view_mode_combo.show()
                    self._apply_view_mode_for_feature()

                self._update_audio_overlay_checkboxes()
                self._update_audio_overlay()

                current_plot = self.plot_container.get_current_plot()
                xmin, xmax = current_plot.get_current_xlim()
                
                  
            if key in ["cameras", "mics"]:
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

    def _on_external_feature_change(self):
        """Handle feature selection change from app_state (e.g., from changepoints widget)."""
        if not self.app_state.ready:
            return

        features_combo = self.combos.get("features")
        if features_combo is None:
            return

        feature = self.app_state.features_sel
        if not feature or feature == features_combo.currentText():
            return

        features_combo.blockSignals(True)
        idx = features_combo.findText(feature)
        if idx >= 0:
            features_combo.setCurrentIndex(idx)
        features_combo.blockSignals(False)

        if feature == "Audio Waveform":
            self._set_audio_time()
            self._apply_view_mode_for_waveform()
            self._update_audio_overlay_checkboxes()
            current_plot = self.plot_container.get_current_plot()
            xmin, xmax = current_plot.get_current_xlim()
            self.update_main_plot(t0=xmin, t1=xmax)
        else:
            self.plot_container.switch_to_lineplot()
            self.app_state.time = get_time_coord(self.app_state.ds[feature])
            self._update_audio_overlay_checkboxes()
            self._update_audio_overlay()
            current_plot = self.plot_container.get_current_plot()
            xmin, xmax = current_plot.get_current_xlim()
            self.update_main_plot(t0=xmin, t1=xmax)

    def _on_all_checkbox_changed(self, key: str, state: int):
        """Handle 'All' checkbox state change for a type variable.

        When checked, disables the combo and sets selection to None so all
        values are shown on the plot. When unchecked, restores the combo.
        Only one 'All' checkbox can be active at a time.
        """
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
        """Update the all_checkbox_states dict in app_state."""
        states = self.app_state.all_checkbox_states.copy()
        if is_checked:
            states[key] = True
        else:
            states.pop(key, None)
        self.app_state.all_checkbox_states = states

    def _restore_or_set_defaults(self):
        """Restore saved selections from app_state or set defaults from available options."""

        for key, vars in self.type_vars_dict.items():
            # Check IOWidget combos first, then DataWidget combos
            combo = self.io_widget.combos.get(key) or self.combos.get(key)

            if combo is not None:
                saved_value = self.app_state.get_key_sel(key) if self.app_state.key_sel_exists(key) else None
                vars_str = [str(var) for var in vars]

                if saved_value in vars_str:
                    combo.setCurrentText(str(saved_value))
                elif saved_value and key == "mics":
                    # Backwards compatibility: old yaml may have base name "mic1"
                    # but combo now has "mic1 (Ch 1)", "mic1 (Ch 2)", etc.
                    match = next((v for v in vars_str if v.startswith(str(saved_value))), None)
                    if match:
                        combo.setCurrentText(match)
                        self.app_state.set_key_sel(key, match)
                    else:
                        combo.setCurrentText(str(vars[0]))
                        self.app_state.set_key_sel(key, str(vars[0]))
                else:
                    combo.setCurrentText(str(vars[0]))
                    self.app_state.set_key_sel(key, str(vars[0]))


        if self.app_state.key_sel_exists("trials"):
            saved_trial = self.app_state.get_key_sel("trials")
            self.app_state.set_key_sel("trials", saved_trial)
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials_sel))
        else:
            # Default to first value
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials[0]))
            self.app_state.trials_sel = self.app_state.trials[0]
            
        # Restore space plot type
        space_plot_type = getattr(self.app_state, 'space_plot_type', 'None')
        if hasattr(self, 'space_plot_combo'):
            self.space_plot_combo.setCurrentText(space_plot_type)

        # Restore "All" checkbox states
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

                

    def _load_trial_with_fallback(self) -> None:
        first_trial = self.app_state.trials[0]
        current_trial = self.app_state.trials_sel

        try:
            self.on_trial_changed()
        except Exception as e:
            if current_trial == first_trial:
                raise RuntimeError(f"Failed to load first trial: {e}") from e

            print(f"Error loading trial {current_trial}: {e}\nReverting to first trial.")
            self.app_state.trials_sel = first_trial
            self.on_trial_changed()

    def on_trial_changed(self):
        """Handle all consequences of a trial change - centralized orchestration."""
        trials_sel = self.app_state.trials_sel

        self.app_state.ds = self.app_state.dt.trial(trials_sel)
        self.app_state.label_ds = self.app_state.label_dt.trial(trials_sel)

        if self.app_state.pred_dt is not None:
            self.app_state.pred_ds = self.app_state.pred_dt.trial(trials_sel)

        self.io_widget.update_device_combos_for_trial(self.app_state.ds)

        default_feature = self.combos["features"].itemText(0)
        feature_sel = getattr(self.app_state, 'features_sel', default_feature)
        if feature_sel == "Audio Waveform":
            self._set_audio_time()
        elif feature_sel:
            self.app_state.time = get_time_coord(self.app_state.ds[feature_sel])

        if hasattr(self, 'view_mode_combo'):
            view_mode = self.view_mode_combo.currentText()
            if view_mode.startswith("Spectrogram") and feature_sel != "Audio Waveform":
                source = self._build_xarray_source()
                if source is not None:
                    self.plot_container.spectrogram_plot.set_source(source)
            elif view_mode.startswith("Heatmap"):
                self.plot_container.heatmap_plot._clear_buffer()

        # Load interval-based labels for this trial
        self.app_state.label_intervals = self.app_state.get_trial_intervals(trials_sel)

        self.app_state.current_frame = 0
        self.update_video_audio()
        self.update_pose()
        self.update_label()
        self.update_main_plot()
        self.update_space_plot()
        self.app_state.verification_changed.emit()

    def update_main_plot(self, **kwargs):
        """Update the line plot or spectrogram with current trial/keypoint/variable selection."""
        import time
        if not self.app_state.ready:
            return

        ds_kwargs = self.app_state.get_ds_kwargs()
        current_plot = self.plot_container.get_current_plot()

        self.plot_container.clear_amplitude_envelope()
        if self.changepoints_widget:
            self.changepoints_widget.update_frequency_limits()

        try:

            current_plot.update_plot(**kwargs)
            self.update_label_plot(ds_kwargs)
   

            # Handle confidence plotting based on checkbox state
            if self.show_confidence_checkbox.isChecked():
                try:
                    label_confidence, _ = sel_valid(self.app_state.label_ds.labels_confidence, ds_kwargs)
                    self.plot_container.show_confidence_plot(label_confidence)
                except (KeyError, AttributeError):
                    pass
            else:
                self.plot_container.hide_confidence_plot()

            # Handle audio overlay
            if self.plot_container.is_lineplot():
 
                self.plot_container.update_audio_overlay()
         

        except (KeyError, AttributeError, ValueError) as e:
            show_error(f"Error updating plot: {e}")


    def update_label_plot(self, ds_kwargs):
        """Update label plot with interval-based labels and predictions."""
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
            # Predictions still use dense format — convert on demand
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








    def update_video_audio(self):
        """Update video and audio."""
        if not self.app_state.ready or not self.app_state.video_folder:
            return

        current_frame = getattr(self.app_state, 'current_frame', 0)

        # Set up video path
        if self.app_state.video_folder and hasattr(self.app_state, 'cameras_sel'):
            video_file = self.app_state.cameras_sel
            video_path = os.path.join(self.app_state.video_folder, video_file)
            self.app_state.video_path = os.path.normpath(video_path)

        # Set up audio path if available (uses centralized get_audio_source)
        has_audio = False
        if self.app_state.audio_folder and hasattr(self.app_state, 'mics_sel'):
            audio_path, _ = self.app_state.get_audio_source()
            if audio_path:
                self.app_state.audio_path = audio_path
                has_audio = True
            else:
                self.app_state.audio_path = None

        if self.audio_widget:
            self.audio_widget.set_enabled_state(has_audio=has_audio)

        if self.changepoints_widget:
            self.changepoints_widget.set_enabled_state(has_audio)

        if has_audio:
            self.show_waveform_checkbox.show()
            self.show_spectrogram_checkbox.show()
        else:
            self.show_waveform_checkbox.hide()
            self.show_spectrogram_checkbox.hide()

        # Pre-load new video data before any layer operations to avoid race conditions
        new_video_data = FastVideoReader(self.app_state.video_path, read_format='rgb24')

        # Force the reader to initialize by accessing shape (prevents lazy-load issues during render)
        _ = new_video_data.shape

        # Cleanup old video sync object first (but don't remove layer yet)
        if self.video:
            try:
                self.video.frame_changed.disconnect(self._on_sync_frame_changed)
                self.video.cleanup()
            except (RuntimeError, TypeError):
                pass
            self.video = None
            self.app_state.video = None

        # Add new video layer BEFORE removing old one to avoid race condition
        # where processEvents triggers a draw with no video layer present
        video_layer = self.viewer.add_image(new_video_data, name="video_new", rgb=True)

        # Now safely remove old video layers
        for layer in list(self.viewer.layers):
            if layer.name in ["video", "Video Stream"]:
                self.viewer.layers.remove(layer)

        # Rename and reposition the new layer
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

        self.video.seek_to_frame(current_frame)
        self.video.frame_changed.connect(self._on_sync_frame_changed)


    def update_label(self):
        """Update label display."""
        self.labels_widget.refresh_labels_shapes_layer()


    def toggle_pause_resume(self):
        """Toggle play/pause state of the video/audio stream."""
        if not self.video:
            return
        self.video.toggle_pause_resume()
        
        
    def _on_sync_frame_changed(self, frame_number: int):
        """Handle frame changes from sync manager."""
        self.app_state.current_frame = frame_number
        self.plot_container.update_time_marker_and_window(frame_number)
        

        current_time = frame_number / self.app_state.ds.fps
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
        self._set_common_color_property()
        self._set_text_property()
        self._add_points_layer()


        # connections = [("nose", "ear_left"), ("nose", "ear_right"), ("ear_left", "neck"), ("ear_right", "neck"),
        #                ("hip_left", "neck"), ("hip_right", "neck"), ("hip_left", "tail_base"), ("hip_right", "tail_base")]
        # self._add_skeleton_layer(connections)
        
        # COMMENTED OUT 
        # self._add_tracks_layer() 
        # if self.data_bboxes is not None:
        #     self._add_boxes_layer()
        self._set_initial_state()

            


    def closeEvent(self, event):
        """Clean up video stream and data cache."""

        SharedAudioCache.clear_cache()

        self.video.stop()

        super().closeEvent(event)


    def _on_space_plot_changed(self):
        """Handle space plot combo change."""
        if not self.app_state.ready:
            return
            
        plot_type = self.space_plot_combo.currentText()
        self.app_state.space_plot_type = plot_type
        self.update_space_plot()

    def update_space_plot(self):
        """Update space plot based on current selection."""
        if not self.app_state.ready:
            return

        plot_type = self.app_state.get_with_default('space_plot_type')

        if plot_type == "Layer controls":
            if self.space_plot:
                self.space_plot.hide()
        else:
            # Create space plot if it doesn't exist
            if not self.space_plot:
                self.space_plot = SpacePlot(self.viewer, self.app_state)
                if self.labels_widget:
                    self.labels_widget.highlight_spaceplot.connect(self._highlight_positions_in_space_plot)

            # Get current selections
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
        """Highlight positions in space plot based on current frame."""
        start_frame = int(start_time * self.app_state.ds.fps)
        end_frame = int(end_time * self.app_state.ds.fps)
        
        if self.space_plot and self.space_plot.dock_widget.isVisible():
            self.space_plot.highlight_positions(start_frame, end_frame)

    def reset_widget_state(self):
        """Reset the data widget to its default state."""
        # Clear all combo boxes in this widget
        for combo in self.combos.values():
            combo.clear()
            combo.addItems(["None"])
            combo.setCurrentText("None")
        
        # Reset checkboxes
        if hasattr(self, 'plot_spec_checkbox'):
            self.plot_spec_checkbox.setChecked(False)
        if hasattr(self, 'clear_audio_checkbox'):
            self.clear_audio_checkbox.setChecked(False)
        
        # Reset space plot combo if it exists
        if hasattr(self, 'space_plot_combo'):
            self.space_plot_combo.clear()
            self.space_plot_combo.addItems(["Layer controls"])
            self.space_plot_combo.setCurrentText("Layer controls")

        # Reset trial conditions combos in navigation widget
        if self.navigation_widget:
            self.navigation_widget.trial_conditions_combo.clear()
            self.navigation_widget.trial_conditions_combo.addItem("None")
            self.navigation_widget.trial_conditions_value_combo.clear()
            self.navigation_widget.trial_conditions_value_combo.addItem("None")
            
        # Clear navigation widget trials combo
        if self.navigation_widget and hasattr(self.navigation_widget, 'trials_combo'):
            self.navigation_widget.trials_combo.clear()
        
        # Reset various state variables
        self.type_vars_dict = {}
        self.video_path = None
        self.audio_path = None
        self.fps = None
        self.source_software = None
        self.file_path = None
        self.file_name = None
        
        # Hide space plot if it exists
        if self.space_plot:
            self.space_plot.hide()
            
        print("DataWidget state reset to default")
