"""Widget for input/output controls and data loading."""

import os
from pathlib import Path
from typing import Optional

import numpy as np

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ethograph import TrialTree
from ethograph.utils.paths import gui_default_settings_path

from .app_state import AppStateSpec
from .dialog_create_nc import CreateNCDialog
from .dialog_select_template import TemplateDialog

class IOWidget(QWidget):
    """Widget to control I/O paths, device selection, and data loading."""

    def __init__(self, app_state, data_widget, labels_widget, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.data_widget = data_widget
        self.labels_widget = labels_widget
        layout = QFormLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)

        # Dictionary to store combo boxes
        self.combos = {}
        # List to store controls for enabling/disabling
        self.controls = []
        
        

        self.reset_button = QPushButton("💡Reset gui_settings.yaml")
        self.reset_button.setObjectName("reset_button")
        self.reset_button.clicked.connect(self._on_reset_gui_clicked)

        self.create_nc_button = QPushButton("➕Create with own data")
        self.create_nc_button.setObjectName("create_nc_button")
        self.create_nc_button.clicked.connect(self._on_create_nc_clicked)

        self.template_button = QPushButton("📋Select template data")
        self.template_button.setObjectName("template_button")
        self.template_button.clicked.connect(self._on_select_template_clicked)

        button_row = QHBoxLayout()
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.create_nc_button)
        button_row.addWidget(self.template_button)
        self.layout().addRow(button_row)

        self._create_path_folder_widgets()
        self._create_device_combos()

        
        self._create_load_button()

        # Restore UI text fields from app state
        if self.app_state.nc_file_path:
            self.nc_file_path_edit.setText(self.app_state.nc_file_path)
        if self.app_state.video_folder:
            self.video_folder_edit.setText(self.app_state.video_folder)
        if self.app_state.audio_folder:
            self.audio_folder_edit.setText(self.app_state.audio_folder)
        if self.app_state.pose_folder:
            self.pose_folder_edit.setText(self.app_state.pose_folder)
        if self.app_state.ephys_folder:
            self.ephys_folder_edit.setText(self.app_state.ephys_folder)

    def _on_reset_gui_clicked(self):
        """Reset the GUI to its initial state."""

        self.downsample_checkbox.setChecked(False)

        self.app_state.delete_yaml()

        # Reset all app_state attributes to their defaults
        for var, (_, default, _) in AppStateSpec.VARS.items():
            setattr(self.app_state, var, default)

        # Clear all dynamic _sel attributes
        for attr in list(dir(self.app_state)):
            if attr.endswith("_sel") or attr.endswith("_sel_previous"):
                try:
                    delattr(self.app_state, attr)
                except AttributeError:
                    pass

        self._clear_all_line_edits()
        self._clear_combo_boxes()

        yaml_path = gui_default_settings_path()
        self.app_state._yaml_path = str(yaml_path)
        self.app_state.save_to_yaml()

    def _on_create_nc_clicked(self):
        """Open the dialog to create a .nc file from various data sources."""
        self._on_reset_gui_clicked()
        
        dialog = CreateNCDialog(self.app_state, self, self)
        dialog.exec_()

    def _on_select_template_clicked(self):
        self._on_reset_gui_clicked()
        
        
        
        dialog = TemplateDialog(self)
        if dialog.exec_() and dialog.selected_template:
            t = dialog.selected_template
            if t["nc_file_path"]:
                self.nc_file_path_edit.setText(t["nc_file_path"])
                self.app_state.nc_file_path = t["nc_file_path"]
            if t["video_folder"]:
                self.video_folder_edit.setText(t["video_folder"])
                self.app_state.video_folder = t["video_folder"]
            if t["audio_folder"]:
                self.audio_folder_edit.setText(t["audio_folder"])
                self.app_state.audio_folder = t["audio_folder"]
            if t["pose_folder"]:
                self.pose_folder_edit.setText(t["pose_folder"])
                self.app_state.pose_folder = t["pose_folder"]
            if t.get("import_labels"):
                self.import_labels_checkbox.setChecked(True)
            if t.get("dataset_key") == "birdpark":
                self.downsample_checkbox.setChecked(True)
                self.downsample_spin.setValue(100)

    def _clear_all_line_edits(self):
        """Clear all QLineEdit fields in the widget."""
        if hasattr(self, 'nc_file_path_edit'):
            self.nc_file_path_edit.clear()
        if hasattr(self, 'video_folder_edit'):
            self.video_folder_edit.clear()
        if hasattr(self, 'audio_folder_edit'):
            self.audio_folder_edit.clear()
        if hasattr(self, 'pose_folder_edit'):
            self.pose_folder_edit.clear()
        if hasattr(self, 'ephys_folder_edit'):
            self.ephys_folder_edit.clear()

    def _clear_combo_boxes(self):
        """Reset all combo boxes to default state."""
        for combo in self.combos.values():
            combo.clear()
            combo.addItems(["None"])
            combo.setCurrentText("None")

    def _create_path_widget(self, label: str, object_name: str, browse_callback):
        """Generalized function to create a line edit and browse button for file/folder paths."""
        
        
        line_edit = QLineEdit()
        line_edit.setObjectName(f"{object_name}_edit")
        if object_name == "labels_path":
            self.label_file_path_edit = line_edit
            if self.app_state.import_labels_nc_data:
                self.label_file_path_edit.setText(self.app_state.nc_file_path)

        browse_button = QPushButton("Browse")
        browse_button.setObjectName(f"{object_name}_browse_button")
        browse_button.clicked.connect(browse_callback)
            


        if object_name == "nc_file_path":
            self.import_labels_checkbox = QCheckBox("Import labels")
            self.import_labels_checkbox.setObjectName("import_labels_checkbox")
            self.import_labels_checkbox.stateChanged.connect(
                lambda state: setattr(self.app_state, 'import_labels_nc_data', state == 2)
            )
            self.import_labels_checkbox.setChecked(bool(self.app_state.import_labels_nc_data))

        clear_button = QPushButton("Clear")
        clear_button.setObjectName(f"{object_name}_clear_button")
        clear_button.clicked.connect(lambda: self._on_clear_path_clicked(object_name, line_edit))

        layout = QHBoxLayout()
        layout.addWidget(line_edit)
        layout.addWidget(browse_button)
        if object_name == "nc_file_path":
            layout.addWidget(self.import_labels_checkbox)
        layout.addWidget(clear_button)
        self.layout().addRow(label, layout)

        return line_edit

    def _on_clear_path_clicked(self, object_name: str, line_edit: QLineEdit):
        """Clear the path field and corresponding app state value."""
        line_edit.setText("")

        if object_name == "nc_file_path":
            self.app_state.nc_file_path = None
        elif object_name == "video_folder":
            self.app_state.video_folder = None
        elif object_name == "audio_folder":
            self.app_state.audio_folder = None
        elif object_name == "pose_folder":
            self.app_state.pose_folder = None
        elif object_name == "ephys_folder":
            self.app_state.ephys_folder = None

    def _create_path_folder_widgets(self):
        """Create file path, video folder, and audio folder selectors."""
        self.nc_file_path_edit = self._create_path_widget(
            label="Get sesssion:",
            object_name="nc_file_path",
            browse_callback=lambda: self.on_browse_clicked("file", "data"),
        )
        
        self.video_folder_edit = self._create_path_widget(
            label="Video folder:",
            object_name="video_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "video"),
        )
        

        self.pose_folder_edit = self._create_path_widget(
            label="Pose folder:",
            object_name="pose_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "pose"),
        )
        
        self.audio_folder_edit = self._create_path_widget(
            label="Audio folder:",
            object_name="audio_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "audio"),
        )

        self.ephys_folder_edit = self._create_path_widget(
            label="Ephys folder:",
            object_name="ephys_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "ephys"),
        )



    def _create_device_combos(self):
        """Create device combo boxes (cameras, mics, pose). Called after data is loaded."""
        pass  # Will be populated when data is loaded

    def create_device_controls(self, type_vars_dict):
        """Create device combo boxes based on loaded data."""
        # Clear existing device controls


        self.labels_file_edit = self._create_path_widget(
            label="Get ./labels.nc:",
            object_name="labels_path",
            browse_callback=lambda: self.on_browse_clicked("file", "labels"),
        )    
        self.controls.append(self.labels_file_edit)
        
        
        



        for key in ['cameras', 'mics', 'pose']:
            if key in self.combos:
                combo = self.combos[key]
                combo.setParent(None)
                combo.deleteLater()
                del self.combos[key]
                if combo in self.controls:
                    self.controls.remove(combo)

        for key, folder_attr in [("cameras", "video_folder"), ("mics", "audio_folder"), ("pose", "pose_folder")]:
            if key in type_vars_dict and getattr(self.app_state, folder_attr):
                self._create_combo_widget(key, type_vars_dict[key])



    def _get_audio_channel_count(self, audio_path: str) -> int:
        """Detect number of channels in an audio file."""
        try:
            from audioio import AudioLoader
            with AudioLoader(audio_path) as loader:
                if loader.shape is None:
                    return 1
                return loader.channels if hasattr(loader, 'channels') else (loader.shape[1] if len(loader.shape) > 1 else 1)
        except Exception:
            return 1

    def _expand_mics_with_channels(self, mic_names) -> list[str]:
        """Expand mic names to include channel info for multi-channel files.

        Returns list of display names and populates app_state.audio_source_map.
        """
        self.app_state.audio_source_map.clear()
        expanded_items = []

        audio_folder = self.app_state.audio_folder
        ds = getattr(self.app_state, 'ds', None)

        if not audio_folder or ds is None:
            for mic in mic_names:
                display_name = str(mic)
                self.app_state.audio_source_map[display_name] = (str(mic), 0)
                expanded_items.append(display_name)
            return expanded_items

        for mic in mic_names:
            mic_file = str(mic)
            try:
                audio_path = os.path.join(audio_folder, mic_file)
                n_channels = self._get_audio_channel_count(audio_path)

                if n_channels > 1:
                    for ch in range(n_channels):
                        display_name = f"{mic_file} (Ch {ch + 1})"
                        self.app_state.audio_source_map[display_name] = (mic_file, ch)
                        expanded_items.append(display_name)
                else:
                    self.app_state.audio_source_map[mic_file] = (mic_file, 0)
                    expanded_items.append(mic_file)
            except Exception:
                self.app_state.audio_source_map[mic_file] = (mic_file, 0)
                expanded_items.append(mic_file)

        return expanded_items

    def _expand_ephys_with_streams(self, ephys_folder, ds) -> list[str]:
        """Discover ephys files and expand multi-stream files into feature entries.

        Scans ephys_folder for known ephys extensions, probes each file for
        streams, and populates app_state.ephys_source_map.

        Returns list of feature display names (e.g. "Amplifier Waveform").
        """
        from .plots_ephystrace import GenericEphysLoader

        self.app_state.ephys_source_map.clear()
        feature_names = []

        ephys_files_attr = ds.attrs.get("ephys", []) if ds is not None else []
        if isinstance(ephys_files_attr, str):
            ephys_files_attr = [ephys_files_attr]

        if ephys_files_attr:
            candidates = [os.path.join(ephys_folder, f) for f in ephys_files_attr]
        else:
            known_exts = set(GenericEphysLoader.KNOWN_EXTENSIONS.keys()) | {".dat", ".bin", ".raw"}
            candidates = []
            try:
                for entry in os.scandir(ephys_folder):
                    if entry.is_file() and Path(entry.name).suffix.lower() in known_exts:
                        candidates.append(entry.path)
            except OSError:
                return feature_names

        for filepath in candidates:
            filename = Path(filepath).name
            ext = Path(filepath).suffix.lower()

            if ext in (".dat", ".bin", ".raw"):
                continue

            try:
                loader = GenericEphysLoader(filepath, stream_id="0")
                streams = loader.streams

                if streams and len(streams) > 1:
                    for sid, info in streams.items():
                        stream_name = info["name"]
                        display_name = f"{stream_name} Waveform"
                        self.app_state.ephys_source_map[display_name] = (filename, sid, 0)
                        feature_names.append(display_name)
                else:
                    display_name = "Ephys Waveform"
                    self.app_state.ephys_source_map[display_name] = (filename, "0", 0)
                    feature_names.append(display_name)
            except (OSError, IOError, ValueError) as e:
                print(f"Skipping ephys file {filename}: {e}")

        return feature_names

    def _create_combo_widget(self, key, vars):
        """Create a combo box widget for a given info key."""
        combo = QComboBox()
        combo.setObjectName(f"{key}_combo")
        combo.currentTextChanged.connect(self._on_combo_changed)

        if key in ("cameras", "pose"):
            combo.currentIndexChanged.connect(
                lambda idx, src=key: self._sync_camera_pose(src, idx)
            )

        if key == "mics":
            expanded_items = self._expand_mics_with_channels(vars)
            combo.addItems(expanded_items)
        else:
            combo.addItems([str(var) for var in vars])

        self.layout().addRow(f"{key.capitalize()}:", combo)
        self.combos[key] = combo
        self.controls.append(combo)
        return combo

    def _sync_camera_pose(self, source: str, index: int):
        target = "pose" if source == "cameras" else "cameras"
        target_combo = self.combos.get(target)
        source_combo = self.combos.get(source)
        if target_combo is None or source_combo is None:
            return
        if source_combo.count() != target_combo.count():
            return
        if target_combo.currentIndex() == index:
            return
        target_combo.blockSignals(True)
        target_combo.setCurrentIndex(index)
        target_combo.blockSignals(False)
        self.app_state.set_key_sel(target, target_combo.currentText())

    def update_device_combos_for_trial(self, ds):
        """Update device combo boxes to reflect the current trial's file lists."""
        for key in ['cameras', 'mics', 'pose']:
            combo = self.combos.get(key)
            if combo is None:
                continue

            new_items = ds.attrs.get(key)
            if new_items is None:
                continue
            new_items = np.atleast_1d(new_items).astype(str)

            prev_index = combo.currentIndex()

            combo.blockSignals(True)
            combo.clear()

            if key == "mics":
                expanded = self._expand_mics_with_channels(new_items)
                combo.addItems(expanded)
            else:
                combo.addItems(list(new_items))

            if prev_index < combo.count():
                combo.setCurrentIndex(prev_index)
            else:
                combo.setCurrentIndex(0)

            combo.blockSignals(False)

            self.app_state.set_key_sel(key, combo.currentText())

    def _on_combo_changed(self):
        """Handle combo box changes and delegate to data widget."""
        if hasattr(self.data_widget, '_on_combo_changed'):
            self.data_widget._on_combo_changed()

    def set_controls_enabled(self, enabled: bool):
        """Enable or disable device controls."""
        for control in self.controls:
            control.setEnabled(enabled)

    def _create_load_button(self):
        """Create a button to load the file to the viewer with optional downsampling."""
        load_layout = QHBoxLayout()

        self.downsample_checkbox = QCheckBox("Downsample:")
        self.downsample_checkbox.setObjectName("downsample_checkbox")
        self.downsample_checkbox.setChecked(self.app_state.downsample_enabled)
        self.downsample_checkbox.setToolTip("Downsample data on load for faster display")
        self.downsample_checkbox.toggled.connect(self._on_downsample_toggled)

        self.downsample_spin = QSpinBox()
        self.downsample_spin.setObjectName("downsample_spin")
        self.downsample_spin.setRange(2, 1000)
        self.downsample_spin.setValue(self.app_state.downsample_factor)
        self.downsample_spin.setEnabled(self.app_state.downsample_enabled)
        self.downsample_spin.setToolTip("Downsample factor (e.g., 100 = keep 1 in 100 samples)")
        self.downsample_spin.setFixedWidth(70)
        self.downsample_spin.valueChanged.connect(self._on_downsample_value_changed)

        self.load_button = QPushButton("Load")
        self.load_button.setObjectName("load_button")
        self.load_button.clicked.connect(lambda: self.data_widget.on_load_clicked())

        load_layout.addWidget(self.downsample_checkbox)
        load_layout.addWidget(self.downsample_spin)
        load_layout.addWidget(self.load_button, stretch=1)

        self.layout().addRow(load_layout)

    def _on_downsample_toggled(self, checked: bool):
        """Enable/disable downsample spinbox based on checkbox state."""
        self.downsample_spin.setEnabled(checked)
        self.app_state.downsample_enabled = checked

    def _on_downsample_value_changed(self, value: int):
        """Update app_state when downsample factor changes."""
        self.app_state.downsample_factor = value

    def disable_downsample_controls(self):
        """Disable downsample controls after data is loaded."""
        self.downsample_checkbox.setEnabled(False)
        self.downsample_spin.setEnabled(False)

    def get_downsample_factor(self) -> Optional[int]:
        """Get the downsample factor if enabled, else None."""
        if self.downsample_checkbox.isChecked():
            return self.downsample_spin.value()
        return None

    def on_browse_clicked(self, browse_type: str = "file", media_type: str | None = None):
        """
        Open a file or folder dialog to select a file or folder.

        Args:
            browse_type: "file" for file dialog, "folder" for folder dialog.
            media_type: "video" or "audio" (used for folder dialog caption).
        """
        if browse_type == "file":
            if media_type == "data":
                result = QFileDialog.getOpenFileName(
                    None,
                    caption="Open file containing feature data",
                    filter="NetCDF files (*.nc)",
                )
                nc_file_path = result[0] if result and len(result) >= 1 else ""
                if not nc_file_path:
                    return

                self.nc_file_path_edit.setText(nc_file_path)
                self.app_state.nc_file_path = nc_file_path
                
            elif media_type == "labels":
                nc_parent = Path(self.app_state.nc_file_path).parent


                result = QFileDialog.getOpenFileName(
                    None,
                    caption="Open file in ./labels/data_labels.nc",
                    dir=str(nc_parent),
                    filter="NetCDF files (*.nc)",
                )
                labels_file_path = result[0] if result and len(result) >= 1 else ""
                if not labels_file_path:
                    return

                if labels_file_path:
                    label_dt_full = TrialTree.open(labels_file_path)
                    self.app_state.label_dt = label_dt_full.get_label_dt()
                    self.app_state.label_ds = self.app_state.label_dt.trial(self.app_state.trials_sel)
                    self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)

                    self.label_file_path_edit.setText(labels_file_path)

                    self.changepoints_widget._update_cp_status()
                    self.labels_widget._mark_changes_unsaved()
                    self.app_state.verification_changed.emit()
                    self.app_state.labels_modified.emit()
                    self.labels_widget.refresh_labels_shapes_layer()

    
            

        elif browse_type == "folder":
            if media_type == "video":
                caption = "Open folder with video files (e.g. mp4, mov)."
            elif media_type == "audio":
                caption = "Open folder with audio files (e.g. wav, mp3, mp4)."
            elif media_type == "pose":
                caption = "Open folder with pose files (e.g. .csv, .h5)."
            elif media_type == "ephys":
                caption = "Open folder with ephys files (e.g. .rhd, .edf, .dat)."

            folder_path = QFileDialog.getExistingDirectory(None, caption=caption)

            if media_type == "video":
                self.video_folder_edit.setText(folder_path)
                self.app_state.video_folder = folder_path
            elif media_type == "audio":
                self.audio_folder_edit.setText(folder_path)
                self.app_state.audio_folder = folder_path
                if hasattr(self.data_widget, 'clear_audio_checkbox'):
                    self.data_widget.clear_audio_checkbox.setChecked(False)
            elif media_type == "pose":
                self.pose_folder_edit.setText(folder_path)
                self.app_state.pose_folder = folder_path
            elif media_type == "ephys":
                self.ephys_folder_edit.setText(folder_path)
                self.app_state.ephys_folder = folder_path

    def get_nc_file_path(self):
        """Get the current NetCDF file path from the text field."""
        return self.nc_file_path_edit.text()