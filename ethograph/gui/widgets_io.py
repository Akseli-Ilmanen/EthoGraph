
"""Widget for input/output controls and data loading."""

from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
    QCheckBox,
    QSpinBox,
)
from .app_state import AppStateSpec
from .dialog_create_nc import CreateNCDialog
from pathlib import Path
import os
from qtpy.QtCore import Qt
from ethograph.utils.io import TrialTree
from ethograph.utils.paths import gui_default_settings_path
from typing import Optional

class IOWidget(QWidget):
    """Widget to control I/O paths, device selection, and data loading."""

    def __init__(self, app_state, data_widget, labels_widget, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.data_widget = data_widget
        self.labels_widget = labels_widget
        self.setLayout(QFormLayout())
        
        # Dictionary to store combo boxes
        self.combos = {}
        # List to store controls for enabling/disabling
        self.controls = []
        # Mapping from display name to (mic_name, channel_idx) for multi-channel audio
        self.mic_channel_map = {}
        
        

        self.reset_button = QPushButton("💡Reset gui_settings.yaml")
        self.reset_button.setObjectName("reset_button")
        self.reset_button.clicked.connect(self._on_reset_gui_clicked)
        self.layout().addRow(self.reset_button)

        self.create_nc_button = QPushButton("➕Create session.nc file with own data")
        self.create_nc_button.setObjectName("create_nc_button")
        self.create_nc_button.clicked.connect(self._on_create_nc_clicked)
        self.layout().addRow(self.create_nc_button)

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
        if self.app_state.tracking_folder:
            self.tracking_folder_edit.setText(self.app_state.tracking_folder)

    def _on_reset_gui_clicked(self):
        """Reset the GUI to its initial state."""


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
        dialog = CreateNCDialog(self.app_state, self, self)
        dialog.exec_()

    def _clear_all_line_edits(self):
        """Clear all QLineEdit fields in the widget."""
        if hasattr(self, 'nc_file_path_edit'):
            self.nc_file_path_edit.clear()
        if hasattr(self, 'video_folder_edit'):
            self.video_folder_edit.clear()
        if hasattr(self, 'audio_folder_edit'):
            self.audio_folder_edit.clear()
        if hasattr(self, 'tracking_folder_edit'):
            self.tracking_folder_edit.clear()

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
            import_labels_checkbox = QCheckBox("Import labels")
            import_labels_checkbox.setObjectName("import_labels_checkbox")
            import_labels_checkbox.setChecked(self.app_state.import_labels_nc_data)
            import_labels_checkbox.stateChanged.connect(lambda state: setattr(self.app_state, 'import_labels_nc_data', state == Qt.Checked))
            


        clear_button = QPushButton("Clear")
        clear_button.setObjectName(f"{object_name}_clear_button")
        clear_button.clicked.connect(lambda: self._on_clear_path_clicked(object_name, line_edit))

        layout = QHBoxLayout()
        layout.addWidget(line_edit)
        layout.addWidget(browse_button)
        if object_name == "nc_file_path":
            layout.addWidget(import_labels_checkbox)
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
        elif object_name == "tracking_folder":
            self.app_state.tracking_folder = None

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
        

        self.tracking_folder_edit = self._create_path_widget(
            label="Tracking folder:",
            object_name="tracking_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "tracking"),
        )
        
        self.audio_folder_edit = self._create_path_widget(
            label="Audio folder:",
            object_name="audio_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "audio"),
        )



    def _create_device_combos(self):
        """Create device combo boxes (cameras, mics, tracking). Called after data is loaded."""
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
        
        
        



        for key in ['cameras', 'mics', 'tracking']:
            if key in self.combos:
                combo = self.combos[key]
                combo.setParent(None)
                combo.deleteLater()
                del self.combos[key]
                if combo in self.controls:
                    self.controls.remove(combo)

        if hasattr(self, '_device_row_layout'):
            while self._device_row_layout.count():
                self._device_row_layout.takeAt(0)
        else:
            self._device_row_layout = QHBoxLayout()
            self.layout().addRow(self._device_row_layout)

        for key, folder_attr in [("cameras", "video_folder"), ("mics", "audio_folder"), ("tracking", "tracking_folder")]:
            if key in type_vars_dict and getattr(self.app_state, folder_attr):
                self._create_combo_widget(key, type_vars_dict[key], self._device_row_layout)



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

        Returns list of display names and populates self.mic_channel_map.
        """
        self.mic_channel_map.clear()
        expanded_items = []

        audio_folder = self.app_state.audio_folder
        ds = getattr(self.app_state, 'ds', None)

        if not audio_folder or ds is None:
            for mic in mic_names:
                display_name = str(mic)
                self.mic_channel_map[display_name] = (str(mic), 0)
                expanded_items.append(display_name)
            return expanded_items

        for mic in mic_names:
            mic_str = str(mic)
            try:
                audio_file = ds.attrs.get(mic_str)
                if not audio_file:
                    display_name = mic_str
                    self.mic_channel_map[display_name] = (mic_str, 0)
                    expanded_items.append(display_name)
                    continue

                audio_path = os.path.join(audio_folder, audio_file)
                n_channels = self._get_audio_channel_count(audio_path)

                if n_channels > 1:
                    for ch in range(n_channels):
                        display_name = f"{mic_str} (Ch {ch + 1})"
                        self.mic_channel_map[display_name] = (mic_str, ch)
                        expanded_items.append(display_name)
                else:
                    display_name = mic_str
                    self.mic_channel_map[display_name] = (mic_str, 0)
                    expanded_items.append(display_name)
            except Exception:
                display_name = mic_str
                self.mic_channel_map[display_name] = (mic_str, 0)
                expanded_items.append(display_name)

        return expanded_items

    def get_mic_and_channel(self, display_name: str) -> tuple[str, int]:
        """Get (mic_name, channel_idx) from a mic combo display name."""
        return self.mic_channel_map.get(display_name, (display_name, 0))

    def _create_combo_widget(self, key, vars, layout):
        """Create a combo box widget for a given info key."""
        combo = QComboBox()
        combo.setObjectName(f"{key}_combo")
        combo.currentTextChanged.connect(self._on_combo_changed)

        if key == "mics":
            expanded_items = self._expand_mics_with_channels(vars)
            combo.addItems(expanded_items)
        else:
            combo.addItems([str(var) for var in vars])

        layout.addWidget(QLabel(f"{key.capitalize()}:"))
        layout.addWidget(combo)
        self.combos[key] = combo
        self.controls.append(combo)
        return combo

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
        self.downsample_checkbox.setChecked(False)
        self.downsample_checkbox.setToolTip("Downsample data on load for faster display")
        self.downsample_checkbox.toggled.connect(self._on_downsample_toggled)

        self.downsample_spin = QSpinBox()
        self.downsample_spin.setObjectName("downsample_spin")
        self.downsample_spin.setRange(2, 1000)
        self.downsample_spin.setValue(100)
        self.downsample_spin.setEnabled(False)
        self.downsample_spin.setToolTip("Downsample factor (e.g., 100 = keep 1 in 100 samples)")
        self.downsample_spin.setFixedWidth(70)

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
                result = QFileDialog.getOpenFileName(
                    None,
                    caption="Open file in ./labels/data_labels.nc",
                    filter="NetCDF files (*.nc)",
                )
                labels_file_path = result[0] if result and len(result) >= 1 else ""
                if not labels_file_path:
                    return

                if labels_file_path:
                    self.app_state.label_dt = TrialTree.load(labels_file_path)
                    self.app_state.label_ds = self.app_state.label_dt.trial(self.app_state.trials_sel)

                    self.label_file_path_edit.setText(labels_file_path)

                    self.labels_widget._update_cp_status()
                    self.labels_widget._mark_changes_unsaved()
                    self.app_state.verification_changed.emit()
                    self.app_state.labels_modified.emit()
                    self.labels_widget.refresh_motif_shapes_layer()

    
            

        elif browse_type == "folder":
            if media_type == "video":
                caption = "Open folder with video files (e.g. mp4, mov)."
            elif media_type == "audio":
                caption = "Open folder with audio files (e.g. wav, mp3, mp4)."
            elif media_type == "tracking":
                caption = "Open folder with tracking files (e.g. .csv, .h5)."

            folder_path = QFileDialog.getExistingDirectory(None, caption=caption)

            if media_type == "video":
                self.video_folder_edit.setText(folder_path)
                self.app_state.video_folder = folder_path
            elif media_type == "audio":
                self.audio_folder_edit.setText(folder_path)
                self.app_state.audio_folder = folder_path
                # Clear audio checkbox if it exists in data_widget
                if hasattr(self.data_widget, 'clear_audio_checkbox'):
                    self.data_widget.clear_audio_checkbox.setChecked(False)
            elif media_type == "tracking":
                self.tracking_folder_edit.setText(folder_path)
                self.app_state.tracking_folder = folder_path

    def get_nc_file_path(self):
        """Get the current NetCDF file path from the text field."""
        return self.nc_file_path_edit.text()