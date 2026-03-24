"""Widget for input/output controls and data loading."""

import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import ethograph as eto
from ethograph.utils.paths import find_mapping_file, gui_default_settings_path
from ethograph.utils.validation import EPHYS_FILE_FILTER

from .app_state import AppStateSpec
from .wizard_overview import NCWizardDialog
from .makepretty import ElidedDelegate, clean_display_labels
from .dialog_select_template import TemplateDialog


class IOWidget(QWidget):
    """Widget to control I/O paths, device selection, and data loading."""

    def __init__(self, app_state, data_widget, labels_widget, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.data_widget = data_widget
        self.labels_widget = labels_widget

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self.combos = {}
        self.controls = []

        self._create_toggle_buttons(main_layout)
        self._create_load_panel(main_layout)
        self._create_controls_panel(main_layout)
        self._wire_app_state_path_signals()
        self._wire_path_edit_signals()

        # Initial state: load tab active, controls greyed out
        self.controls_toggle.setEnabled(False)
        self._show_panel("load")

        # Restore UI text fields from app state
        if self.app_state.nc_file_path:
            self.nc_file_path_edit.setText(self.app_state.nc_file_path)
        if self.app_state.video_folder:
            self.video_folder_edit.setText(self.app_state.video_folder)
        if self.app_state.audio_folder:
            self.audio_folder_edit.setText(self.app_state.audio_folder)
        if self.app_state.pose_folder:
            self.pose_folder_edit.setText(self.app_state.pose_folder)
        if self.app_state.ephys_path:
            self.ephys_path_edit.setText(self.app_state.ephys_path)
        if self.app_state.kilosort_folder:
            self.kilosort_folder_edit.setText(self.app_state.kilosort_folder)

    def _wire_app_state_path_signals(self):
        self.app_state.nc_file_path_changed.connect(
            lambda value: self.nc_file_path_edit.setText(value or "")
        )
        self.app_state.video_folder_changed.connect(
            lambda value: self.video_folder_edit.setText(value or "")
        )
        self.app_state.audio_folder_changed.connect(
            lambda value: self.audio_folder_edit.setText(value or "")
        )
        self.app_state.pose_folder_changed.connect(
            lambda value: self.pose_folder_edit.setText(value or "")
        )
        self.app_state.ephys_path_changed.connect(
            lambda value: self.ephys_path_edit.setText(value or "")
        )
        self.app_state.kilosort_folder_changed.connect(
            lambda value: self.kilosort_folder_edit.setText(value or "")
        )
        self.app_state.remote_video_changed.connect(self.remote_video_checkbox.setChecked)

    def _wire_path_edit_signals(self):
        self.nc_file_path_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.nc_file_path_edit, "nc_file_path")
        )
        self.video_folder_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.video_folder_edit, "video_folder")
        )
        self.audio_folder_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.audio_folder_edit, "audio_folder")
        )
        self.pose_folder_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.pose_folder_edit, "pose_folder")
        )
        self.ephys_path_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.ephys_path_edit, "ephys_path")
        )
        self.kilosort_folder_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.kilosort_folder_edit, "kilosort_folder")
        )

    def _sync_line_edit_to_state(self, line_edit, attr_name):
        value = line_edit.text().strip() or None
        if getattr(self.app_state, attr_name, None) != value:
            setattr(self.app_state, attr_name, value)

    # ------------------------------------------------------------------
    # Toggle buttons
    # ------------------------------------------------------------------

    def _create_toggle_buttons(self, main_layout):
        toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_widget.setLayout(toggle_layout)

        self.load_toggle = QPushButton("Load data")
        self.load_toggle.setCheckable(True)
        self.load_toggle.clicked.connect(self._toggle_load)
        toggle_layout.addWidget(self.load_toggle)

        self.controls_toggle = QPushButton("I/O controls")
        self.controls_toggle.setCheckable(True)
        self.controls_toggle.clicked.connect(self._toggle_controls)
        toggle_layout.addWidget(self.controls_toggle)

        main_layout.addWidget(toggle_widget)

    def _show_panel(self, panel_name):
        panels = {
            "load": (self.load_panel, self.load_toggle),
            "controls": (self.controls_panel, self.controls_toggle),
        }
        for name, (panel, toggle) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)

    def _toggle_load(self):
        if self.load_toggle.isChecked():
            self._show_panel("load")
        else:
            if self.controls_toggle.isEnabled():
                self._show_panel("controls")
            else:
                self.load_toggle.setChecked(True)

    def _toggle_controls(self):
        self._show_panel("controls" if self.controls_toggle.isChecked() else "load")

    # ------------------------------------------------------------------
    # Load panel
    # ------------------------------------------------------------------

    def _create_load_panel(self, main_layout):
        self.load_panel = QWidget()
        self._load_layout = QFormLayout()
        self._load_layout.setSpacing(2)
        self._load_layout.setContentsMargins(0, 0, 0, 0)
        self.load_panel.setLayout(self._load_layout)

        # Button row
        self.reset_button = QPushButton("💡Reset gui_settings.yaml")
        self.reset_button.setObjectName("reset_button")
        self.reset_button.clicked.connect(self._on_reset_gui_clicked)

        self.create_nc_button = QPushButton("➕Create with own data")
        self.create_nc_button.setObjectName("create_nc_button")
        self.create_nc_button.clicked.connect(self._on_create_nc_clicked)

        self.template_button = QPushButton("📋Select templates")
        self.template_button.setObjectName("template_button")
        self.template_button.clicked.connect(self._on_select_template_clicked)

        button_row = QHBoxLayout()
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.create_nc_button)
        button_row.addWidget(self.template_button)
        self._load_layout.addRow(button_row)

        # Path widgets
        self.nc_file_path_edit = self._create_path_widget(
            self._load_layout,
            label="Get sesssion:",
            object_name="nc_file_path",
            browse_callback=lambda: self.on_browse_clicked("file", "data"),
        )
        self.video_folder_edit = self._create_path_widget(
            self._load_layout,
            label="Video folder:",
            object_name="video_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "video"),
        )
        self.remote_video_checkbox.setChecked(bool(self.app_state.remote_video))
        self.pose_folder_edit = self._create_path_widget(
            self._load_layout,
            label="Pose folder:",
            object_name="pose_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "pose"),
        )

        self.audio_folder_edit = self._create_path_widget(
            self._load_layout,
            label="Audio folder:",
            object_name="audio_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "audio"),
        )
        self.ephys_path_edit = self._create_path_widget(
            self._load_layout,
            label="Ephys file:",
            object_name="ephys_path",
            browse_callback=lambda: self.on_browse_clicked("file", "ephys"),
        )

        self.kilosort_folder_edit = self._create_path_widget(
            self._load_layout,
            label="Kilosort folder:",
            object_name="kilosort_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "kilosort"),
        )

        # Downsample + Load button
        self._create_load_button(self._load_layout)

        main_layout.addWidget(self.load_panel)

    # ------------------------------------------------------------------
    # Controls panel
    # ------------------------------------------------------------------

    def _create_controls_panel(self, main_layout):
        self.controls_panel = QWidget()
        self._controls_layout = QFormLayout()
        self._controls_layout.setSpacing(2)
        self._controls_layout.setContentsMargins(0, 0, 0, 0)
        self.controls_panel.setLayout(self._controls_layout)

        self._create_mapping_row(self._controls_layout)
        self._create_predictions_row(self._controls_layout)

        main_layout.addWidget(self.controls_panel)

    def _create_mapping_row(self, target_layout):
        mapping_row = QWidget()
        mapping_layout = QHBoxLayout()
        mapping_layout.setContentsMargins(0, 0, 0, 0)
        mapping_row.setLayout(mapping_layout)

        self.mapping_file_path_edit = QLineEdit()
        default_mapping = find_mapping_file()
        self.mapping_file_path_edit.setText(str(default_mapping) if default_mapping else "")
        self.mapping_file_path_edit.setToolTip("Path to mapping.txt file")
        mapping_layout.addWidget(self.mapping_file_path_edit)

        self.browse_mapping_btn = QPushButton("Browse")
        mapping_layout.addWidget(self.browse_mapping_btn)

        self.temp_labels_button = QPushButton("Create temporary labels")
        self.temp_labels_button.setToolTip("Create custom labels for this session only")
        mapping_layout.addWidget(self.temp_labels_button)

        target_layout.addRow("Mapping:", mapping_row)

    def _create_predictions_row(self, target_layout):
        pred_row = QWidget()
        pred_layout = QHBoxLayout()
        pred_layout.setContentsMargins(0, 0, 0, 0)
        pred_row.setLayout(pred_layout)

        pred_layout.addWidget(QLabel("Predictions:"))

        self.pred_file_path_edit = QLineEdit()
        self.pred_file_path_edit.setReadOnly(True)
        pred_layout.addWidget(self.pred_file_path_edit)

        self.import_predictions_btn = QPushButton("Browse")
        self.import_predictions_btn.setToolTip("Import predictions.nc file from labels\\... folder")
        pred_layout.addWidget(self.import_predictions_btn)

        self.pred_show_predictions = QCheckBox("Show predictions")
        self.pred_show_predictions.setEnabled(False)
        self.pred_show_predictions.setChecked(False)
        pred_layout.addWidget(self.pred_show_predictions)

        target_layout.addRow(pred_row)

    def _create_labels_row(self, target_layout):
        labels_row = QWidget()
        labels_layout = QHBoxLayout()
        labels_layout.setContentsMargins(0, 0, 0, 0)
        labels_row.setLayout(labels_layout)

        self.labels_format_combo = QComboBox()
        self.labels_format_combo.addItem(".nc file")
        if self.app_state.audio_folder:
            from ethograph.utils.label_intervals import CROWSETTA_SEQ_FORMATS
            for fmt in CROWSETTA_SEQ_FORMATS:
                self.labels_format_combo.addItem(fmt)
        self.labels_format_combo.setToolTip("Label file format to import")
        labels_layout.addWidget(self.labels_format_combo)

        self.label_file_path_edit = QLineEdit()
        if self.import_labels_checkbox.isChecked() and self.app_state.nc_file_path:
            self.label_file_path_edit.setText(self.app_state.nc_file_path or "")
        labels_layout.addWidget(self.label_file_path_edit)

        labels_browse_btn = QPushButton("Browse")
        labels_browse_btn.clicked.connect(self._on_labels_browse_clicked)
        labels_layout.addWidget(labels_browse_btn)

        target_layout.addRow("Labels:", labels_row)

    def _on_labels_browse_clicked(self):
        fmt = self.labels_format_combo.currentText()
        if fmt == ".nc file":
            self.on_browse_clicked("file", "labels")
            return
        self._import_crowsetta_labels(fmt)

    def _import_crowsetta_labels(self, format_name):
        filter_map = {
            "aud-seq": "Text files (*.txt)",
            "simple-seq": "CSV/Text files (*.csv *.txt)",
            "generic-seq": "CSV files (*.csv)",
            "notmat": "NotMat files (*.not.mat)",
            "textgrid": "TextGrid files (*.TextGrid)",
            "timit": "PHN files (*.phn)",
            "yarden": "Yarden annotation files (*.mat)",
        }
        file_filter = filter_map.get(format_name, "All files (*)")

        nc_parent = ""
        if self.app_state.nc_file_path:
            nc_parent = str(Path(self.app_state.nc_file_path).parent)

        result = QFileDialog.getOpenFileName(
            self,
            caption=f"Open {format_name} annotation file",
            dir=nc_parent,
            filter=file_filter,
        )
        file_path = result[0] if result and result[0] else ""
        if not file_path:
            return

        self.label_file_path_edit.setText(file_path)
        self._do_crowsetta_import(format_name, file_path)

    def _do_crowsetta_import(self, format_name, file_path):
        from ethograph.utils.label_intervals import (
            crowsetta_to_intervals,
            resolve_crowsetta_mapping,
        )

        configs_dir = eto.get_project_root() / "configs"
        mapping_path = self.mapping_file_path_edit.text()

        try:
            name_to_id, new_mapping_path, warning = resolve_crowsetta_mapping(
                file_path, format_name, mapping_path, configs_dir,
            )
        except Exception as e:
            QMessageBox.critical(self, "Mapping error", str(e))
            return

        if warning:
            QMessageBox.warning(self, "Mapping warning", warning)

        if new_mapping_path:
            self.mapping_file_path_edit.setText(new_mapping_path)
            if self.labels_widget:
                self.labels_widget._reload_mapping(new_mapping_path)

        individual = "ind0"
        ds = getattr(self.app_state, "ds", None)
        if ds is not None and "individuals" in ds.coords:
            individual = str(ds.coords["individuals"].values[0])

        try:
            intervals_df = crowsetta_to_intervals(
                file_path, format_name, name_to_id, individual,
            )
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Failed to parse {format_name} file:\n{e}")
            return

        if intervals_df.empty:
            QMessageBox.information(self, "No labels", "No non-background labels found in file.")
            return

        self.app_state.label_intervals = intervals_df

        label_dt = getattr(self.app_state, "label_dt", None)
        if label_dt is not None:
            trial = getattr(self.app_state, "trials_sel", None)
            if trial is not None:
                self.app_state.set_trial_intervals(trial, intervals_df)

        if hasattr(self, "changepoints_widget") and self.changepoints_widget:
            self.changepoints_widget._update_cp_status()
        if self.labels_widget:
            self.labels_widget._mark_changes_unsaved()
            self.labels_widget.refresh_labels_shapes_layer()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()

    # ------------------------------------------------------------------
    # Post-load behavior
    # ------------------------------------------------------------------

    def on_load_complete(self):
        """Disable load panel, enable and switch to controls panel."""
        for child in self.load_panel.findChildren(QWidget):
            child.setEnabled(False)
        self.controls_toggle.setEnabled(True)
        self._show_panel("controls")

        self._ensure_crowsetta_formats()

        canary_path = getattr(self, "_canary_labels_path", None)
        if canary_path:
            self.labels_format_combo.setCurrentText("aud-seq")
            self.label_file_path_edit.setText(canary_path)
            del self._canary_labels_path

        self._auto_populate_nwb_video_folder()
        self._auto_import_crowsetta_labels()
        self._apply_nwb_epoch_mapping()

    def _auto_populate_nwb_video_folder(self):
        """If the loaded NWB file downloaded trial clips, auto-fill the video folder field."""
        dt = getattr(self.app_state, "dt", None)
        if dt is None:
            return
        video_folder = dt.attrs.get("nwb_video_folder")
        if not video_folder:
            return
        self.video_folder_edit.setText(str(video_folder))
        self.app_state.video_folder = str(video_folder)
        print(f"[NWB] Auto-set video folder: {video_folder}")

    def _apply_nwb_epoch_mapping(self):
        """If NWB epochs were imported, write mapping file and load into labels widget."""
        dt = getattr(self.app_state, "dt", None)
        if dt is None:
            return
        epoch_mapping = dt.attrs.get("nwb_epoch_mapping")
        if not epoch_mapping or not isinstance(epoch_mapping, dict):
            return

        from ethograph.utils.label_intervals import write_mapping_file

        configs_dir = eto.get_project_root() / "configs"
        mapping_path = configs_dir / "mapping_nwb_epochs.txt"
        write_mapping_file(mapping_path, epoch_mapping)
        self.mapping_file_path_edit.setText(str(mapping_path))
        if self.labels_widget:
            self.labels_widget._reload_mapping(str(mapping_path))

        n_labels = len(epoch_mapping) - 1  # exclude background
        print(f"[NWB] Auto-created mapping with {n_labels} epoch labels: {mapping_path}")

    def _ensure_crowsetta_formats(self):
        """Add crowsetta formats to labels combo if not already present."""
        from ethograph.utils.label_intervals import CROWSETTA_SEQ_FORMATS

        existing = [self.labels_format_combo.itemText(i)
                     for i in range(self.labels_format_combo.count())]
        for fmt in CROWSETTA_SEQ_FORMATS:
            if fmt not in existing:
                self.labels_format_combo.addItem(fmt)

    def _auto_import_crowsetta_labels(self):
        """If a crowsetta format and path are set, auto-import after load."""
        fmt = self.labels_format_combo.currentText()
        file_path = self.label_file_path_edit.text().strip()
        if fmt == ".nc file" or not file_path or not Path(file_path).exists():
            return
        self._do_crowsetta_import(fmt, file_path)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _on_reset_gui_clicked(self):
        self.downsample_checkbox.setChecked(False)
        self.import_labels_checkbox.setChecked(False)
        self.app_state.delete_yaml()

        for var in AppStateSpec.VARS:
            default = AppStateSpec.get_default(var)
            setattr(self.app_state, var, default)

        for attr in list(dir(self.app_state)):
            if attr.endswith("_sel"):
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
        self._clear_all_line_edits()
        dialog = NCWizardDialog(self.app_state, self, self)
        dialog.exec_()

    def _on_select_template_clicked(self):
        self._clear_all_line_edits()
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
            if t.get("labels_file"):
                self._canary_labels_path = t["labels_file"]



    def _on_load_clicked(self):
        from .dialog_busy_progress import BusyProgressDialog

        dialog = BusyProgressDialog("Loading data...", parent=self)

        def _update(msg: str) -> None:
            dialog.setLabelText(msg)
            dialog.pump_events()

        self.app_state._progress_callback = _update
        dialog.execute_blocking(self.data_widget.on_load_clicked)

    def _clear_all_line_edits(self):
        for attr in ('nc_file_path_edit', 'video_folder_edit', 'audio_folder_edit',
                  'pose_folder_edit', 'ephys_path_edit', 'kilosort_folder_edit', 
                      'label_file_path_edit', 'pred_file_path_edit'):
            widget = getattr(self, attr, None)
            if widget:
                widget.clear()
            state_key = attr.removesuffix("_edit")
            if hasattr(self.app_state, state_key):
                setattr(self.app_state, state_key, None)
        self.downsample_checkbox.setChecked(False)

    def _clear_combo_boxes(self):
        for combo in self.combos.values():
            combo.clear()
            combo.addItems(["None"])
            combo.setCurrentText("None")

    def _create_path_widget(self, target_layout, label, object_name, browse_callback):
        line_edit = QLineEdit()
        line_edit.setObjectName(f"{object_name}_edit")
        if object_name == "nc_file_path":
            line_edit.setPlaceholderText(
                "Path to .nc/.nwb file"
            )

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
            

        if object_name == "video_folder":
            self.remote_video_checkbox = QCheckBox("Remote")
            self.remote_video_checkbox.setObjectName("remote_video_checkbox")
            self.remote_video_checkbox.setToolTip(
                "Video URLs are stored in the dataset (e.g. DANDI). No local folder needed."
            )
            self.remote_video_checkbox.stateChanged.connect(
                lambda state: self._on_remote_video_toggled(state == 2, line_edit, browse_button)
            )

        clear_button = QPushButton("Clear")
        clear_button.setObjectName(f"{object_name}_clear_button")
        clear_button.clicked.connect(lambda: self._on_clear_path_clicked(object_name, line_edit))

        row_layout = QHBoxLayout()
        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        if object_name == "nc_file_path":
            row_layout.addWidget(self.import_labels_checkbox)
        if object_name == "video_folder":
            row_layout.addWidget(self.remote_video_checkbox)
        row_layout.addWidget(clear_button)
        target_layout.addRow(label, row_layout)

        return line_edit

    def _on_remote_video_toggled(self, checked, line_edit, browse_button):
        self.app_state.remote_video = checked
        line_edit.setEnabled(not checked)
        browse_button.setEnabled(not checked)


    def _on_clear_path_clicked(self, object_name, line_edit):
        line_edit.setText("")
        attr_map = {
            "nc_file_path": "nc_file_path",
            "video_folder": "video_folder",
            "audio_folder": "audio_folder",
            "pose_folder": "pose_folder",
            "ephys_path": "ephys_path",
            "kilosort_folder": "kilosort_folder",
        }
        attr = attr_map.get(object_name)
        if attr:
            setattr(self.app_state, attr, None)
        if self.labels_widget:
            self.labels_widget._update_human_verified_status()
    # Device controls (populated after load)
    # ------------------------------------------------------------------

    def create_device_controls(self, type_vars_dict):
        self._create_labels_row(self._controls_layout)
        self.controls.append(self.label_file_path_edit)

    def _expand_ephys_with_streams(self, ephys_path, ds):
        """Discover Neo streams from the ephys file for the Neo-Viewer."""
        from .plots_ephystrace import GenericEphysLoader

        self.app_state.ephys_source_map.clear()
        feature_names = []

        if not ephys_path:
            return feature_names

        filepath = os.path.normpath(str(ephys_path))

        try:
            loader = GenericEphysLoader(filepath, stream_id="0")
            streams = loader.streams

            if streams and len(streams) > 1:
                for sid, info in streams.items():
                    display_name = info["name"]
                    self.app_state.ephys_source_map[display_name] = (filepath, sid, 0)
                    feature_names.append(display_name)
            else:
                display_name = "Ephys Waveform"
                self.app_state.ephys_source_map[display_name] = (filepath, "0", 0)
                feature_names.append(display_name)
        except (OSError, IOError, ValueError) as e:
            print(f"Skipping ephys file {Path(filepath).name}: {e}")

        return feature_names

    def _create_combo_widget(self, key, vars):
        combo = QComboBox()
        combo.setObjectName(f"{key}_combo")
        combo.currentTextChanged.connect(self._on_combo_changed)
        combo.addItems([str(var) for var in vars])

        self._controls_layout.addRow(f"{key.capitalize()}:", combo)
        self.combos[key] = combo
        self.controls.append(combo)
        return combo

    def _on_combo_changed(self):
        if hasattr(self.data_widget, '_on_combo_changed'):
            self.data_widget._on_combo_changed()

    def set_controls_enabled(self, enabled):
        for control in self.controls:
            control.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Load button + downsample
    # ------------------------------------------------------------------

    def _create_load_button(self, target_layout):
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
        self.load_button.clicked.connect(self._on_load_clicked)

        load_layout.addWidget(self.downsample_checkbox)
        load_layout.addWidget(self.downsample_spin)
        load_layout.addWidget(self.load_button, stretch=1)

        target_layout.addRow(load_layout)

    def _on_downsample_toggled(self, checked):
        self.downsample_spin.setEnabled(checked)
        self.app_state.downsample_enabled = checked

    def _on_downsample_value_changed(self, value):
        self.app_state.downsample_factor = value

    def disable_downsample_controls(self):
        self.downsample_checkbox.setEnabled(False)
        self.downsample_spin.setEnabled(False)

    def get_downsample_factor(self):
        if self.downsample_checkbox.isChecked():
            return self.downsample_spin.value()
        return None

    # ------------------------------------------------------------------
    # Browse dialogs
    # ------------------------------------------------------------------

    def on_browse_clicked(self, browse_type="file", media_type=None):
        if browse_type == "file":
            if media_type == "data":
                result = QFileDialog.getOpenFileName(
                    None,
                    caption="Open file containing feature data",
                    filter="Data files (*.nc *.nwb)",
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
                    label_dt_full = eto.open(labels_file_path)
                    self.app_state.label_dt = label_dt_full.get_label_dt()
                    self.app_state.label_ds = self.app_state.label_dt.trial(self.app_state.trials_sel)
                    self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)

                    self.label_file_path_edit.setText(labels_file_path)

                    if hasattr(self, "changepoints_widget") and self.changepoints_widget:
                        self.changepoints_widget._update_cp_status()
                    if self.labels_widget:
                        self.labels_widget._mark_changes_unsaved()
                        self.labels_widget.refresh_labels_shapes_layer()
                        self.labels_widget._update_human_verified_status()
                    if self.data_widget and self.data_widget.plot_container:
                        self.data_widget.plot_container.labels_redraw_needed.emit()

            elif media_type == "ephys":
                result = QFileDialog.getOpenFileName(
                    None,
                    caption="Open ephys recording file",
                    filter=EPHYS_FILE_FILTER,
                )
                ephys_path = result[0] if result and len(result) >= 1 else ""
                if not ephys_path:
                    return

                self.ephys_path_edit.setText(ephys_path)
                self.app_state.ephys_path = ephys_path
                if self.app_state.dt is not None:
                    self.app_state.dt.set_media_files(ephys=ephys_path)
                self._auto_detect_kilosort(ephys_path)

        elif browse_type == "folder":
            if media_type == "video":
                caption = "Open folder with video files (e.g. mp4, mov)."
            elif media_type == "audio":
                caption = "Open folder with audio files (e.g. wav, mp3, mp4)."
            elif media_type == "pose":
                caption = "Open folder with pose files (e.g. .csv, .h5)."
            elif media_type == "kilosort":
                caption = "Select Kilosort output folder."

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
            elif media_type == "kilosort":
                if folder_path:
                    self.kilosort_folder_edit.setText(folder_path)
                    self.app_state.kilosort_folder = folder_path

    def _auto_detect_kilosort(self, ephys_path: str):
        ephys_parent = Path(ephys_path).parent
        for folder_name in ("kilosort4", "kilosort"):
            ks_folder = ephys_parent / folder_name
            if ks_folder.is_dir():
                self.kilosort_folder_edit.setText(str(ks_folder))
                self.app_state.kilosort_folder = str(ks_folder)
                return
        self.kilosort_folder_edit.clear()
        self.app_state.kilosort_folder = None

    def get_nc_file_path(self):
        return self.nc_file_path_edit.text().strip()

    # ------------------------------------------------------------------
    # Wire signals to other widgets (called from MetaWidget)
    # ------------------------------------------------------------------

    def wire_label_signals(self):
        """Connect mapping/predictions UI to LabelsWidget methods."""
        self.mapping_file_path_edit.returnPressed.connect(
            lambda: self.labels_widget._reload_mapping(self.mapping_file_path_edit.text())
        )
        self.browse_mapping_btn.clicked.connect(self.labels_widget._browse_mapping_file)
        self.temp_labels_button.clicked.connect(self.labels_widget._create_temporary_labels)
        self.import_predictions_btn.clicked.connect(self.labels_widget._import_predictions_from_file)
        self.pred_show_predictions.stateChanged.connect(self.labels_widget._on_pred_show_predictions_changed)

    def wire_ephys_signals(self, ephys_widget):
        """Connect kilosort UI to EphysWidget methods."""
        self.kilosort_folder_edit.returnPressed.connect(ephys_widget._load_kilosort_folder)
