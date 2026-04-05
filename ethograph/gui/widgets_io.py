"""Widget for input/output controls and data loading."""

import logging
import os
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ethograph.labels.export import correct_offsets_trial

import ethograph as eto
from ethograph.utils.paths import default_config_dir, find_config, find_mapping_file
from ethograph.io.validation import EPHYS_FILE_FILTER
from ethograph.io.metadata_table import metadata_tsv_path
from ethograph.labels.tsv_store import labels_tsv_path, load_labels_tsv

from .app_state import AppStateSpec
from .notify import notify_dialog
from .wizard_overview import NCWizardDialog
from .dialog_select_template import TemplateDialog

logger = logging.getLogger(__name__)


def _populate_if_exists(line_edit: QLineEdit, path: str | Path | None) -> None:
    """Set a QLineEdit's text only if *path* points to an existing file or folder."""
    if path is None:
        return
    p = Path(path)
    if p.exists():
        line_edit.setText(str(p))


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
        self._create_export_panel(main_layout)
        self._wire_app_state_path_signals()
        self._wire_path_edit_signals()

        # Initial state: load tab active, others greyed out
        self.controls_toggle.setEnabled(False)
        self.export_toggle.setEnabled(False)
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
        if self.app_state.neurons_path:
            self.neurons_path_edit.setText(self.app_state.neurons_path)

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
        self.app_state.neurons_path_changed.connect(
            lambda value: self.neurons_path_edit.setText(value or "")
        )
        self.app_state.nwb_file_path_changed.connect(
            lambda value: self.nwb_file_path_edit.setText(value or "")
        )


    def _wire_path_edit_signals(self):
        self.nc_file_path_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.nc_file_path_edit, "nc_file_path")
        )
        self.nwb_file_path_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.nwb_file_path_edit, "nwb_file_path")
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
        self.neurons_path_edit.editingFinished.connect(
            lambda: self._sync_line_edit_to_state(self.neurons_path_edit, "neurons_path")
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

        self.controls_toggle = QPushButton("Import labels")
        self.controls_toggle.setCheckable(True)
        self.controls_toggle.clicked.connect(self._toggle_controls)
        toggle_layout.addWidget(self.controls_toggle)

        self.export_toggle = QPushButton("Export labels")
        self.export_toggle.setCheckable(True)
        self.export_toggle.clicked.connect(self._toggle_export)
        toggle_layout.addWidget(self.export_toggle)

        main_layout.addWidget(toggle_widget)

    def _show_panel(self, panel_name):
        panels = {
            "load": (self.load_panel, self.load_toggle),
            "controls": (self.controls_panel, self.controls_toggle),
            "export": (self.export_panel, self.export_toggle),
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

    def _toggle_export(self):
        self._show_panel("export" if self.export_toggle.isChecked() else "controls")

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
        self.nwb_file_path_edit = self._create_path_widget(
            self._load_layout,
            label="Alignment:",
            object_name="nwb_file_path",
            browse_callback=lambda: self._browse_nwb_file(),
        )
        self.metadata_path_edit = self._create_path_widget(
            self._load_layout,
            label="Metadata:",
            object_name="metadata_path",
            browse_callback=lambda: self._browse_metadata_file(),
        )
        self.video_folder_edit = self._create_path_widget(
            self._load_layout,
            label="Video folder:",
            object_name="video_folder",
            browse_callback=lambda: self.on_browse_clicked("folder", "video"),
        )
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

        self.neurons_path_edit = self._create_path_widget(
            self._load_layout,
            label="Units (Kilosort/Pynapple):",
            object_name="neurons_path",
            browse_callback=self._browse_neurons,
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

        labels_group = QGroupBox("Labels")
        self._labels_group_layout = QFormLayout()
        self._labels_group_layout.setSpacing(2)
        self._labels_group_layout.setContentsMargins(4, 4, 4, 4)
        labels_group.setLayout(self._labels_group_layout)

        self._create_mapping_row(self._labels_group_layout)
        # Labels row inserted here dynamically by create_device_controls()
        self._labels_row_index = self._labels_group_layout.rowCount()

        self._controls_layout.addRow(labels_group)
        self._create_predictions_row(self._controls_layout)

        main_layout.addWidget(self.controls_panel)

    # ------------------------------------------------------------------
    # Export panel
    # ------------------------------------------------------------------

    def _create_export_panel(self, main_layout):
        self.export_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.export_panel.setLayout(layout)

        # Human verification row
        hv_row = QHBoxLayout()
        hv_row.addWidget(QLabel("Apply human verification to:"))
        self.human_verify_trial_btn = QPushButton("Single Trial")
        self.human_verify_trial_btn.clicked.connect(lambda: self._human_verification_true("single_trial"))
        hv_row.addWidget(self.human_verify_trial_btn)
        self.human_verify_all_trials_btn = QPushButton("All Trials")
        self.human_verify_all_trials_btn.clicked.connect(lambda: self._human_verification_true("all_trials"))
        hv_row.addWidget(self.human_verify_all_trials_btn)
        hv_row.addStretch()
        layout.addLayout(hv_row)

        # Correct offsets row
        co_row = QHBoxLayout()
        co_row.addWidget(QLabel("Apply offset correction to:"))
        self.correct_offsets_trial_btn = QPushButton("Single Trial")
        self.correct_offsets_trial_btn.clicked.connect(lambda: self._apply_correct_offsets("single_trial"))
        co_row.addWidget(self.correct_offsets_trial_btn)
        self.correct_offsets_all_trials_btn = QPushButton("All Trials")
        self.correct_offsets_all_trials_btn.clicked.connect(lambda: self._apply_correct_offsets("all_trials"))
        co_row.addWidget(self.correct_offsets_all_trials_btn)
        co_row.addStretch()
        layout.addLayout(co_row)

        # Purge small labels row
        purge_row = QHBoxLayout()
        purge_row.addWidget(QLabel("Purge labels shorter than:"))
        self.purge_min_duration_spin = QDoubleSpinBox()
        self.purge_min_duration_spin.setRange(0.001, 10.0)
        self.purge_min_duration_spin.setValue(0.01)
        self.purge_min_duration_spin.setSuffix(" s")
        self.purge_min_duration_spin.setSingleStep(0.01)
        self.purge_min_duration_spin.setDecimals(3)
        purge_row.addWidget(self.purge_min_duration_spin)
        self.purge_trial_btn = QPushButton("Single Trial")
        self.purge_trial_btn.clicked.connect(lambda: self._apply_purge_small_labels("single_trial"))
        purge_row.addWidget(self.purge_trial_btn)
        self.purge_all_trials_btn = QPushButton("All Trials")
        self.purge_all_trials_btn.clicked.connect(lambda: self._apply_purge_small_labels("all_trials"))
        purge_row.addWidget(self.purge_all_trials_btn)
        purge_row.addStretch()
        layout.addLayout(purge_row)

        # Save button
        self.save_labels_button = QPushButton("Save labels (Ctrl+S)")
        self.save_labels_button.setToolTip("Save labels TSV + local backup + optional remote backup")
        self.save_labels_button.clicked.connect(self._save_labels)
        layout.addWidget(self.save_labels_button)

        # Local backup (read-only display)
        local_backup_row = QHBoxLayout()
        local_backup_label = QLabel("Local backup:")
        local_backup_label.setFixedWidth(90)
        local_backup_row.addWidget(local_backup_label)
        self.local_backup_edit = QLineEdit()
        self.local_backup_edit.setReadOnly(True)
        self.local_backup_edit.setPlaceholderText("label_backups/")
        if self.app_state.nc_file_path:
            backup_dir = str(Path(self.app_state.nc_file_path).parent / "label_backups")
            self.local_backup_edit.setText(backup_dir)
        local_backup_row.addWidget(self.local_backup_edit)
        layout.addLayout(local_backup_row)

        # Remote backup (optional)
        remote_group = QGroupBox("Remote backup")
        remote_group_layout = QVBoxLayout()
        remote_group_layout.setSpacing(4)
        remote_group.setLayout(remote_group_layout)

        remote_path_row = QHBoxLayout()
        self.remote_backup_edit = QLineEdit()
        self.remote_backup_edit.setPlaceholderText("(optional) cloud/git folder...")
        self.remote_backup_edit.setToolTip(
            "Optional path to a remote folder for label backups.\n"
            "Useful for syncing labels via cloud storage or a git repository."
        )
        if self.app_state.remote_backup_path:
            self.remote_backup_edit.setText(self.app_state.remote_backup_path)
        self.remote_backup_edit.editingFinished.connect(
            lambda: setattr(self.app_state, "remote_backup_path", self.remote_backup_edit.text().strip() or None)
        )
        remote_path_row.addWidget(self.remote_backup_edit)
        remote_browse_btn = QPushButton("Browse file")
        remote_browse_btn.clicked.connect(self._browse_remote_backup)
        remote_path_row.addWidget(remote_browse_btn)
        remote_group_layout.addLayout(remote_path_row)

        remote_options_row = QHBoxLayout()
        self.remote_save_mode_combo = QComboBox()
        self.remote_save_mode_combo.addItem("Save with timestamp")
        self.remote_save_mode_combo.addItem("Overwrite file")
        self.remote_save_mode_combo.addItem("Overwrite + git commit")
        self.remote_save_mode_combo.setToolTip(
            "Timestamp: each save creates a new file (safe, auditable).\n"
            "Overwrite: saves a single file, no version control.\n"
            "Overwrite + git commit: saves a single file and auto-commits.\n"
            "  Requires the remote folder to be a git repo (run 'git init' once)."
        )
        mode_map = {"timestamp": 0, "overwrite": 1, "git": 2}
        self.remote_save_mode_combo.setCurrentIndex(mode_map.get(self.app_state.remote_backup_mode, 0))

        def _on_mode_changed(text):
            if text == "Overwrite + git commit":
                self.app_state.remote_backup_mode = "git"
            elif text == "Overwrite file":
                self.app_state.remote_backup_mode = "overwrite"
            else:
                self.app_state.remote_backup_mode = "timestamp"
        self.remote_save_mode_combo.currentTextChanged.connect(_on_mode_changed)
        remote_options_row.addWidget(self.remote_save_mode_combo)

        self.remote_depth_combo = QComboBox()
        self.remote_depth_combo.setToolTip(
            "Controls the subfolder structure inside the remote backup root.\n"
            "Flat: all files land directly in the remote root.\n"
            "Higher levels mirror parent directories to avoid filename collisions."
        )
        self._populate_remote_depth_combo()
        self.remote_depth_combo.currentIndexChanged.connect(
            lambda idx: setattr(self.app_state, "remote_path_depth", idx)
        )
        remote_options_row.addWidget(self.remote_depth_combo)
        remote_group_layout.addLayout(remote_options_row)

        layout.addWidget(remote_group)

        self.app_state.nc_file_path_changed.connect(self._populate_remote_depth_combo)

        main_layout.addWidget(self.export_panel)

    # ------------------------------------------------------------------
    # Export panel handlers
    # ------------------------------------------------------------------

    def _save_labels(self):
        remote_path = self.remote_backup_edit.text().strip() or None
        remote_mode = self.app_state.remote_backup_mode
        try:
            self.app_state.save_labels(remote_path=remote_path, remote_mode=remote_mode)
        except Exception as e:
            notify_dialog(str(e), "error", "Save Error", self)

    def _populate_remote_depth_combo(self):
        if not hasattr(self, "remote_depth_combo"):
            return
        nc_path = self.app_state.nc_file_path
        combo = self.remote_depth_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Flat (no subfolders)")
        if nc_path:
            parts = Path(nc_path).parent.parts[1:]
            for i, _ in enumerate(parts):
                subfolder = "/".join(parts[len(parts) - i - 1:])
                combo.addItem(subfolder)
        saved_depth = self.app_state.remote_path_depth
        combo.setCurrentIndex(min(saved_depth, combo.count() - 1))
        combo.blockSignals(False)

    def _browse_remote_backup(self):
        folder = QFileDialog.getExistingDirectory(self, "Select remote backup folder")
        if folder:
            self.remote_backup_edit.setText(folder)

    def _human_verification_true(self, mode=None):
        if self.app_state.trials_sel is None:
            return
        if mode == "single_trial":
            self.app_state.set_trial_meta_attr(self.app_state.trials_sel, 'human_verified', 1)
        elif mode == "all_trials":
            for trial in self.app_state.trials:
                self.app_state.set_trial_meta_attr(trial, 'human_verified', 1)

        self._update_human_verified_status()
        self.app_state.changes_saved = False
        if self.data_widget:
            self.data_widget.update_trials_combo()
        if hasattr(self, "meta_widget") and self.meta_widget:
            self.meta_widget.update_labels_widget_title()

    def _update_human_verified_status(self):
        if not hasattr(self, "human_verify_trial_btn"):
            return
        default_style = ""
        verified_style = "background-color: green; color: white;"

        if self.app_state.trials_sel is None:
            self.human_verify_trial_btn.setStyleSheet(default_style)
            self.human_verify_all_trials_btn.setStyleSheet(default_style)
            return

        trial_meta = self.app_state.get_trial_meta(self.app_state.trials_sel)
        if trial_meta.get('human_verified', 0):
            self.human_verify_trial_btn.setStyleSheet(verified_style)
        else:
            self.human_verify_trial_btn.setStyleSheet(default_style)

        all_verified = all(
            self.app_state.get_trial_meta(t).get('human_verified', 0)
            for t in self.app_state.trials
        )
        if all_verified and self.app_state.trials:
            self.human_verify_all_trials_btn.setStyleSheet(verified_style)
        else:
            self.human_verify_all_trials_btn.setStyleSheet(default_style)

    def _apply_correct_offsets(self, mode: str):
        if self.app_state.trials_sel is None:
            return

        if mode == "single_trial":
            trial = self.app_state.trials_sel
            df = correct_offsets_trial(self.app_state.get_trial_intervals(trial))
            self.app_state.set_trial_intervals(trial, df)
            self.app_state.label_intervals = df
            self.app_state.set_trial_meta_attr(trial, "offsets_corrected", 1)
        elif mode == "all_trials":
            for trial in self.app_state.trials:
                df = correct_offsets_trial(self.app_state.get_trial_intervals(trial))
                self.app_state.set_trial_intervals(trial, df)
                self.app_state.set_trial_meta_attr(trial, "offsets_corrected", 1)
            self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)

        self._update_correct_offsets_status()
        self.app_state.changes_saved = False
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()

    def _update_correct_offsets_status(self):
        if not hasattr(self, "correct_offsets_trial_btn"):
            return
        default_style = ""
        applied_style = "background-color: green; color: white;"

        if self.app_state.trials_sel is None:
            self.correct_offsets_trial_btn.setStyleSheet(default_style)
            self.correct_offsets_all_trials_btn.setStyleSheet(default_style)
            return

        trial_corrected = self.app_state.get_trial_meta(self.app_state.trials_sel).get("offsets_corrected", 0)
        self.correct_offsets_trial_btn.setStyleSheet(applied_style if trial_corrected else default_style)

        all_corrected = all(
            self.app_state.get_trial_meta(t).get("offsets_corrected", 0)
            for t in self.app_state.trials
        )
        self.correct_offsets_all_trials_btn.setStyleSheet(applied_style if all_corrected else default_style)

    def _apply_purge_small_labels(self, mode: str):
        if self.app_state.trials_sel is None:
            return

        min_duration = self.purge_min_duration_spin.value()

        def purge(df):
            if df.empty:
                return df
            mask = (df["offset_s"] - df["onset_s"]) >= min_duration
            return df[mask].copy().reset_index(drop=True)

        if mode == "single_trial":
            trial = self.app_state.trials_sel
            df = purge(self.app_state.get_trial_intervals(trial))
            self.app_state.set_trial_intervals(trial, df)
            self.app_state.label_intervals = df
            self.app_state.set_trial_meta_attr(trial, "small_labels_purged", 1)
        elif mode == "all_trials":
            for trial in self.app_state.trials:
                df = purge(self.app_state.get_trial_intervals(trial))
                self.app_state.set_trial_intervals(trial, df)
                self.app_state.set_trial_meta_attr(trial, "small_labels_purged", 1)
            self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)

        self._update_purge_small_labels_status()
        self.app_state.changes_saved = False
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()

    def _update_purge_small_labels_status(self):
        if not hasattr(self, "purge_trial_btn"):
            return
        default_style = ""
        applied_style = "background-color: green; color: white;"

        if self.app_state.trials_sel is None:
            self.purge_trial_btn.setStyleSheet(default_style)
            self.purge_all_trials_btn.setStyleSheet(default_style)
            return

        trial_purged = self.app_state.get_trial_meta(self.app_state.trials_sel).get("small_labels_purged", 0)
        self.purge_trial_btn.setStyleSheet(applied_style if trial_purged else default_style)

        all_purged = all(
            self.app_state.get_trial_meta(t).get("small_labels_purged", 0)
            for t in self.app_state.trials
        )
        self.purge_all_trials_btn.setStyleSheet(applied_style if all_purged else default_style)

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

        target_layout.addRow("Name mapping:", mapping_row)

        self.temp_labels_button = QPushButton("Create temporary labels")
        self.temp_labels_button.setToolTip("Create custom labels for this session only")
        target_layout.addRow("", self.temp_labels_button)

    def _create_predictions_row(self, target_layout):
        pred_group = QGroupBox("Predictions")
        pred_group_layout = QVBoxLayout()
        pred_group_layout.setContentsMargins(4, 4, 4, 4)
        pred_group_layout.setSpacing(2)
        pred_group.setLayout(pred_group_layout)

        # Row 1: folder path + browse
        folder_row = QHBoxLayout()
        folder_row.setContentsMargins(0, 0, 0, 0)
        self.pred_file_path_edit = QLineEdit()
        self.pred_file_path_edit.setReadOnly(True)
        self.pred_file_path_edit.setPlaceholderText("No predictions folder selected")
        folder_row.addWidget(self.pred_file_path_edit)
        self.import_predictions_btn = QPushButton("Browse")
        self.import_predictions_btn.setToolTip("Select predictions folder (corr/ or uncorr/ subfolder)")
        folder_row.addWidget(self.import_predictions_btn)
        pred_group_layout.addLayout(folder_row)

        # Row 2: show checkbox + threshold + PDF button
        controls_row = QHBoxLayout()
        controls_row.setContentsMargins(0, 0, 0, 0)

        controls_row.addWidget(QLabel("Frame thr:"))
        self.pred_confidence_threshold_spin = QDoubleSpinBox()
        self.pred_confidence_threshold_spin.setRange(0.0, 1.0)
        self.pred_confidence_threshold_spin.setSingleStep(0.05)
        self.pred_confidence_threshold_spin.setDecimals(2)
        self.pred_confidence_threshold_spin.setValue(0.75)
        self.pred_confidence_threshold_spin.setToolTip(
            "Frame-level confidence threshold — frames below this are marked red."
        )
        controls_row.addWidget(self.pred_confidence_threshold_spin)

        controls_row.addWidget(QLabel("Segment thr:"))
        self.pred_segment_confidence_threshold_spin = QDoubleSpinBox()
        self.pred_segment_confidence_threshold_spin.setRange(0.0, 1.0)
        self.pred_segment_confidence_threshold_spin.setSingleStep(0.05)
        self.pred_segment_confidence_threshold_spin.setDecimals(2)
        self.pred_segment_confidence_threshold_spin.setValue(0.6)
        self.pred_segment_confidence_threshold_spin.setToolTip(
            "Segment-level mean confidence threshold — segments below this are highlighted red."
        )
        controls_row.addWidget(self.pred_segment_confidence_threshold_spin)

        self.pred_confidence_pdf_btn = QPushButton("Update confidence (+ PDF)")
        self.pred_confidence_pdf_btn.setToolTip(
            "Regenerate confidence PDF with current thresholds and update low/high confidence classification."
        )
        self.pred_confidence_pdf_btn.setEnabled(False)
        controls_row.addWidget(self.pred_confidence_pdf_btn)

        pred_group_layout.addLayout(controls_row)
        target_layout.addRow(pred_group)

    def _create_labels_row_at_index(self):
        """Create two labels rows: input (browse) and output (auto-generated TSV)."""
        # Row 1: format combo + input path + browse
        input_row = QWidget()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_row.setLayout(input_layout)

        self.labels_format_combo = QComboBox()
        self.labels_format_combo.addItem(".tsv")
        self.labels_format_combo.addItem(".nc (legacy)")
        self.labels_format_combo.addItem("pynapple (.npz)")
        self.labels_format_combo.addItem("pynapple (.nwb)")
        if self.app_state.audio_folder:
            from ethograph.labels.converters import CROWSETTA_SEQ_FORMATS
            for fmt in CROWSETTA_SEQ_FORMATS:
                self.labels_format_combo.addItem(fmt)
        self.labels_format_combo.setToolTip("Label file format to import")
        self.labels_format_combo.currentTextChanged.connect(self._on_labels_format_changed)
        input_layout.addWidget(self.labels_format_combo)

        self.label_file_path_edit = QLineEdit()
        if self.import_labels_checkbox.isChecked() and self.app_state.nc_file_path:
            _populate_if_exists(self.label_file_path_edit, labels_tsv_path(self.app_state.nc_file_path))
        input_layout.addWidget(self.label_file_path_edit)

        self.labels_browse_btn = QPushButton("Browse")
        self.labels_browse_btn.clicked.connect(self._on_labels_browse_clicked)
        input_layout.addWidget(self.labels_browse_btn)

        self._labels_input_label = QLabel("Labels path:")
        self._labels_group_layout.insertRow(self._labels_row_index, self._labels_input_label, input_row)

        # Row 2: output TSV path (read-only, no browse, greyed out for .tsv)
        self.labels_output_edit = QLineEdit()
        self.labels_output_edit.setReadOnly(True)
        self.labels_output_edit.setPlaceholderText("Converted .tsv output will appear here")
        self._labels_output_label = QLabel("Labels output:")
        self._labels_group_layout.insertRow(self._labels_row_index + 1, self._labels_output_label, self.labels_output_edit)

        # Initial state: .tsv selected → output row disabled
        self._set_labels_output_enabled(False)

    def _set_labels_output_enabled(self, enabled: bool):
        self.labels_output_edit.setEnabled(enabled)
        self._labels_output_label.setEnabled(enabled)

    def _on_labels_format_changed(self, fmt: str):
        is_tsv = fmt == ".tsv"
        self._set_labels_output_enabled(not is_tsv)
        if is_tsv:
            self._labels_input_label.setText("Labels path:")
            self.labels_output_edit.clear()
        else:
            self._labels_input_label.setText("Labels input:")

    def _on_labels_browse_clicked(self):
        fmt = self.labels_format_combo.currentText()
        if fmt == ".tsv":
            self._import_tsv_labels()
            return
        if fmt == ".nc (legacy)":
            self.on_browse_clicked("file", "labels")
            return
        if fmt.startswith("pynapple"):
            self._import_pynapple_labels()
            return
        self._import_crowsetta_labels(fmt)

    def _import_tsv_labels(self):
        nc_parent = ""
        if self.app_state.nc_file_path:
            nc_parent = str(Path(self.app_state.nc_file_path).parent)

        result = QFileDialog.getOpenFileName(
            self,
            caption="Open labels TSV file",
            dir=nc_parent,
            filter="TSV files (*.tsv)",
        )
        file_path = result[0] if result and result[0] else ""
        if not file_path:
            return

        self.app_state._all_labels_df = load_labels_tsv(file_path)
        self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)
        self.label_file_path_edit.setText(file_path)

        if hasattr(self, "changepoints_widget") and self.changepoints_widget:
            self.changepoints_widget._update_cp_status()
        if self.labels_widget:
            self.labels_widget._mark_changes_unsaved()
            self.labels_widget.refresh_labels_shapes_layer()
        self._update_human_verified_status()
        self._update_correct_offsets_status()
        self._update_purge_small_labels_status()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()

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
        from ethograph.labels.converters import (
            crowsetta_to_intervals,
            resolve_crowsetta_mapping,
        )

        data_dir = Path(self.app_state.nc_file_path).parent if self.app_state.nc_file_path else None
        configs_dir = default_config_dir(data_dir)
        mapping_path = self.mapping_file_path_edit.text()

        try:
            name_to_id, new_mapping_path, warning = resolve_crowsetta_mapping(
                file_path, format_name, mapping_path, configs_dir,
            )
        except (OSError, ValueError, KeyError) as e:
            logger.exception("Crowsetta mapping resolution failed")
            notify_dialog(str(e), "error", "Mapping error", self)
            return

        if warning:
            notify_dialog(warning, "warning", "Mapping warning", self)

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
        except (OSError, ValueError, KeyError) as e:
            logger.exception("Failed to parse %s file", format_name)
            notify_dialog(f"Failed to parse {format_name} file:\n{e}", "error", "Import error", self)
            return

        if intervals_df.empty:
            notify_dialog("No non-background labels found in file.", "info", "No labels", self)
            return

        self._apply_imported_intervals(intervals_df)

    def _import_pynapple_labels(self):
        """Import labels from a pynapple .npz or .nwb file."""
        import pynapple as nap

        fmt = self.labels_format_combo.currentText()
        ext_filter = {
            "pynapple (.npz)": "NPZ files (*.npz);;All files (*)",
            "pynapple (.nwb)": "NWB files (*.nwb);;All files (*)",
        }.get(fmt, "All files (*)")

        nc_parent = ""
        if self.app_state.nc_file_path:
            nc_parent = str(Path(self.app_state.nc_file_path).parent)

        result = QFileDialog.getOpenFileName(
            self, caption=f"Open {fmt} file for labels", dir=nc_parent, filter=ext_filter,
        )
        file_path = result[0] if result and result[0] else ""
        if not file_path:
            return

        self.label_file_path_edit.setText(file_path)

        try:
            data = nap.load_file(file_path)
        except Exception as e:
            logger.exception("Failed to load pynapple file")
            notify_dialog(f"Failed to load pynapple file:\n{e}", "error", "Import error", self)
            return

        intervalsets = {}
        for key in data.keys():
            try:
                val = data[key]
            except Exception:
                continue
            if isinstance(val, nap.IntervalSet) and key.lower() != "trials":
                intervalsets[key] = val

        if not intervalsets:
            notify_dialog("No IntervalSets found (excluding 'trials').", "info", "No labels", self)
            return

        from ethograph.labels.converters import build_mapping_from_labels, write_mapping_file
        from ethograph.labels.intervals import _rows_to_df

        label_names = sorted(intervalsets.keys())
        name_to_id = build_mapping_from_labels(label_names)

        data_dir = Path(self.app_state.nc_file_path).parent if self.app_state.nc_file_path else None
        configs_dir = default_config_dir(data_dir)
        mapping_path = configs_dir / "mapping_pynapple.txt"
        write_mapping_file(mapping_path, name_to_id)
        self.mapping_file_path_edit.setText(str(mapping_path))
        if self.labels_widget:
            self.labels_widget._reload_mapping(str(mapping_path))

        individual = "ind0"
        ds = getattr(self.app_state, "ds", None)
        if ds is not None and "individuals" in ds.coords:
            individual = str(ds.coords["individuals"].values[0])

        rows: list[dict] = []
        for name, iset in intervalsets.items():
            label_id = name_to_id.get(name, 0)
            if label_id == 0:
                continue
            starts = np.asarray(iset.start)
            ends = np.asarray(iset.end)
            for s, e in zip(starts, ends):
                rows.append({
                    "onset_s": float(s),
                    "offset_s": float(e),
                    "labels": label_id,
                    "individual": individual,
                })

        intervals_df = _rows_to_df(rows)
        if intervals_df.empty:
            notify_dialog("No intervals found in IntervalSets.", "info", "No labels", self)
            return

        self._apply_imported_intervals(intervals_df)

    def _apply_imported_intervals(self, intervals_df):
        """Common post-import: save converted TSV, load into app state, refresh UI."""
        from ethograph.labels.tsv_store import save_labels_tsv, TRIAL_META_DEFAULTS

        self.app_state.label_intervals = intervals_df

        trial = getattr(self.app_state, "trials_sel", None)
        if trial is not None:
            self.app_state.set_trial_intervals(trial, intervals_df)

        # For non-.tsv formats, save the converted TSV and show in output row
        fmt = self.labels_format_combo.currentText()
        if fmt != ".tsv" and self.app_state.nc_file_path:
            tsv_out = labels_tsv_path(self.app_state.nc_file_path)
            all_df = self.app_state._all_labels_df
            if all_df is not None and not all_df.empty:
                save_labels_tsv(tsv_out, all_df)
            self.labels_output_edit.setText(str(tsv_out))

        if hasattr(self, "changepoints_widget") and self.changepoints_widget:
            self.changepoints_widget._update_cp_status()
        if self.labels_widget:
            self.labels_widget._mark_changes_unsaved()
            self.labels_widget.refresh_labels_shapes_layer()
        self._update_human_verified_status()
        self._update_correct_offsets_status()
        self._update_purge_small_labels_status()
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
        self.export_toggle.setEnabled(True)
        self._show_panel("controls")

        self._ensure_crowsetta_formats()

        canary_path = getattr(self, "_canary_labels_path", None)
        if canary_path:
            self.labels_format_combo.setCurrentText("aud-seq")
            self.label_file_path_edit.setText(canary_path)
            del self._canary_labels_path

        self._auto_populate_nwb_video_folder()
        self._auto_discover_nwb()
        self._auto_discover_metadata()
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
        video_folder = self._maybe_downsample_videos(str(video_folder))
        self.video_folder_edit.setText(video_folder)
        self.app_state.video_folder = video_folder
        logger.info("NWB auto-set video folder: %s", video_folder)

    def _apply_nwb_epoch_mapping(self):
        """If NWB epochs were imported, write mapping file and load into labels widget."""
        dt = getattr(self.app_state, "dt", None)
        if dt is None:
            return
        epoch_mapping = dt.attrs.get("nwb_epoch_mapping")
        if not epoch_mapping or not isinstance(epoch_mapping, dict):
            return

        from ethograph.labels.converters import write_mapping_file

        data_dir = Path(self.app_state.nc_file_path).parent if self.app_state.nc_file_path else None
        mapping_path = default_config_dir(data_dir) / "mapping_nwb_epochs.txt"
        write_mapping_file(mapping_path, epoch_mapping)
        self.mapping_file_path_edit.setText(str(mapping_path))
        if self.labels_widget:
            self.labels_widget._reload_mapping(str(mapping_path))

        n_labels = len(epoch_mapping) - 1  # exclude background
        logger.info("NWB auto-created mapping with %d epoch labels: %s", n_labels, mapping_path)

    def _ensure_crowsetta_formats(self):
        """Add crowsetta formats to labels combo if not already present."""
        from ethograph.labels.converters import CROWSETTA_SEQ_FORMATS

        existing = [self.labels_format_combo.itemText(i)
                     for i in range(self.labels_format_combo.count())]
        for fmt in CROWSETTA_SEQ_FORMATS:
            if fmt not in existing:
                self.labels_format_combo.addItem(fmt)

    def _auto_import_crowsetta_labels(self):
        """If a crowsetta format and path are set, auto-import after load."""
        fmt = self.labels_format_combo.currentText()
        file_path = self.label_file_path_edit.text().strip()
        if fmt in (".tsv", ".nc (legacy)") or not file_path or not Path(file_path).exists():
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

        global_settings = default_config_dir() / "gui_settings.yaml"
        self.app_state._yaml_path = str(global_settings)
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
                video_folder = self._maybe_downsample_videos(t["video_folder"])
                self.video_folder_edit.setText(video_folder)
                self.app_state.video_folder = video_folder
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

            self._on_load_clicked()



    def _on_load_clicked(self):
        from .dialog_busy_progress import BusyProgressDialog

        dialog = BusyProgressDialog("Loading data...", parent=self)

        def _update(msg: str) -> None:
            dialog.setLabelText(msg)
            dialog.pump_events()

        self.app_state._progress_callback = _update
        dialog.execute_blocking(self.data_widget.on_load_clicked)

    def _clear_all_line_edits(self):
        for attr in ('nc_file_path_edit', 'nwb_file_path_edit', 'metadata_path_edit',
                  'video_folder_edit', 'audio_folder_edit', 'pose_folder_edit',
                  'ephys_path_edit', 'neurons_path_edit', 'label_file_path_edit',
                  'pred_file_path_edit'):
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
                "Path to .nc / .nwb / .npz file or pynapple folder"
            )

        browse_button = QPushButton("Browse")
        browse_button.setObjectName(f"{object_name}_browse_button")
        browse_button.clicked.connect(browse_callback)

        if object_name == "nc_file_path":
            self.import_labels_checkbox = QCheckBox("Import labels")
            self.import_labels_checkbox.setObjectName("import_labels_checkbox")
            self.import_labels_checkbox.setToolTip(
                "Load labels from {name}_labels.tsv alongside the .nc file.\n"
            )
            self.import_labels_checkbox.stateChanged.connect(self._on_import_labels_checked)
            self.import_labels_checkbox.setChecked(bool(self.app_state.import_labels_nc_data))

        clear_button = QPushButton("Clear")
        clear_button.setObjectName(f"{object_name}_clear_button")
        clear_button.clicked.connect(lambda: self._on_clear_path_clicked(object_name, line_edit))

        row_layout = QHBoxLayout()
        row_layout.addWidget(line_edit)
        row_layout.addWidget(browse_button)
        if object_name == "nc_file_path":
            browse_folder_button = QPushButton("Browse folder")
            browse_folder_button.setObjectName(f"{object_name}_browse_folder_button")
            browse_folder_button.setToolTip("Browse for a pynapple data folder")
            browse_folder_button.clicked.connect(self._browse_data_folder)
            row_layout.addWidget(browse_folder_button)
        row_layout.addWidget(clear_button)
        target_layout.addRow(label, row_layout)

        return line_edit

    def _on_import_labels_checked(self, state):
        self.app_state.import_labels_nc_data = state == 2
        if state == 2 and self.app_state.nc_file_path:
            tsv = labels_tsv_path(self.app_state.nc_file_path)
            if hasattr(self, "label_file_path_edit"):
                _populate_if_exists(self.label_file_path_edit, tsv)

    def _on_clear_path_clicked(self, object_name, line_edit):
        line_edit.setText("")
        attr_map = {
            "nc_file_path": "nc_file_path",
            "video_folder": "video_folder",
            "audio_folder": "audio_folder",
            "pose_folder": "pose_folder",
            "ephys_path": "ephys_path",
            "neurons_path": "neurons_path",
        }
        attr = attr_map.get(object_name)
        if attr:
            setattr(self.app_state, attr, None)
        self._update_human_verified_status()
        self._update_correct_offsets_status()
        self._update_purge_small_labels_status()
    # Device controls (populated after load)
    # ------------------------------------------------------------------

    def create_device_controls(self, catalog):
        self._create_labels_row_at_index()
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
            logger.error("Skipping ephys file %s: %s", Path(filepath).name, e)

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

        load_layout.addWidget(self.import_labels_checkbox)
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

    def _browse_nwb_file(self):
        """Browse for an NWB session/alignment file."""
        result = QFileDialog.getOpenFileName(
            None,
            caption="Open NWB session file",
            filter="NWB files (*.nwb);;All files (*)",
        )
        path = result[0] if result and len(result) >= 1 else ""
        if path:
            self.nwb_file_path_edit.setText(path)
            self.app_state.nwb_file_path = path  # auto-syncs to dt.nwb_path

    def _browse_metadata_file(self):
        """Browse for a metadata TSV file."""
        result = QFileDialog.getOpenFileName(
            None,
            caption="Open metadata file",
            filter="TSV files (*.tsv);;All files (*)",
        )
        path = result[0] if result and len(result) >= 1 else ""
        if path:
            self.metadata_path_edit.setText(path)
            self.app_state.metadata_path = path

    def _auto_discover_nwb(self):
        """Auto-discover alignment.nwb near the loaded data file."""
        from ethograph.utils.paths import find_nwb_file

        nc_path = self.app_state.nc_file_path
        if not nc_path:
            return
        data_dir = str(Path(nc_path).parent)
        nwb = find_nwb_file(data_dir)
        if nwb is not None:
            self.nwb_file_path_edit.setText(str(nwb))
            self.app_state.nwb_file_path = str(nwb)  # auto-syncs to dt.nwb_path
            logger.info("Auto-discovered NWB alignment: %s", nwb)

    def _auto_discover_metadata(self):
        """Auto-discover {stem}_metadata.tsv near the loaded data file."""
        nc_path = self.app_state.nc_file_path
        if not nc_path:
            return
        tsv = metadata_tsv_path(nc_path)
        _populate_if_exists(self.metadata_path_edit, tsv)

    def _browse_data_file(self):
        """Browse for a data file (.nc, .nwb, .npz)."""
        result = QFileDialog.getOpenFileName(
            None,
            caption="Open data file",
            filter="Data files (*.nc *.nwb *.npz);;All files (*)",
        )
        path = result[0] if result and len(result) >= 1 else ""
        if path:
            self.nc_file_path_edit.setText(path)
            self.app_state.nc_file_path = path

    def _browse_data_folder(self):
        """Browse for a pynapple data folder."""
        path = QFileDialog.getExistingDirectory(
            None,
            caption="Open pynapple data folder",
        )
        if path:
            self.nc_file_path_edit.setText(path)
            self.app_state.nc_file_path = path

    def _maybe_downsample_videos(self, folder: str) -> str:
        """Offer to downsample high-res videos. Returns the folder to use."""
        from .dialog_video_downsample import offer_downsample
        return offer_downsample(folder, parent=self)

    def on_browse_clicked(self, browse_type="file", media_type=None):
        if browse_type == "file":
            if media_type == "data":
                self._browse_data_file()
                return

            elif media_type == "labels":
                nc_parent = Path(self.app_state.nc_file_path).parent

                result = QFileDialog.getOpenFileName(
                    None,
                    caption="Load labels TSV file",
                    dir=str(nc_parent),
                    filter="TSV files (*.tsv)",
                )
                labels_file_path = result[0] if result and len(result) >= 1 else ""
                if not labels_file_path:
                    return

                self.app_state._all_labels_df = load_labels_tsv(labels_file_path)

                self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)
                self.label_file_path_edit.setText(labels_file_path)

                if hasattr(self, "changepoints_widget") and self.changepoints_widget:
                    self.changepoints_widget._update_cp_status()
                if self.labels_widget:
                    self.labels_widget._mark_changes_unsaved()
                    self.labels_widget.refresh_labels_shapes_layer()
                self._update_human_verified_status()
                self._update_correct_offsets_status()
                self._update_purge_small_labels_status()
                if self.data_widget:
                    self.data_widget.update_main_plot(preserve_x_range=True)
                    if self.data_widget.plot_container:
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
                self._auto_detect_neurons(ephys_path)

        elif browse_type == "folder":
            if media_type == "video":
                caption = "Open folder with video files (e.g. mp4, mov)."
            elif media_type == "audio":
                caption = "Open folder with audio files (e.g. wav, mp3, mp4)."
            elif media_type == "pose":
                caption = "Open folder with pose files (e.g. .csv, .h5)."

            folder_path = QFileDialog.getExistingDirectory(None, caption=caption)

            if media_type == "video":
                if folder_path:
                    folder_path = self._maybe_downsample_videos(folder_path)
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

    def _browse_neurons(self):
        """Browse for a Kilosort folder or Pynapple file (.npz, .nwb)."""
        start_dir = self.app_state.neurons_path or ""
        if start_dir and Path(start_dir).is_file():
            start_dir = str(Path(start_dir).parent)

        dialog = QDialog(self)
        dialog.setWindowTitle("Load neuron data")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select the type of neuron data to load:"))

        btn_ks = QPushButton("Kilosort Folder")
        btn_ks.setToolTip("Select a Kilosort output folder (spike_times.npy, etc.)")
        btn_nap = QPushButton("Pynapple File (.npz / .nwb)")
        btn_nap.setToolTip("Select a Pynapple-compatible file with a 'units' TsGroup")

        layout.addWidget(btn_ks)
        layout.addWidget(btn_nap)

        chosen_path = [None]

        def _on_kilosort():
            folder = QFileDialog.getExistingDirectory(
                dialog, "Select Kilosort output folder", start_dir,
            )
            if folder:
                chosen_path[0] = folder
                dialog.accept()

        def _on_pynapple():
            path, _ = QFileDialog.getOpenFileName(
                dialog, "Select Pynapple file", start_dir,
                "Pynapple files (*.npz *.nwb);;All files (*)",
            )
            if path:
                chosen_path[0] = path
                dialog.accept()

        btn_ks.clicked.connect(_on_kilosort)
        btn_nap.clicked.connect(_on_pynapple)

        if dialog.exec_() == QDialog.Accepted and chosen_path[0]:
            self.neurons_path_edit.setText(chosen_path[0])
            self.app_state.neurons_path = chosen_path[0]
            self.neurons_path_edit.returnPressed.emit()

    def _auto_detect_neurons(self, ephys_path: str):
        ephys_parent = Path(ephys_path).parent
        for folder_name in ("kilosort4", "kilosort"):
            ks_folder = ephys_parent / folder_name
            if ks_folder.is_dir():
                self.neurons_path_edit.setText(str(ks_folder))
                self.app_state.neurons_path = str(ks_folder)
                return
        self.neurons_path_edit.clear()
        self.app_state.neurons_path = None

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
        self.import_predictions_btn.clicked.connect(self.labels_widget._import_predictions_from_folder)
        self.pred_confidence_pdf_btn.clicked.connect(self.labels_widget._plot_confidence_pdf)
        self.pred_confidence_threshold_spin.valueChanged.connect(self.labels_widget._on_confidence_threshold_changed)
        self.pred_segment_confidence_threshold_spin.valueChanged.connect(self.labels_widget._on_confidence_threshold_changed)

    def wire_ephys_signals(self, ephys_widget):
        """Connect neurons UI to EphysWidget methods."""
        self.neurons_path_edit.returnPressed.connect(ephys_widget._load_neurons)
