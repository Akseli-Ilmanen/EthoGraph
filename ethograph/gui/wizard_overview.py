"""Multi-step wizard for creating .nc files from multiple trials / modalities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.makepretty import styled_link

if TYPE_CHECKING:
    from ethograph.gui.wizard_media_files import FilePattern


# ─── shared state ─────────────────────────────────────────────────────────────


@dataclass
class ModalityConfig:
    enabled: bool = False
    file_mode: str = "single"  # "single" | "aligned_to_trial" | "aligned_to_session"
    single_file_path: str = ""
    folder_path: str = ""
    pattern: FilePattern | None = None
    nested_subfolders: bool = False
    fps: int = None
    fps_by_camera: dict[str, int] = field(default_factory=dict)
    audio_sr: float = None
    n_channels: int = 1
    source_software: str = "DeepLabCut"
    constant_offset: float = 0.0
    video_motion: bool = False
    kilosort_folder: str = ""
    ephys_sr: int = None
    gap_mode: str = "gap_between"  # "gap_between" | "onset_interval"
    gap_value: float = 0.0
    offset_constant_across_devices: bool = True
    device_offsets: dict[str, float] = field(default_factory=dict)

    @property
    def is_aligned_mode(self) -> bool:
        return self.file_mode == "aligned_to_trial"

    @property
    def is_continuous_mode(self) -> bool:
        return self.file_mode == "aligned_to_session"


@dataclass
class WizardState:
    video: ModalityConfig = field(default_factory=ModalityConfig)
    pose: ModalityConfig = field(default_factory=ModalityConfig)
    audio: ModalityConfig = field(default_factory=ModalityConfig)
    npy: ModalityConfig = field(default_factory=ModalityConfig)
    ephys: ModalityConfig = field(default_factory=ModalityConfig)

    files_aligned_to_trials: bool = True
    trial_table: pd.DataFrame | None = None
    trial_table_path: str | None = None  # Path to imported CSV/TSV

    camera_names: list[str] = field(default_factory=list)
    mic_names: list[str] = field(default_factory=list)
    pose_camera_mapping: list[tuple[str, str]] = field(default_factory=list)

    individuals: list[str] = field(default_factory=list)
    output_path: str = ""

    file_durations: dict[str, dict[str, float]] = field(default_factory=dict)

    def modality_configs(self) -> list[tuple[str, ModalityConfig]]:
        return [
            ("video", self.video),
            ("pose", self.pose),
            ("audio", self.audio),
            ("npy", self.npy),
            ("ephys", self.ephys),
        ]

    def enabled_modalities(self) -> list[tuple[str, ModalityConfig]]:
        return [(name, cfg) for name, cfg in self.modality_configs() if cfg.enabled]

    def has_aligned_modalities(self) -> bool:
        return any(cfg.is_aligned_mode for _, cfg in self.enabled_modalities())

    def has_continuous_modalities(self) -> bool:
        return any(cfg.is_continuous_mode for _, cfg in self.enabled_modalities())

    def is_fully_aligned(self) -> bool:
        """Only scenario where is fully aligned, if trial intervals correspond to files, and video, pose, and audio are all aligned."""
        enabled = self.enabled_modalities()
        non_ephys = [(name, cfg) for name, cfg in enabled if name != "ephys"]
        return bool(non_ephys) and all(cfg.is_aligned_mode for _, cfg in non_ephys) and not self.has_continuous_modalities()


# ─── Page 0: mode selection ──────────────────────────────────────────────────


class _ModeSelectionPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Create trials.nc file</b><br>"
            "Select how your data is organized:"
        ))
        layout.addSpacing(12)

        self._top_group = QButtonGroup(self)

        # --- Single file section ---
        single_box = QGroupBox("Single trial")
        sb_lay = QVBoxLayout(single_box)
        self._explanation = QLabel("Allows multiple modalities but only one file per modality (video/audio/pose/ephys). Great for quickly gettign started.")
        self._rb_pose = QRadioButton("1) Generate from pose file (DeepLabCut, SLEAP, ...)")
        self._rb_xarray = QRadioButton("2) Generate from xarray dataset (Movement style)")
        self._rb_audio = QRadioButton("3) Generate from audio file")
        self._rb_npy = QRadioButton("4) Generate from npy file")
        self._rb_ephys = QRadioButton("5) Generate from ephys file and/or kilosort folder")
        self._single_radios = [
            self._rb_pose, self._rb_xarray, self._rb_audio,
            self._rb_npy, self._rb_ephys,
        ]
        for rb in self._single_radios:
            sb_lay.addWidget(rb)
            self._top_group.addButton(rb)
        self._rb_pose.setChecked(True)
        layout.addWidget(single_box)

        # --- Multi file section ---
        multi_box = QGroupBox("Multiple trials")
        mb_lay = QVBoxLayout(multi_box)
        self._rb_multi = QRadioButton("Configure multi-trial dataset from multiple files within/across modalities with custom meta data.")
        self._top_group.addButton(self._rb_multi)
        mb_lay.addWidget(self._rb_multi)
        layout.addWidget(multi_box)

        # --- NWB section ---
        nwb_box = QGroupBox("NWB file")
        nb_lay = QVBoxLayout(nwb_box)
        self._rb_nwb_local = QRadioButton("1) Local .nwb file")
        self._rb_nwb_dandi = QRadioButton("2) DANDI archive (streaming)")
        self._top_group.addButton(self._rb_nwb_local)
        self._top_group.addButton(self._rb_nwb_dandi)
        nb_lay.addWidget(self._rb_nwb_local)
        nb_lay.addWidget(self._rb_nwb_dandi)
        layout.addWidget(nwb_box)

        layout.addSpacing(10)
        tut_box = QGroupBox("Examples")
        tut_lay = QVBoxLayout(tut_box)
        tut_text = QLabel(
            "These are real-world datasets that have been created or converted to trials.nc format:"
        )
        tut_text.setWordWrap(True)
        tut_lay.addWidget(tut_text)
        tut_lay.addSpacing(5)
        tut_link = QLabel(styled_link(
            "https://github.com/Akseli-Ilmanen/EthoGraph/tree/main/tutorials",
            "View tutorials for creating custom .nc files"
        ))
        tut_link.setOpenExternalLinks(True)
        tut_link.setTextFormat(Qt.RichText)
        tut_lay.addWidget(tut_link)
        layout.addWidget(tut_box)
        layout.addStretch()

    def get_mode(self) -> str:
        if self._rb_multi.isChecked():
            return "multi"
        if self._rb_nwb_local.isChecked() or self._rb_nwb_dandi.isChecked():
            return "nwb"
        return "single"

    def get_single_type(self) -> str:
        for rb, name in zip(
            self._single_radios,
            ["pose", "xarray", "audio", "npy", "ephys"],
        ):
            if rb.isChecked():
                return name

    def is_nwb_local(self) -> bool:
        return self._rb_nwb_local.isChecked()


# ─── Page 1: modality selection ──────────────────────────────────────────────


class _ModalitySelectionPage(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>Step 1 — Select modalities</b><br>"
            "Check each data type and choose aligned-trial files or continuous-session recording."
        ))
        layout.addSpacing(10)

        self._rows: dict[str, dict] = {}
        for name, label, allow_multi, device_label in [
            ("video", "Video", True, "Cameras"),
            ("pose", "Pose", True, "Cameras"),
            ("audio", "Audio", True, "Mics"),
            ("ephys", "Ephys", False, ""),
        ]:
            row_widget, row_data = self._build_modality_row(name, label, allow_multi, device_label)
            layout.addWidget(row_widget)
            self._rows[name] = row_data

        layout.addStretch()

    def _build_modality_row(
        self, name: str, label: str, allow_multi: bool, device_label: str
    ) -> tuple[QWidget, dict]:
        box = QGroupBox()
        box.setCheckable(True)
        box.setChecked(False)
        box_lay = QVBoxLayout(box)

        cb_label = QCheckBox(label)
        cb_label.setChecked(False)
        box.toggled.connect(cb_label.setChecked)
        cb_label.toggled.connect(box.setChecked)
        box_lay.addWidget(cb_label)

        data: dict = {"box": box, "checkbox": cb_label}

        if allow_multi:
            mode_row = QHBoxLayout()
            rb_single = QRadioButton("Single file")
            if name in {"video", "pose", "audio"}:
                rb_multi_reg = QRadioButton(
                    f"Files aligned to trial period (Trials x {device_label})"
                )
                rb_multi_irr = QRadioButton(
                    f"Continuous recording across session ({device_label})"
                )
            else:
                rb_multi_reg = QRadioButton("Multiple files (regular/no gaps)")
                rb_multi_irr = QRadioButton("Multiple files (variable gaps)")
            rb_single.setChecked(True)
            bg = QButtonGroup(box)
            bg.addButton(rb_single)
            bg.addButton(rb_multi_reg)
            bg.addButton(rb_multi_irr)
            mode_row.addWidget(rb_single)
            mode_row.addWidget(rb_multi_reg)
            mode_row.addWidget(rb_multi_irr)
            mode_row.addStretch()
            box_lay.addLayout(mode_row)
            data["rb_single"] = rb_single
            data["rb_multi_reg"] = rb_multi_reg
            data["rb_multi_irr"] = rb_multi_irr
        else:
            hint = QLabel("    (continuous recording across session)")
            hint.setStyleSheet("color: #888;")
            box_lay.addWidget(hint)
            data["rb_single"] = None

        return box, data

    def collect_state(self, state: WizardState) -> None:
        for name in ["video", "pose", "audio", "ephys"]:
            row = self._rows[name]
            cfg: ModalityConfig = getattr(state, name)
            cfg.enabled = row["box"].isChecked()
            if row.get("rb_single") is None:
                cfg.file_mode = "aligned_to_session" if cfg.enabled else "single"
            elif row["rb_single"].isChecked():
                cfg.file_mode = "single"
            elif row["rb_multi_reg"].isChecked():
                cfg.file_mode = "aligned_to_trial"
                cfg.gap_mode = "gap_between"
                cfg.gap_value = 0.0
            else:
                cfg.file_mode = "aligned_to_session"
                # Reset gap values for irregular mode
                cfg.gap_mode = "gap_between"
                cfg.gap_value = 0.0

    def validate(self) -> str | None:
        if not any(self._rows[n]["box"].isChecked() for n in self._rows):
            return "Please select at least one modality."
        return None


# ─── main wizard dialog ─────────────────────────────────────────────────────


class NCWizardDialog(QDialog):
    def __init__(self, app_state, io_widget, parent: QWidget | None = None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self._state = WizardState()

        self.setWindowTitle("Create trials.nc — Wizard")
        self.setMinimumWidth(950)
        self.setMinimumHeight(750)
        self.resize(1050, 800)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._stack = QStackedWidget()
        self._page_mode = _ModeSelectionPage()
        self._page_modality = _ModalitySelectionPage()
        self._stack.addWidget(self._page_mode)
        self._stack.addWidget(self._page_modality)

        # Pages 2-4 are created lazily
        self._page_config = None
        self._page_trials = None
        self._page_timeline = None

        layout.addWidget(self._stack)

        # Navigation bar
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Previous")
        self._prev_btn.clicked.connect(self._on_previous)
        self._prev_btn.setEnabled(False)
        self._prev_btn.setAutoDefault(False)
        self._prev_btn.setDefault(False)

        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._on_next)
        self._next_btn.setAutoDefault(False)
        self._next_btn.setDefault(False)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        cancel_btn.setAutoDefault(False)
        cancel_btn.setDefault(False)

        nav.addWidget(self._prev_btn)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        nav.addWidget(cancel_btn)
        layout.addLayout(nav)

    def _current_page(self) -> int:
        return self._stack.currentIndex()

    def _on_previous(self):
        page = self._current_page()
        if page > 0:
            self._stack.setCurrentIndex(page - 1)
            self._update_nav()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_next(self):
        page = self._current_page()

        if page == 0:
            self._handle_mode_selection()
        elif page == 1:
            err = self._page_modality.validate()
            if err:
                QMessageBox.warning(self, "Input error", err)
                return
            self._page_modality.collect_state(self._state)
            self._ensure_config_page()
            self._stack.setCurrentIndex(2)
            self._update_nav()
        elif page == 2:
            err = self._page_config.validate(self._state)
            if err:
                QMessageBox.warning(self, "Input error", err)
                return
            self._page_config.collect_state(self._state)
            self._ensure_trials_page()
            self._page_trials.populate_from_state(self._state)
            self._stack.setCurrentIndex(3)
            self._update_nav()
        elif page == 3:
            err = self._page_trials.validate(self._state)
            if err:
                QMessageBox.warning(self, "Input error", err)
                return
            self._page_trials.collect_state(self._state)
            self._ensure_timeline_page()
            from ethograph.gui.dialog_busy_progress import BusyProgressDialog
            progress = BusyProgressDialog("Scanning files…", parent=self)
            _, err = progress.execute(self._page_timeline.populate_from_state, self._state)
            if err:
                QMessageBox.critical(self, "Error", f"Failed to scan files:\n{err}")
                return
            self._stack.setCurrentIndex(4)
            self._update_nav()
        elif page == 4:
            self._page_timeline.collect_state(self._state)
            self._generate()

    def _handle_mode_selection(self):
        mode = self._page_mode.get_mode()

        if mode == "single":
            single_type = self._page_mode.get_single_type()
            self._open_single_dialog(single_type)

        elif mode == "nwb":
            self._open_nwb_dialog()

        elif mode == "multi":
            self._stack.setCurrentIndex(1)
            self._update_nav()

    def _open_single_dialog(self, single_type: str):
        from ethograph.gui.wizard_single import (
            AudioFileDialog,
            EphysFileDialog,
            NpyFileDialog,
            PoseFileDialog,
            XarrayDatasetDialog,
        )

        dialog_map = {
            "pose": PoseFileDialog,
            "xarray": XarrayDatasetDialog,
            "audio": AudioFileDialog,
            "npy": NpyFileDialog,
            "ephys": EphysFileDialog,
        }
        dialog_cls = dialog_map[single_type]
        dialog = dialog_cls(self.app_state, self.io_widget, self)
        if dialog.exec_():
            self.accept()

    def _open_nwb_dialog(self):
        from ethograph.gui.wizard_nwb import NWBImportDialog

        dialog = NWBImportDialog(self.app_state, self.io_widget, self)
        if self._page_mode.is_nwb_local():
            dialog._page_source._rb_local.setChecked(True)
        else:
            dialog._page_source._rb_dandi.setChecked(True)
            dialog._page_source._toggle_source()
        if dialog.exec_():
            self.accept()

    def _ensure_config_page(self):
        from ethograph.gui.wizard_multi_tabs import ModalityConfigPage

        if self._page_config is not None:
            self._stack.removeWidget(self._page_config)
            self._page_config.deleteLater()
        self._page_config = ModalityConfigPage(self._state)
        self._stack.insertWidget(2, self._page_config)

    def _ensure_trials_page(self):
        from ethograph.gui.wizard_multi_trials import TrialsPage

        if self._page_trials is not None:
            self._stack.removeWidget(self._page_trials)
            self._page_trials.deleteLater()
        self._page_trials = TrialsPage()
        self._stack.insertWidget(3, self._page_trials)

    def _ensure_timeline_page(self):
        from ethograph.gui.wizard_multi_timeline import TimelinePage

        if self._page_timeline is not None:
            self._stack.removeWidget(self._page_timeline)
            self._page_timeline.deleteLater()
        self._page_timeline = TimelinePage()
        self._stack.insertWidget(4, self._page_timeline)

    def _update_nav(self):
        page = self._current_page()
        self._prev_btn.setEnabled(page > 0)
        if page == 4:
            self._next_btn.setText("Generate .nc file")
        elif page == 0:
            self._next_btn.setText("Next →")
        else:
            self._next_btn.setText("Next →")

    def _generate(self):
        from ethograph.gui.dialog_busy_progress import BusyProgressDialog
        from ethograph.gui.wizard_multi_builder import build_multi_trial_dt

        output_path = self._state.output_path
        if not output_path:
            QMessageBox.warning(self, "Missing output", "Please select an output path.")
            return

        def _build():
            return build_multi_trial_dt(self._state)

        progress = BusyProgressDialog("Building TrialTree...", parent=self)
        (dt, error) = progress.execute(_build)

        if progress.was_cancelled or error:
            if error:
                QMessageBox.critical(self, "Error", f"Failed to create trials.nc:\n{error}")
            return

        save_progress = BusyProgressDialog("Saving .nc file…", parent=self)
        _, save_error = save_progress.execute(dt.to_netcdf, output_path)
        if save_error:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{save_error}")
            return

        self._populate_io_fields()
        QMessageBox.information(self, "Success", f"Successfully created:\n{output_path}")
        self.accept()

    def _populate_io_fields(self):
        self.app_state.nc_file_path = self._state.output_path
        self.io_widget.nc_file_path_edit.setText(self._state.output_path)

        if self._state.video.enabled and self._state.video.folder_path:
            self.app_state.video_folder = self._state.video.folder_path
            self.io_widget.video_folder_edit.setText(self._state.video.folder_path)
        if self._state.audio.enabled and self._state.audio.folder_path:
            self.app_state.audio_folder = self._state.audio.folder_path
            if hasattr(self.io_widget, "audio_folder_edit"):
                self.io_widget.audio_folder_edit.setText(self._state.audio.folder_path)
        if self._state.pose.enabled and self._state.pose.folder_path:
            self.app_state.pose_folder = self._state.pose.folder_path
            self.io_widget.pose_folder_edit.setText(self._state.pose.folder_path)
