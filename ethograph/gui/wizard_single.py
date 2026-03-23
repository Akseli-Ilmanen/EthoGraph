"""Dialog for creating .nc files from various data sources."""

import webbrowser
from pathlib import Path
from typing import Optional, get_args

import av
import numpy as np
import xarray as xr
from movement.io import load_poses
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.data_loader import (
    wizard_single_from_audio,
    wizard_single_from_ds,
    wizard_single_from_ephys,
    wizard_single_from_npy_file,
    wizard_single_from_pose,
)
from ethograph.gui.wizard_nwb import NWBImportDialog
from ethograph.utils.audio import get_audio_sr
from ethograph.utils.validation import (
    AUDIO_FILE_FILTER,
    EPHYS_EXTENSIONS_STR,
    EPHYS_FILE_FILTER,
    VIDEO_FILE_FILTER,
)



def get_video_fps(video_path: str) -> Optional[int]:
    """Read FPS from video file using PyAV, rounded to nearest integer."""
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            return round(fps)
    except (OSError, ValueError, ZeroDivisionError):
        return None



AVAILABLE_SOFTWARES = list(get_args(load_poses.from_file.__annotations__["source_software"]))

MOVEMENT_DOCS_URL = "https://movement.neuroinformatics.dev/latest/user_guide/movement_dataset.html"
TUTORIALS_URL = "https://github.com/Akseli-Ilmanen/EthoGraph/tree/main/tutorials"


class PoseFileDialog(QDialog):
    """Dialog for generating .nc file from pose estimation output."""

    def __init__(self, app_state, io_widget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Generate from Pose File")
        self.setMinimumWidth(550)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()


        self.software_combo = QComboBox()
        self.software_combo.addItems(AVAILABLE_SOFTWARES)
        form_layout.addRow("Source software:", self.software_combo)

        pose_widget = QWidget()
        pose_layout = QHBoxLayout(pose_widget)
        pose_layout.setContentsMargins(0, 0, 0, 0)
        self.pose_edit = QLineEdit()
        self.pose_edit.setPlaceholderText("Select pose file...")
        self.pose_edit.setReadOnly(True)
        pose_browse = QPushButton("Browse")
        pose_browse.clicked.connect(self._on_pose_browse)
        pose_layout.addWidget(self.pose_edit)
        pose_layout.addWidget(pose_browse)
        form_layout.addRow("Pose file:", pose_widget)
        

        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Select video file (.mp4, .mov) - optional")
        self.video_edit.setReadOnly(True)
        video_browse = QPushButton("Browse")
        video_browse.clicked.connect(self._on_video_browse)
        video_clear = QPushButton("Clear")
        video_clear.clicked.connect(lambda: self.video_edit.clear())
        video_layout.addWidget(self.video_edit)
        video_layout.addWidget(video_browse)
        video_layout.addWidget(video_clear)
        form_layout.addRow("Video file (optional):", video_widget)



        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 1000)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.setSuffix(" fps")
        form_layout.addRow("Frame rate:", self.fps_spinbox)

        self.video_offset_spinbox = QDoubleSpinBox()
        self.video_offset_spinbox.setRange(-100000.0, 100000.0)
        self.video_offset_spinbox.setDecimals(4)
        self.video_offset_spinbox.setValue(0.0)
        self.video_offset_spinbox.setSuffix(" s")
        self.video_offset_spinbox.setToolTip(
            "Time (seconds) where video starts relative to pose stream. "
            "Negative means video starts earlier."
        )
        form_layout.addRow("Video onset in pose:", self.video_offset_spinbox)

        output_widget = QWidget()
        output_layout = QHBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output location for trials.nc...")
        self.output_edit.setReadOnly(True)
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self._on_output_browse)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse)
        form_layout.addRow("Output path:", output_widget)

        layout.addLayout(form_layout)
        layout.addSpacing(20)

        self.generate_button = QPushButton("Generate .nc file")
        self.generate_button.clicked.connect(self._on_generate)
        layout.addWidget(self.generate_button)

        layout.addSpacing(10)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_pose_browse(self):
        software = self.software_combo.currentText()
        result = QFileDialog.getOpenFileName(
            self,
            caption=f"Select {software} pose file",
        )
        if result and result[0]:
            self.pose_edit.setText(result[0])

    def _on_video_browse(self):
        result = QFileDialog.getOpenFileName(
            self,
            caption="Select video file",
            filter=VIDEO_FILE_FILTER,
        )
        if result and result[0]:
            self.video_edit.setText(result[0])
            fps = get_video_fps(result[0])
            if fps is not None:
                self.fps_spinbox.setValue(int(fps))

    def _on_output_browse(self):
        result = QFileDialog.getSaveFileName(
            self,
            caption="Save trials.nc file",
            filter="NetCDF files (*.nc);;All files (*)",
        )
        if result and result[0]:
            path = result[0]
            if not path.endswith(".nc"):
                path += ".nc"
            self.output_edit.setText(path)

    def _on_generate(self):
        if not self.pose_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a pose file.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output path.")
            return

        software = self.software_combo.currentText()
        pose_path = self.pose_edit.text()
        video_path = self.video_edit.text() or None
        fps = self.fps_spinbox.value()
        video_offset = self.video_offset_spinbox.value()
        output_path = self.output_edit.text()

        try:
            dt = wizard_single_from_pose(
                video_path=video_path,
                fps=fps,
                pose_path=pose_path,
                source_software=software,
                video_offset=video_offset,
            )
            dt.to_netcdf(output_path)

            self._populate_io_fields(output_path, video_path, pose_path=pose_path)

            QMessageBox.information(
                self,
                "Success",
                f"Successfully created:\n{output_path}",
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create trials.nc file:\n{e}")

    def _populate_io_fields(self, output_path: str, video_path: Optional[str], pose_path: str):
        pose_folder = str(Path(pose_path).parent)

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

        if video_path:
            video_folder = str(Path(video_path).parent)
            self.app_state.video_folder = video_folder
            self.io_widget.video_folder_edit.setText(video_folder)

        self.app_state.pose_folder = pose_folder
        self.io_widget.pose_folder_edit.setText(pose_folder)


class XarrayDatasetDialog(QDialog):
    """Dialog for loading an xarray dataset (Movement style)."""

    def __init__(self, app_state, io_widget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Load xarray Dataset (Movement style)")
        self.setMinimumWidth(550)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Load a Movement-style xarray dataset. "
            '<a href="' + MOVEMENT_DOCS_URL + '">See Movement documentation</a> '
            "for the expected format."
        )
        info_label.setOpenExternalLinks(True)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addSpacing(15)

        form_layout = QFormLayout()


        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Select video file (.mp4, .mov) - optional")
        self.video_edit.setReadOnly(True)
        video_browse = QPushButton("Browse")
        video_browse.clicked.connect(self._on_video_browse)
        video_clear = QPushButton("Clear")
        video_clear.clicked.connect(lambda: self.video_edit.clear())
        video_layout.addWidget(self.video_edit)
        video_layout.addWidget(video_browse)
        video_layout.addWidget(video_clear)
        form_layout.addRow("Video file (optional):", video_widget)

        self.video_offset_spinbox = QDoubleSpinBox()
        self.video_offset_spinbox.setRange(-100000.0, 100000.0)
        self.video_offset_spinbox.setDecimals(4)
        self.video_offset_spinbox.setValue(0.0)
        self.video_offset_spinbox.setSuffix(" s")
        self.video_offset_spinbox.setToolTip(
            "Time (seconds) where video starts relative to dataset stream. "
            "Negative means video starts earlier."
        )
        form_layout.addRow("Video onset in dataset:", self.video_offset_spinbox)


        dataset_widget = QWidget()
        dataset_layout = QHBoxLayout(dataset_widget)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText("Select Movement dataset (.nc)...")
        self.dataset_edit.setReadOnly(True)
        dataset_browse = QPushButton("Browse")
        dataset_browse.clicked.connect(self._on_dataset_browse)
        dataset_layout.addWidget(self.dataset_edit)
        dataset_layout.addWidget(dataset_browse)
        form_layout.addRow("Dataset file:", dataset_widget)



        output_widget = QWidget()
        output_layout = QHBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output location for trials.nc...")
        self.output_edit.setReadOnly(True)
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self._on_output_browse)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse)
        form_layout.addRow("Output path:", output_widget)

        layout.addLayout(form_layout)
        layout.addSpacing(20)

        self.generate_button = QPushButton("Generate .nc file")
        self.generate_button.clicked.connect(self._on_generate)
        layout.addWidget(self.generate_button)

        layout.addSpacing(10)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_dataset_browse(self):
        result = QFileDialog.getOpenFileName(
            self,
            caption="Select Movement dataset file",
            filter="NetCDF files (*.nc);;All files (*)",
        )
        if result and result[0]:
            self.dataset_edit.setText(result[0])

    def _on_video_browse(self):
        result = QFileDialog.getOpenFileName(
            self,
            caption="Select video file",
            filter=VIDEO_FILE_FILTER,
        )
        if result and result[0]:
            self.video_edit.setText(result[0])

    def _on_output_browse(self):
        result = QFileDialog.getSaveFileName(
            self,
            caption="Save trials.nc file",
            filter="NetCDF files (*.nc);;All files (*)",
        )
        if result and result[0]:
            path = result[0]
            if not path.endswith(".nc"):
                path += ".nc"
            self.output_edit.setText(path)

    def _on_generate(self):
        if not self.dataset_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a dataset file.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output path.")
            return

        dataset_path = self.dataset_edit.text()
        video_path = self.video_edit.text() or None
        video_offset = self.video_offset_spinbox.value()
        output_path = self.output_edit.text()

        try:
            ds = xr.open_dataset(dataset_path, engine="netcdf4")
            dt = wizard_single_from_ds(video_path=video_path, ds=ds, video_offset=video_offset)
            dt.to_netcdf(output_path)

            self._populate_io_fields(output_path, video_path)

            QMessageBox.information(
                self,
                "Success",
                f"Successfully created:\n{output_path}",
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create trials.nc file:\n{e}")

    def _populate_io_fields(self, output_path: str, video_path: Optional[str]):

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

        if video_path:
            video_folder = str(Path(video_path).parent)
            self.app_state.video_folder = video_folder
            self.io_widget.video_folder_edit.setText(video_folder)


class AudioFileDialog(QDialog):
    """Dialog for generating .nc file from audio file."""

    def __init__(self, app_state, io_widget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Generate from Audio File")
        self.setMinimumWidth(550)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Generate a .nc file from audio data. "
            "If your .mp4 video contains audio, you can use that file as the audio source."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addSpacing(15)

        form_layout = QFormLayout()

        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Select video file (.mp4, .mov) — optional")
        self.video_edit.setReadOnly(True)
        video_browse = QPushButton("Browse")
        video_browse.clicked.connect(self._on_video_browse)
        video_clear = QPushButton("Clear")
        video_clear.clicked.connect(lambda: self.video_edit.clear())
        video_layout.addWidget(self.video_edit)
        video_layout.addWidget(video_browse)
        video_layout.addWidget(video_clear)
        form_layout.addRow("Video file (optional):", video_widget)

        audio_widget = QWidget()
        audio_layout = QHBoxLayout(audio_widget)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("Select audio file (.wav, .mp3, .mp4)...")
        self.audio_edit.setReadOnly(True)
        audio_browse = QPushButton("Browse")
        audio_browse.clicked.connect(self._on_audio_browse)
        audio_layout.addWidget(self.audio_edit)
        audio_layout.addWidget(audio_browse)
        form_layout.addRow("Audio file:", audio_widget)

        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 1000)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.setSuffix(" fps")
        form_layout.addRow("Video frame rate:", self.fps_spinbox)

        self.video_offset_spinbox = QDoubleSpinBox()
        self.video_offset_spinbox.setRange(-100000.0, 100000.0)
        self.video_offset_spinbox.setDecimals(4)
        self.video_offset_spinbox.setValue(0.0)
        self.video_offset_spinbox.setSuffix(" s")
        self.video_offset_spinbox.setToolTip(
            "Time (seconds) where video starts relative to audio stream. "
            "Negative means video starts earlier."
        )
        form_layout.addRow("Video onset in audio:", self.video_offset_spinbox)

        self.audio_sr_spinbox = QDoubleSpinBox()
        self.audio_sr_spinbox.setRange(1000, 192000)
        self.audio_sr_spinbox.setDecimals(2)
        self.audio_sr_spinbox.setValue(44100)
        self.audio_sr_spinbox.setSuffix(" Hz")
        self.audio_sr_spinbox.setReadOnly(True)
        self.audio_sr_spinbox.setToolTip("Sample rate determined by audio file.")
        form_layout.addRow("Audio sample rate:", self.audio_sr_spinbox)

        self.individuals_edit = QLineEdit()
        self.individuals_edit.setPlaceholderText("e.g., bird1, bird2, bird3 (leave empty for default)")
        form_layout.addRow("Individuals (optional):", self.individuals_edit)

        self.video_motion_checkbox = QCheckBox("Load video motion features")
        self.video_motion_checkbox.setToolTip("Extract motion features from video (may take longer)")
        form_layout.addRow("", self.video_motion_checkbox)

        output_widget = QWidget()
        output_layout = QHBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output location for trials.nc...")
        self.output_edit.setReadOnly(True)
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self._on_output_browse)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse)
        form_layout.addRow("Output path:", output_widget)

        layout.addLayout(form_layout)
        layout.addSpacing(20)

        self.generate_button = QPushButton("Generate .nc file")
        self.generate_button.clicked.connect(self._on_generate)
        layout.addWidget(self.generate_button)

        layout.addSpacing(10)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_video_browse(self):
        result = QFileDialog.getOpenFileName(
            self,
            caption="Select video file",
            filter=VIDEO_FILE_FILTER,
        )
        if result and result[0]:
            self.video_edit.setText(result[0])
            fps = get_video_fps(result[0])
            if fps is not None:
                self.fps_spinbox.setValue(int(fps))

    def _on_audio_browse(self):
        result = QFileDialog.getOpenFileName(
            self,
            caption="Select audio file",
            filter=AUDIO_FILE_FILTER,
        )
        if result and result[0]:
            self.audio_edit.setText(result[0])
            audio_sr = get_audio_sr(result[0])
            self.audio_sr_spinbox.setValue(audio_sr)

    def _on_output_browse(self):
        result = QFileDialog.getSaveFileName(
            self,
            caption="Save trials.nc file",
            filter="NetCDF files (*.nc);;All files (*)",
        )
        if result and result[0]:
            path = result[0]
            if not path.endswith(".nc"):
                path += ".nc"
            self.output_edit.setText(path)

    def _on_generate(self):
        if not self.audio_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an audio file.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output path.")
            return

        video_path = self.video_edit.text() or None
        audio_path = self.audio_edit.text()
        fps = self.fps_spinbox.value()
        audio_sr = self.audio_sr_spinbox.value()
        video_offset = self.video_offset_spinbox.value()
        output_path = self.output_edit.text()

        individuals = None
        if self.individuals_edit.text().strip():
            individuals = [s.strip() for s in self.individuals_edit.text().split(",")]

        video_motion = self.video_motion_checkbox.isChecked()

        try:
            dt = wizard_single_from_audio(
                video_path=video_path,
                fps=fps,
                audio_path=audio_path,
                audio_sr=audio_sr,
                individuals=individuals,
                video_motion=video_motion,
                video_offset=video_offset,
            )
            dt.to_netcdf(output_path)

            self._populate_io_fields(output_path, video_path, audio_path)

            QMessageBox.information(
                self,
                "Success",
                f"Successfully created:\n{output_path}",
            )
            self.accept()
        except Exception as e:
            print(f"Error creating trials.nc file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create trials.nc file:\n{e}")

    def _populate_io_fields(self, output_path: str, video_path: Optional[str], audio_path: str):
        audio_folder = str(Path(audio_path).parent)

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

        if video_path:
            video_folder = str(Path(video_path).parent)
            self.app_state.video_folder = video_folder
            self.io_widget.video_folder_edit.setText(video_folder)

        self.app_state.audio_folder = audio_folder
        self.io_widget.audio_folder_edit.setText(audio_folder)


class NpyFileDialog(QDialog):
    """Dialog for generating .nc file from npy file."""

    def __init__(self, app_state, io_widget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Generate from Npy File")
        self.setMinimumWidth(550)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Generate a .nc file from a numpy (.npy) file. "
            "The file should contain a 2D array with shape (n_frames, n_variables)."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addSpacing(15)

        form_layout = QFormLayout()

        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Select video file (.mp4, .mov) - optional")
        self.video_edit.setReadOnly(True)
        video_browse = QPushButton("Browse")
        video_browse.clicked.connect(self._on_video_browse)
        video_clear = QPushButton("Clear")
        video_clear.clicked.connect(lambda: self.video_edit.clear())
        video_layout.addWidget(self.video_edit)
        video_layout.addWidget(video_browse)
        video_layout.addWidget(video_clear)
        form_layout.addRow("Video file (optional):", video_widget)

        npy_widget = QWidget()
        npy_layout = QHBoxLayout(npy_widget)
        npy_layout.setContentsMargins(0, 0, 0, 0)
        self.npy_edit = QLineEdit()
        self.npy_edit.setPlaceholderText("Select numpy file (.npy)...")
        self.npy_edit.setReadOnly(True)
        npy_browse = QPushButton("Browse")
        npy_browse.clicked.connect(self._on_npy_browse)
        npy_layout.addWidget(self.npy_edit)
        npy_layout.addWidget(npy_browse)
        form_layout.addRow("Npy file:", npy_widget)

        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 1000)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.setSuffix(" fps")
        form_layout.addRow("Video frame rate:", self.fps_spinbox)

        self.video_offset_spinbox = QDoubleSpinBox()
        self.video_offset_spinbox.setRange(-100000.0, 100000.0)
        self.video_offset_spinbox.setDecimals(4)
        self.video_offset_spinbox.setValue(0.0)
        self.video_offset_spinbox.setSuffix(" s")
        self.video_offset_spinbox.setToolTip(
            "Time (seconds) where video starts relative to npy data stream. "
            "Negative means video starts earlier."
        )
        form_layout.addRow("Video onset in npy:", self.video_offset_spinbox)

        self.data_sr_spinbox = QSpinBox()
        self.data_sr_spinbox.setRange(1, 100000)
        self.data_sr_spinbox.setValue(30)
        self.data_sr_spinbox.setSuffix(" Hz")
        form_layout.addRow("Data sampling rate:", self.data_sr_spinbox)

        self.individuals_edit = QLineEdit()
        self.individuals_edit.setPlaceholderText("e.g., bird1, bird2, bird3 (leave empty for default)")
        form_layout.addRow("Individuals (optional):", self.individuals_edit)

        self.video_motion_checkbox = QCheckBox("Load video motion features")
        self.video_motion_checkbox.setToolTip("Extract motion features from video (may take longer)")
        form_layout.addRow("", self.video_motion_checkbox)

        output_widget = QWidget()
        output_layout = QHBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output location for trials.nc...")
        self.output_edit.setReadOnly(True)
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self._on_output_browse)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse)
        form_layout.addRow("Output path:", output_widget)

        layout.addLayout(form_layout)
        layout.addSpacing(20)

        self.generate_button = QPushButton("Generate .nc file")
        self.generate_button.clicked.connect(self._on_generate)
        layout.addWidget(self.generate_button)

        layout.addSpacing(10)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_video_browse(self):
        result = QFileDialog.getOpenFileName(
            self,
            caption="Select video file",
            filter=VIDEO_FILE_FILTER,
        )
        if result and result[0]:
            self.video_edit.setText(result[0])
            fps = get_video_fps(result[0])
            if fps is not None:
                self.fps_spinbox.setValue(int(fps))

    def _on_npy_browse(self):
        result = QFileDialog.getOpenFileName(
            self,
            caption="Select numpy file",
            filter="Numpy files (*.npy);;All files (*)",
        )
        if result and result[0]:
            self.npy_edit.setText(result[0])

    def _on_output_browse(self):
        result = QFileDialog.getSaveFileName(
            self,
            caption="Save trials.nc file",
            filter="NetCDF files (*.nc);;All files (*)",
        )
        if result and result[0]:
            path = result[0]
            if not path.endswith(".nc"):
                path += ".nc"
            self.output_edit.setText(path)

    def _on_generate(self):
        if not self.npy_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a npy file.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output path.")
            return

        video_path = self.video_edit.text() or None
        npy_path = self.npy_edit.text()
        fps = self.fps_spinbox.value()
        data_sr = self.data_sr_spinbox.value()
        video_offset = self.video_offset_spinbox.value()
        output_path = self.output_edit.text()

        individuals = None
        if self.individuals_edit.text().strip():
            individuals = [s.strip() for s in self.individuals_edit.text().split(",")]

        video_motion = self.video_motion_checkbox.isChecked()

        try:
            dt = wizard_single_from_npy_file(
                video_path=video_path,
                fps=fps,
                npy_path=npy_path,
                data_sr=data_sr,
                individuals=individuals,
                video_motion=video_motion,
                video_offset=video_offset,
            )
            dt.to_netcdf(output_path)

            self._populate_io_fields(output_path, video_path)

            QMessageBox.information(
                self,
                "Success",
                f"Successfully created:\n{output_path}",
            )
            self.accept()
        except Exception as e:
            print(f"Error creating trials.nc file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create trials.nc file:\n{e}")

    def _populate_io_fields(self, output_path: str, video_path: Optional[str]):

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

        if video_path:
            video_folder = str(Path(video_path).parent)
            self.app_state.video_folder = video_folder
            self.io_widget.video_folder_edit.setText(video_folder)


class EphysFileDialog(QDialog):
    """Dialog for generating .nc file from ephys recording and/or kilosort folder."""

    EPHYS_FILTER = EPHYS_FILE_FILTER

    def __init__(self, app_state, io_widget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("Generate from Ephys File and/or Kilosort Folder")
        self.setMinimumWidth(550)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Generate a .nc file from an electrophysiology recording and/or kilosort folder.\n"
            f"Supported ephys extensions: {EPHYS_EXTENSIONS_STR}\n\n"
            "At least one of ephys file or kilosort folder is required."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addSpacing(15)

        form_layout = QFormLayout()

        # --- Ephys file ---
        ephys_widget = QWidget()
        ephys_layout = QHBoxLayout(ephys_widget)
        ephys_layout.setContentsMargins(0, 0, 0, 0)
        self.ephys_edit = QLineEdit()
        self.ephys_edit.setPlaceholderText("Select ephys file...")
        self.ephys_edit.setReadOnly(True)
        ephys_browse = QPushButton("Browse")
        ephys_browse.clicked.connect(self._on_ephys_browse)
        ephys_clear = QPushButton("Clear")
        ephys_clear.clicked.connect(lambda: self.ephys_edit.clear())
        ephys_layout.addWidget(self.ephys_edit)
        ephys_layout.addWidget(ephys_browse)
        ephys_layout.addWidget(ephys_clear)
        form_layout.addRow("Ephys file:", ephys_widget)

        # --- Kilosort folder ---
        ks_widget = QWidget()
        ks_layout = QHBoxLayout(ks_widget)
        ks_layout.setContentsMargins(0, 0, 0, 0)
        self.kilosort_edit = QLineEdit()
        self.kilosort_edit.setPlaceholderText("Select kilosort output folder...")
        self.kilosort_edit.setReadOnly(True)
        ks_browse = QPushButton("Browse")
        ks_browse.clicked.connect(self._on_kilosort_browse)
        ks_clear = QPushButton("Clear")
        ks_clear.clicked.connect(lambda: self.kilosort_edit.clear())
        ks_layout.addWidget(self.kilosort_edit)
        ks_layout.addWidget(ks_browse)
        ks_layout.addWidget(ks_clear)
        form_layout.addRow("Kilosort folder:", ks_widget)

        # --- Ephys sampling rate ---
        self.sr_spinbox = QSpinBox()
        self.sr_spinbox.setRange(1, 200000)
        self.sr_spinbox.setValue(30000)
        self.sr_spinbox.setSuffix(" Hz")
        self.sr_spinbox.setToolTip(
            "Auto-detected from ephys file or kilosort params.py; set manually for raw binary"
        )
        form_layout.addRow("Ephys sampling rate:", self.sr_spinbox)

        # --- N channels ---
        self.n_channels_spinbox = QSpinBox()
        self.n_channels_spinbox.setRange(1, 10000)
        self.n_channels_spinbox.setValue(1)
        self.n_channels_spinbox.setToolTip("Auto-detected for known formats; set manually for raw binary")
        form_layout.addRow("N channels:", self.n_channels_spinbox)

        # --- Video file (optional) ---
        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Select video file (.mp4, .mov) — optional")
        self.video_edit.setReadOnly(True)
        video_browse = QPushButton("Browse")
        video_browse.clicked.connect(self._on_video_browse)
        video_clear = QPushButton("Clear")
        video_clear.clicked.connect(lambda: self.video_edit.clear())
        video_layout.addWidget(self.video_edit)
        video_layout.addWidget(video_browse)
        video_layout.addWidget(video_clear)
        form_layout.addRow("Video file (optional):", video_widget)

        # --- Video frame rate ---
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 1000)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.setSuffix(" fps")
        form_layout.addRow("Video frame rate:", self.fps_spinbox)

        # --- Video onset in ephys ---
        self.video_offset_spinbox = QDoubleSpinBox()
        self.video_offset_spinbox.setRange(-100000.0, 100000.0)
        self.video_offset_spinbox.setDecimals(4)
        self.video_offset_spinbox.setValue(0.0)
        self.video_offset_spinbox.setSuffix(" s")
        self.video_offset_spinbox.setToolTip(
            "Time (seconds) where the video starts relative to the ephys "
            "recording start. Negative if video started before ephys."
        )
        form_layout.addRow("Video onset in ephys:", self.video_offset_spinbox)

        # --- Video motion checkbox ---
        self.video_motion_checkbox = QCheckBox("Load video motion features")
        form_layout.addRow("", self.video_motion_checkbox)

        # --- Audio file (optional) ---
        audio_widget = QWidget()
        audio_layout = QHBoxLayout(audio_widget)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("Select audio file (.wav, .mp3, .mp4) — optional")
        self.audio_edit.setReadOnly(True)
        audio_browse = QPushButton("Browse")
        audio_browse.clicked.connect(self._on_audio_browse)
        audio_clear = QPushButton("Clear")
        audio_clear.clicked.connect(lambda: self.audio_edit.clear())
        audio_layout.addWidget(self.audio_edit)
        audio_layout.addWidget(audio_browse)
        audio_layout.addWidget(audio_clear)
        form_layout.addRow("Audio file (optional):", audio_widget)

        # --- Audio sampling rate ---
        self.audio_sr_spinbox = QDoubleSpinBox()
        self.audio_sr_spinbox.setRange(1000, 192000)
        self.audio_sr_spinbox.setDecimals(2)
        self.audio_sr_spinbox.setValue(44100)
        self.audio_sr_spinbox.setSuffix(" Hz")
        self.audio_sr_spinbox.setReadOnly(True)
        self.audio_sr_spinbox.setToolTip("Sample rate determined by audio file.")
        form_layout.addRow("Audio sampling rate:", self.audio_sr_spinbox)

        # --- Audio onset in ephys ---
        self.audio_offset_spinbox = QDoubleSpinBox()
        self.audio_offset_spinbox.setRange(-100000.0, 100000.0)
        self.audio_offset_spinbox.setDecimals(4)
        self.audio_offset_spinbox.setValue(0.0)
        self.audio_offset_spinbox.setSuffix(" s")
        self.audio_offset_spinbox.setToolTip(
            "Time (seconds) where the audio starts relative to the ephys "
            "recording start. Negative if audio started before ephys."
        )
        form_layout.addRow("Audio onset in ephys:", self.audio_offset_spinbox)

        # --- Individuals (optional) ---
        self.individuals_edit = QLineEdit()
        self.individuals_edit.setPlaceholderText("e.g., bird1, bird2 (leave empty for default)")
        form_layout.addRow("Individuals (optional):", self.individuals_edit)

        # --- Output path ---
        output_widget = QWidget()
        output_layout = QHBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select output location for trials.nc...")
        self.output_edit.setReadOnly(True)
        output_browse = QPushButton("Browse")
        output_browse.clicked.connect(self._on_output_browse)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse)
        form_layout.addRow("Output path:", output_widget)

        layout.addLayout(form_layout)
        layout.addSpacing(20)

        self.generate_button = QPushButton("Generate .nc file")
        self.generate_button.clicked.connect(self._on_generate)
        layout.addWidget(self.generate_button)

        layout.addSpacing(10)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_ephys_browse(self):
        result = QFileDialog.getOpenFileName(
            self, caption="Select ephys file", filter=self.EPHYS_FILTER,
        )
        if not (result and result[0]):
            return
        self.ephys_edit.setText(result[0])
        self._probe_ephys_file(result[0])
        self._auto_detect_kilosort(result[0])

    def _probe_ephys_file(self, path: str):
        """Probe the ephys file to auto-detect SR and channels."""
        from ethograph.gui.plots_ephystrace import GenericEphysLoader

        try:
            loader = GenericEphysLoader(path)
        except ValueError:
            return

        self.sr_spinbox.setValue(int(loader.rate))
        self.n_channels_spinbox.setValue(loader.n_channels)

    def _auto_detect_kilosort(self, ephys_path: str):
        ks_folder = Path(ephys_path).parent / "kilosort4"
        if ks_folder.is_dir() and not self.kilosort_edit.text():
            self.kilosort_edit.setText(str(ks_folder))
            self._read_kilosort_params(ks_folder)

    def _on_kilosort_browse(self):
        start_dir = self.ephys_edit.text() or ""
        if start_dir:
            start_dir = str(Path(start_dir).parent)
        folder = QFileDialog.getExistingDirectory(
            self, "Select kilosort output folder", start_dir,
        )
        if folder:
            self.kilosort_edit.setText(folder)
            self._read_kilosort_params(Path(folder))

    def _read_kilosort_params(self, folder: Path):
        params_file = folder / "params.py"
        if not params_file.exists():
            return
        try:
            namespace = {}
            exec(params_file.read_text(), namespace)
            sr = namespace.get("sample_rate")
            if sr is not None:
                self.sr_spinbox.setValue(int(float(sr)))
        except (SyntaxError, ValueError, NameError, TypeError, OSError):
            pass

    def _on_video_browse(self):
        result = QFileDialog.getOpenFileName(
            self, caption="Select video file",
            filter=VIDEO_FILE_FILTER,
        )
        if result and result[0]:
            self.video_edit.setText(result[0])
            fps = get_video_fps(result[0])
            if fps is not None:
                self.fps_spinbox.setValue(int(fps))

    def _on_audio_browse(self):
        result = QFileDialog.getOpenFileName(
            self, caption="Select audio file",
            filter=AUDIO_FILE_FILTER,
        )
        if result and result[0]:
            self.audio_edit.setText(result[0])
            audio_sr = get_audio_sr(result[0])
            self.audio_sr_spinbox.setValue(audio_sr)

    def _on_output_browse(self):
        result = QFileDialog.getSaveFileName(
            self, caption="Save trials.nc file",
            filter="NetCDF files (*.nc);;All files (*)",
        )
        if result and result[0]:
            path = result[0]
            if not path.endswith(".nc"):
                path += ".nc"
            self.output_edit.setText(path)

    def _on_generate(self):
        has_ephys = bool(self.ephys_edit.text())
        has_kilosort = bool(self.kilosort_edit.text())
        if not has_ephys and not has_kilosort:
            QMessageBox.warning(
                self, "Missing Input",
                "Please select at least one of: ephys file or kilosort folder.",
            )
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output path.")
            return

        ephys_path = self.ephys_edit.text() or None
        kilosort_folder = self.kilosort_edit.text() or None
        video_path = self.video_edit.text() or None
        audio_path = self.audio_edit.text() or None
        fps = self.fps_spinbox.value()
        output_path = self.output_edit.text()

        individuals = None
        if self.individuals_edit.text().strip():
            individuals = [s.strip() for s in self.individuals_edit.text().split(",")]

        video_motion = self.video_motion_checkbox.isChecked()
        video_offset = self.video_offset_spinbox.value()
        audio_offset = self.audio_offset_spinbox.value()

        try:
            dt = wizard_single_from_ephys(
                video_path=video_path,
                fps=fps,
                audio_path=audio_path,
                individuals=individuals,
                video_motion=video_motion,
                video_offset=video_offset,
                audio_offset=audio_offset,
            )

            dt.to_netcdf(output_path)
            self._populate_io_fields(
                output_path, video_path, ephys_path, kilosort_folder, audio_path,
            )
            QMessageBox.information(
                self, "Success", f"Successfully created:\n{output_path}",
            )
            self.accept()
        except Exception as e:
            print(f"Error creating trials.nc file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create trials.nc file:\n{e}")

    def _populate_io_fields(
        self, output_path: str, video_path: Optional[str],
        ephys_path: Optional[str], kilosort_folder: Optional[str],
        audio_path: Optional[str],
    ):  
        

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)   


        if video_path:
            video_folder = str(Path(video_path).parent)
            self.app_state.video_folder = video_folder
            self.io_widget.video_folder_edit.setText(video_folder)

        if ephys_path:
            self.app_state.ephys_path = ephys_path
            if hasattr(self.io_widget, 'ephys_path_edit'):
                self.io_widget.ephys_path_edit.setText(ephys_path)

        if kilosort_folder:
            self.app_state.kilosort_folder = kilosort_folder
            if hasattr(self.io_widget, 'kilosort_folder_edit'):
                self.io_widget.kilosort_folder_edit.setText(kilosort_folder)

        if audio_path:
            audio_folder = str(Path(audio_path).parent)
            self.app_state.audio_folder = audio_folder
            if hasattr(self.io_widget, 'audio_folder_edit'):
                self.io_widget.audio_folder_edit.setText(audio_folder)


