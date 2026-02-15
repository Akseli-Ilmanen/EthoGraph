"""Dialog for creating .nc files from various data sources."""

import webbrowser
from pathlib import Path
from typing import Optional, get_args


import av
import xarray as xr
from movement.io import load_poses
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    minimal_dt_from_audio,
    minimal_dt_from_ds,
    minimal_dt_from_npy_file,
    minimal_dt_from_pose,
)
from ethograph.utils.audio import get_audio_sr



def get_video_fps(video_path: str) -> Optional[int]:
    """Read FPS from video file using PyAV, rounded to nearest integer."""
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            return round(fps)
    except Exception:
        return None



AVAILABLE_SOFTWARES = list(get_args(load_poses.from_file.__annotations__["source_software"]))

MOVEMENT_DOCS_URL = "https://movement.neuroinformatics.dev/latest/user_guide/movement_dataset.html"
TUTORIALS_URL = "https://github.com/Akseli-Ilmanen/EthoGraph/tree/main/tutorials"


class CreateNCDialog(QDialog):
    """Main dialog for selecting the data source type."""

    def __init__(self, app_state, io_widget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.app_state = app_state
        self.io_widget = io_widget
        self.setWindowTitle("➕ Create trials.nc file with own data")
        self.setMinimumWidth(450)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        description = QLabel(
            "Select how you want to create your trials.nc file (TrialTree style):"
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        layout.addSpacing(15)

        self.pose_button = QPushButton("1) Generate from pose file (DeepLabCut, SLEAP, ...)")
        self.pose_button.clicked.connect(self._on_pose_clicked)
        layout.addWidget(self.pose_button)

        self.xarray_button = QPushButton("2) Generate from xarray dataset (Movement style)")
        self.xarray_button.clicked.connect(self._on_xarray_clicked)
        layout.addWidget(self.xarray_button)

        self.audio_button = QPushButton("3) Generate from audio file")
        self.audio_button.clicked.connect(self._on_audio_clicked)
        layout.addWidget(self.audio_button)

        self.npy_button = QPushButton("4) Generate from npy file")
        self.npy_button.clicked.connect(self._on_npy_clicked)
        layout.addWidget(self.npy_button)

        self.tutorial_button = QPushButton("5) Tutorials for creating custom .nc files")
        self.tutorial_button.clicked.connect(self._on_tutorials_clicked)
        layout.addWidget(self.tutorial_button)

        layout.addSpacing(15)

        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_pose_clicked(self):
        dialog = PoseFileDialog(self.app_state, self.io_widget, self)
        if dialog.exec_():
            self.accept()

    def _on_xarray_clicked(self):
        dialog = XarrayDatasetDialog(self.app_state, self.io_widget, self)
        if dialog.exec_():
            self.accept()

    def _on_audio_clicked(self):
        dialog = AudioFileDialog(self.app_state, self.io_widget, self)
        if dialog.exec_():
            self.accept()

    def _on_npy_clicked(self):
        dialog = NpyFileDialog(self.app_state, self.io_widget, self)
        if dialog.exec_():
            self.accept()

    def _on_tutorials_clicked(self):
        webbrowser.open(TUTORIALS_URL)


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
        self.video_edit.setPlaceholderText("Select video file (.mp4, .mov)...")
        self.video_edit.setReadOnly(True)
        video_browse = QPushButton("Browse")
        video_browse.clicked.connect(self._on_video_browse)
        video_layout.addWidget(self.video_edit)
        video_layout.addWidget(video_browse)
        form_layout.addRow("Video file:", video_widget)

        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 1000)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.setSuffix(" fps")
        form_layout.addRow("Frame rate:", self.fps_spinbox)

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
            filter="Video files (*.mp4 *.mov *.avi);;All files (*)",
        )
        if result and result[0]:
            self.video_edit.setText(result[0])
            fps = get_video_fps(result[0])
            if fps is not None:
                self.fps_spinbox.setValue(fps)

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
        if not self.video_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a video file.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output path.")
            return

        software = self.software_combo.currentText()
        pose_path = self.pose_edit.text()
        video_path = self.video_edit.text()
        fps = self.fps_spinbox.value()
        output_path = self.output_edit.text()

        try:
            dt = minimal_dt_from_pose(
                video_path=video_path,
                fps=fps,
                pose_path=pose_path,
                source_software=software,
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

    def _populate_io_fields(self, output_path: str, video_path: str, pose_path: str):
        video_folder = str(Path(video_path).parent)
        pose_folder = str(Path(pose_path).parent)

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

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

        video_widget = QWidget()
        video_layout = QHBoxLayout(video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Select video file (.mp4, .mov)...")
        self.video_edit.setReadOnly(True)
        video_browse = QPushButton("Browse")
        video_browse.clicked.connect(self._on_video_browse)
        video_layout.addWidget(self.video_edit)
        video_layout.addWidget(video_browse)
        form_layout.addRow("Video file:", video_widget)

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
            filter="Video files (*.mp4 *.mov *.avi);;All files (*)",
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
        if not self.video_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a video file.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output path.")
            return

        dataset_path = self.dataset_edit.text()
        video_path = self.video_edit.text()
        output_path = self.output_edit.text()

        try:
            ds = xr.open_dataset(dataset_path)
            dt = minimal_dt_from_ds(video_path=video_path, ds=ds)
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

    def _populate_io_fields(self, output_path: str, video_path: str):
        video_folder = str(Path(video_path).parent)

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

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
        self.video_edit.setPlaceholderText("Select video file (.mp4, .mov)...")
        self.video_edit.setReadOnly(True)
        video_browse = QPushButton("Browse")
        video_browse.clicked.connect(self._on_video_browse)
        video_layout.addWidget(self.video_edit)
        video_layout.addWidget(video_browse)
        form_layout.addRow("Video file:", video_widget)

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

        self.audio_sr_spinbox = QSpinBox()
        self.audio_sr_spinbox.setRange(1000, 192000)
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
            filter="Video files (*.mp4 *.mov *.avi);;All files (*)",
        )
        if result and result[0]:
            self.video_edit.setText(result[0])
            fps = get_video_fps(result[0])
            if fps is not None:
                self.fps_spinbox.setValue(fps)

    def _on_audio_browse(self):
        result = QFileDialog.getOpenFileName(
            self,
            caption="Select audio file",
            filter="Audio files (*.wav *.mp3 *.mp4 *.flac);;All files (*)",
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
        if not self.video_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a video file.")
            return
        if not self.audio_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an audio file.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output path.")
            return

        video_path = self.video_edit.text()
        audio_path = self.audio_edit.text()
        fps = self.fps_spinbox.value()
        audio_sr = self.audio_sr_spinbox.value()
        output_path = self.output_edit.text()

        individuals = None
        if self.individuals_edit.text().strip():
            individuals = [s.strip() for s in self.individuals_edit.text().split(",")]

        video_motion = self.video_motion_checkbox.isChecked()

        try:
            dt = minimal_dt_from_audio(
                video_path=video_path,
                fps=fps,
                audio_path=audio_path,
                audio_sr=audio_sr,
                individuals=individuals,
                video_motion=video_motion,
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

    def _populate_io_fields(self, output_path: str, video_path: str, audio_path: str):
        video_folder = str(Path(video_path).parent)
        audio_folder = str(Path(audio_path).parent)

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

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
        self.video_edit.setPlaceholderText("Select video file (.mp4, .mov)...")
        self.video_edit.setReadOnly(True)
        video_browse = QPushButton("Browse")
        video_browse.clicked.connect(self._on_video_browse)
        video_layout.addWidget(self.video_edit)
        video_layout.addWidget(video_browse)
        form_layout.addRow("Video file:", video_widget)

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
            filter="Video files (*.mp4 *.mov *.avi);;All files (*)",
        )
        if result and result[0]:
            self.video_edit.setText(result[0])
            fps = get_video_fps(result[0])
            if fps is not None:
                self.fps_spinbox.setValue(fps)

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
        if not self.video_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a video file.")
            return
        if not self.npy_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select a npy file.")
            return
        if not self.output_edit.text():
            QMessageBox.warning(self, "Missing Input", "Please select an output path.")
            return

        video_path = self.video_edit.text()
        npy_path = self.npy_edit.text()
        fps = self.fps_spinbox.value()
        data_sr = self.data_sr_spinbox.value()
        output_path = self.output_edit.text()

        individuals = None
        if self.individuals_edit.text().strip():
            individuals = [s.strip() for s in self.individuals_edit.text().split(",")]

        video_motion = self.video_motion_checkbox.isChecked()

        try:
            dt = minimal_dt_from_npy_file(
                video_path=video_path,
                fps=fps,
                npy_path=npy_path,
                data_sr=data_sr,
                individuals=individuals,
                video_motion=video_motion,
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

    def _populate_io_fields(self, output_path: str, video_path: str):
        video_folder = str(Path(video_path).parent)

        self.app_state.nc_file_path = output_path
        self.io_widget.nc_file_path_edit.setText(output_path)

        self.app_state.video_folder = video_folder
        self.io_widget.video_folder_edit.setText(video_folder)
