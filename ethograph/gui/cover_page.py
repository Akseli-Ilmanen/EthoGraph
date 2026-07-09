"""Full-screen start / cover page shown before a dataset is loaded.

Presents three entry points (matching the design in the project brief):

1. **Template datasets** — reuse :meth:`IOWidget._on_select_template_clicked`.
2. **Drag & drop files** — drop single, already-aligned media/feature/label
   files; ethograph classifies them by extension, optionally asks which video
   belongs to which camera, builds a throwaway ``alignment.tmp.nwb`` so the
   normal loader can consume loose media, and loads.
3. **Data wizard** — reuse :meth:`IOWidget._on_create_nc_clicked`.

The page is modal and disappears once a dataset is loaded
(``app_state.ready``).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.io.validation import (
    AUDIO_EXTENSIONS,
    EPHYS_EXTENSIONS,
    EPHYS_EXTENSIONS_RAW,
    POSE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)

from .notify import notify, notify_dialog

logger = logging.getLogger(__name__)

FEATURE_EXTENSIONS = {".nc", ".nwb", ".npz"}
LABEL_EXTENSIONS = {".tsv"}


def classify_files(paths: list[str]) -> dict[str, list[str]]:
    """Bucket dropped file paths by ethograph data type (by extension).

    A ``.nwb`` can be pose, ephys or a feature/session file; it is treated as a
    *session* (feature) file here since that is the loadable unit.  A folder is
    returned under the ``session`` bucket (pynapple folder).
    """
    buckets: dict[str, list[str]] = {
        "session": [],
        "video": [],
        "pose": [],
        "audio": [],
        "ephys": [],
        "labels": [],
        "unknown": [],
    }
    for p in paths:
        path = Path(p)
        if path.is_dir():
            buckets["session"].append(p)
            continue
        ext = path.suffix.lower()
        if ext in FEATURE_EXTENSIONS:
            buckets["session"].append(p)
        elif ext in VIDEO_EXTENSIONS:
            buckets["video"].append(p)
        elif ext in POSE_EXTENSIONS:
            buckets["pose"].append(p)
        elif ext in AUDIO_EXTENSIONS:
            buckets["audio"].append(p)
        elif ext in EPHYS_EXTENSIONS or ext in EPHYS_EXTENSIONS_RAW:
            buckets["ephys"].append(p)
        elif ext in LABEL_EXTENSIONS:
            buckets["labels"].append(p)
        else:
            buckets["unknown"].append(p)
    return buckets


def _audio_sample_rate(path: str) -> float:
    """Read a real audio sample rate (never hardcode a fallback).

    Tries soundfile (wav/flac/ogg), then PyAV (mp4/mov/mkv containers), then the
    stdlib ``wave`` module.  Raises if no audio stream / rate can be read.
    """
    try:
        import soundfile as sf

        return float(sf.info(path).samplerate)
    except Exception:  # noqa: BLE001 - fall through to PyAV / wave
        pass
    try:
        import av

        with av.open(path) as container:
            for stream in container.streams:
                if stream.type == "audio" and stream.rate:
                    return float(stream.rate)
    except Exception:  # noqa: BLE001 - fall through to wave
        pass
    import wave

    with wave.open(path, "rb") as w:
        return float(w.getframerate())


class _CamMatchDialog(QDialog):
    """Single-trial video↔camera / pose↔camera assignment.

    Shown when more than one video (or pose) file is dropped: the user orders
    the videos as cam1, cam2, … and the pose files are paired by row.
    """

    def __init__(self, videos: list[str], poses: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign cameras")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Order the videos top-to-bottom as cam1, cam2, …  Pose files on "
                "the same row are paired with that camera."
            )
        )

        cols = QHBoxLayout()
        self._video_list = QListWidget()
        self._video_list.setDragDropMode(QListWidget.InternalMove)
        for v in videos:
            self._video_list.addItem(Path(v).name)
        self._pose_list = QListWidget()
        self._pose_list.setDragDropMode(QListWidget.InternalMove)
        for p in poses:
            self._pose_list.addItem(Path(p).name)

        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("<b>Videos (cam order)</b>"))
        vbox.addWidget(self._video_list)
        pbox = QVBoxLayout()
        pbox.addWidget(QLabel("<b>Pose files</b>"))
        pbox.addWidget(self._pose_list)
        cols.addLayout(vbox)
        cols.addLayout(pbox)
        layout.addLayout(cols)

        self._videos = {Path(v).name: v for v in videos}
        self._poses = {Path(p).name: p for p in poses}

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def ordered_videos(self) -> list[str]:
        return [self._videos[self._video_list.item(i).text()] for i in range(self._video_list.count())]

    def ordered_poses(self) -> list[str]:
        return [self._poses[self._pose_list.item(i).text()] for i in range(self._pose_list.count())]


class _DropList(QListWidget):
    """A QListWidget that accepts file drops and records the paths."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.paths: list[str] = []
        self.setStyleSheet(
            "QListWidget { border: 2px dashed rgba(255,255,255,60); border-radius: 8px;"
            " min-height: 180px; }"
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p and p not in self.paths:
                self.paths.append(p)
                self.addItem(Path(p).name)
        event.acceptProposedAction()

    def clear_paths(self):
        self.paths = []
        self.clear()


class CoverPage(QDialog):
    """Modal start page shown until a dataset is loaded."""

    def __init__(self, shell, io_widget, parent=None):
        super().__init__(parent or shell)
        self.shell = shell
        self.io_widget = io_widget
        self.app_state = io_widget.app_state
        self.setWindowTitle("ethograph — get started")
        self.setModal(True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(16)

        intro = QLabel(
            "You have 3 options:  1) Test the GUI with template datasets,  "
            "2) Drag & drop single aligned media files,  "
            "3) Use the ethograph custom loader — the wizard can help."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 12pt;")
        outer.addWidget(intro)

        body = QHBoxLayout()
        body.setSpacing(24)
        body.addLayout(self._build_left_column(), 1)
        body.addWidget(self._vline())
        body.addLayout(self._build_right_column(), 1)
        outer.addLayout(body)

        skip_row = QHBoxLayout()
        skip_row.addStretch()
        skip_btn = QPushButton("Skip / open empty")
        skip_btn.clicked.connect(self.reject)
        skip_row.addWidget(skip_btn)
        outer.addLayout(skip_row)

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    @staticmethod
    def _vline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet("color: rgba(255,255,255,40);")
        return line

    def _build_left_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(10)

        template_btn = QPushButton("🐦  Template datasets")
        template_btn.setMinimumHeight(48)
        template_btn.clicked.connect(self._on_template)
        col.addWidget(template_btn)

        col.addWidget(QLabel("<b>Drag && drop files*</b>"))
        self._drop = _DropList()
        col.addWidget(self._drop, 1)

        row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._drop.clear_paths)
        load_btn = QPushButton("Load")
        load_btn.setMinimumHeight(36)
        load_btn.clicked.connect(self._on_load_dropped)
        row.addWidget(clear_btn)
        row.addWidget(load_btn, 1)
        col.addLayout(row)

        note = QLabel("*Files have to be aligned (single trial assumed).")
        note.setStyleSheet("color: rgba(255,255,255,120); font-size: 8pt;")
        col.addWidget(note)
        return col

    def _build_right_column(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(10)

        wizard_btn = QPushButton("🧙  Data wizard — prepare my data")
        wizard_btn.setMinimumHeight(48)
        wizard_btn.clicked.connect(self._on_wizard)
        col.addWidget(wizard_btn)

        desc = QLabel(
            "The custom loader lets you point at a session file (.nc / .nwb / "
            ".npz / pynapple folder) plus media folders, metadata and alignment.\n\n"
            "Use the wizard if your data is not yet aligned into a single session."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: rgba(255,255,255,150);")
        col.addWidget(desc)
        col.addStretch()
        return col

    # ------------------------------------------------------------------
    # Option handlers
    # ------------------------------------------------------------------

    def _on_template(self):
        self.io_widget._on_select_template_clicked()
        self._close_if_loaded()

    def _on_wizard(self):
        self.io_widget._on_create_nc_clicked()
        # The wizard populates fields but may not auto-load; trigger a load if a
        # session path is now set.
        if getattr(self.app_state, "nc_file_path", None) and not self._is_loaded():
            self.io_widget._on_load_clicked()
        self._close_if_loaded()

    def _on_load_dropped(self):
        paths = list(self._drop.paths)
        if not paths:
            notify("Drop some files first.", "warning")
            return
        buckets = classify_files(paths)
        try:
            self._populate_io_from_buckets(buckets)
        except Exception as e:  # noqa: BLE001 - outermost GUI boundary
            logger.exception("Failed to prepare dropped files")
            notify_dialog(f"Could not prepare dropped files:\n{e}", "critical")
            return
        self.io_widget._on_load_clicked()
        self._close_if_loaded()

    # ------------------------------------------------------------------
    # Drag & drop → IO fields
    # ------------------------------------------------------------------

    def _populate_io_from_buckets(self, buckets: dict[str, list[str]]):
        io = self.io_widget
        app_state = self.app_state

        videos = buckets["video"]
        poses = buckets["pose"]

        cam_map: list[tuple[str, str | None]] = []  # (video, pose|None)
        if len(videos) > 1 or len(poses) > 1:
            dlg = _CamMatchDialog(videos, poses, parent=self)
            if not dlg.exec_():
                raise RuntimeError("Camera assignment cancelled.")
            videos = dlg.ordered_videos()
            poses = dlg.ordered_poses()
        for i, v in enumerate(videos):
            cam_map.append((v, poses[i] if i < len(poses) else None))

        session_files = buckets["session"]
        if session_files:
            # A real session/feature file was provided — use it directly and let
            # the standard loader resolve media from the folders below.
            app_state.nc_file_path = session_files[0]
        else:
            # Pure media: synthesise a single-trial alignment.tmp.nwb.
            nwb_path = self._build_tmp_alignment(cam_map, buckets["audio"])
            app_state.nwb_file_path = str(nwb_path)
            app_state.nc_file_path = str(nwb_path)

        if videos:
            app_state.video_folder = str(Path(videos[0]).parent)
        if poses:
            app_state.pose_folder = str(Path(poses[0]).parent)
        if buckets["audio"]:
            app_state.audio_folder = str(Path(buckets["audio"][0]).parent)
        if buckets["ephys"]:
            app_state.ephys_path = buckets["ephys"][0]
        if buckets["labels"] and hasattr(io, "import_labels_checkbox"):
            io.import_labels_checkbox.setChecked(True)

    def _build_tmp_alignment(self, cam_map, audio_files) -> Path:
        """Create a single-trial alignment.tmp.nwb from loose media files."""
        import pandas as pd

        from ethograph.gui.video_manager import probe_video
        from ethograph.io.nwb_alignment import align_media_from_streams

        if not cam_map:
            raise RuntimeError("No video files to build an alignment from.")

        streams: list[dict] = []
        stop_time = 0.0
        for i, (video, pose) in enumerate(cam_map):
            probe = probe_video(video)
            fps = probe.fps
            if not fps:
                raise RuntimeError(f"Could not read frame rate from {Path(video).name}.")
            duration = probe.nframes / fps if probe.nframes else 0.0
            stop_time = max(stop_time, duration)
            streams.append({"name": f"video_cam-{i + 1}", "files": [video], "rate": fps})
            if pose:
                # Pose shares the matching video's frame rate (per spec).
                streams.append({"name": f"pose_cam-{i + 1}", "files": [pose], "rate": fps})

        for j, audio in enumerate(audio_files):
            streams.append(
                {
                    "name": f"audio_mic-{j + 1}",
                    "files": [audio],
                    "rate": _audio_sample_rate(audio),
                    "starting_time": 0.0,
                }
            )

        if stop_time <= 0.0:
            raise RuntimeError("Could not determine session duration from the video.")

        trials = pd.DataFrame({"trial": [1], "start_time": [0.0], "stop_time": [stop_time]})

        out_dir = Path(tempfile.gettempdir()) / "ethograph_tmp_alignment"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "alignment.tmp.nwb"
        if out_path.exists():
            out_path.unlink()
        align_media_from_streams(trials, streams, out_path)
        return out_path

    # ------------------------------------------------------------------
    # Loaded-state helpers
    # ------------------------------------------------------------------

    def _is_loaded(self) -> bool:
        return bool(getattr(self.app_state, "ready", False)) or (
            getattr(self.app_state, "dt", None) is not None
        )

    def _close_if_loaded(self):
        if self._is_loaded():
            self.accept()


def maybe_show_cover_page(shell) -> None:
    """Show the cover page at startup unless a dataset is already loaded."""
    meta = getattr(shell, "meta_widget", None)
    io_widget = getattr(meta, "io_widget", None)
    if io_widget is None:
        return
    app_state = io_widget.app_state
    if getattr(app_state, "ready", False) or getattr(app_state, "dt", None) is not None:
        return
    page = CoverPage(shell, io_widget)
    shell._cover_page = page
    # Size to most of the window.
    page.resize(int(shell.width() * 0.9), int(shell.height() * 0.9))
    page.exec_()
