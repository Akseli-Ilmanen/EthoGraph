"""Full-screen start / cover page shown before a dataset is loaded.

Presents three entry points (matching the design in the project brief):

1. **Template datasets** — reuse :meth:`IOWidget._on_select_template_clicked`.
2. **Drag & drop files** — drop single, already-aligned media/feature/label
   files; ethograph classifies them by extension, optionally asks which video
   belongs to which camera, builds a throwaway alignment NWB in the system
   temp dir (unique name per drop; stale ones are cleaned up best-effort) so
   the normal loader can consume loose media, and loads.
3. **Data wizard** — reuse :meth:`IOWidget._on_create_nc_clicked`.

The page runs *before* the main window is shown: it accepts once a dataset
is loaded (``app_state.ready``); closing the dialog (X / Esc) means the GUI
never opens.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
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
NPY_EXTENSIONS = {".npy"}
LABEL_EXTENSIONS = {".tsv"}
# Pose extensions whose source software cannot be inferred from the suffix alone
# (a ``.slp`` is always SLEAP; a ``.h5``/``.csv`` could be several tools).
AMBIGUOUS_POSE_EXTENSIONS = {".h5", ".hdf5", ".csv"}
# Ordered list of pose/bbox source softwares offered in the follow-up prompt.
POSE_SOFTWARES = ["DeepLabCut", "SLEAP", "LightningPose", "Anipose", "VIA-tracks"]

# One accent colour per entry point — repeated on the card border, the number
# badge and (for drag & drop) the drop zone, so the three options read as
# three distinct paths.
_ACCENTS = {
    "template": "#4fc3f7",
    "drop": "#81c784",
    "custom": "#ffb74d",
}


def classify_files(paths: list[str]) -> dict[str, list[str]]:
    """Bucket dropped file paths by ethograph data type (by extension).

    A ``.nwb`` can be pose, ephys or a feature/session file; it is treated as a
    *session* (feature) file here since that is the loadable unit.  A folder is a
    Kilosort output when it holds ``spike_times.npy`` (``neurons`` bucket),
    otherwise a pynapple ``session`` folder.
    """
    buckets: dict[str, list[str]] = {
        "session": [],
        "video": [],
        "pose": [],
        "audio": [],
        "ephys": [],
        "neurons": [],
        "npy": [],
        "labels": [],
        "unknown": [],
    }
    for p in paths:
        path = Path(p)
        if path.is_dir():
            if (path / "spike_times.npy").exists():
                buckets["neurons"].append(p)
            else:
                buckets["session"].append(p)
            continue
        ext = path.suffix.lower()
        if ext in FEATURE_EXTENSIONS:
            buckets["session"].append(p)
        elif ext in NPY_EXTENSIONS:
            buckets["npy"].append(p)
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


def _audio_info(path: str) -> tuple[float, float]:
    """Read a real audio (sample_rate, duration_s) — never hardcode a fallback.

    Tries soundfile (wav/flac/ogg), then PyAV (mp4/mov/mkv containers), then the
    stdlib ``wave`` module.  Raises if no audio stream / rate can be read.
    """
    try:
        import soundfile as sf

        info = sf.info(path)
        return float(info.samplerate), float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001 - fall through to PyAV / wave
        pass
    try:
        import av

        with av.open(path) as container:
            for stream in container.streams:
                if stream.type == "audio" and stream.rate:
                    duration = float(container.duration) / av.time_base if container.duration else 0.0
                    return float(stream.rate), duration
    except Exception:  # noqa: BLE001 - fall through to wave
        pass
    import wave

    with wave.open(path, "rb") as w:
        rate = float(w.getframerate())
        return rate, w.getnframes() / rate


class _CamMatchDialog(QDialog):
    """Single-trial video↔camera / pose↔camera assignment.

    Shown only when pose files need pairing with videos (poses dropped and
    more than one video or pose): the user orders the videos as cam1, cam2, …
    and the pose files are paired by row.
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


class _DropDetailsDialog(QDialog):
    """Follow-up prompt for the few dropped inputs that cannot be inferred.

    Everything detectable (video fps, audio rate, ephys params) is read on drop
    and never surfaced. Only genuinely unknowable values appear here: a numpy
    file's sample rate, and a pose file's source software when its extension is
    ambiguous. The dialog shows only the rows that are actually needed.
    """

    def __init__(self, need_npy_sr: bool, npy_name: str | None, need_pose_software: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("A few more details")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)

        self._sr_spin = None
        self._software_combo = None

        if need_npy_sr:
            layout.addWidget(
                QLabel(
                    f"<b>{npy_name}</b><br>Sampling rate of the numpy data "
                    "(samples per second) — this cannot be read from the file."
                )
            )
            row = QHBoxLayout()
            row.addWidget(QLabel("Data sampling rate:"))
            self._sr_spin = QSpinBox()
            self._sr_spin.setRange(1, 1000000)
            self._sr_spin.setValue(30)
            self._sr_spin.setSuffix(" Hz")
            row.addWidget(self._sr_spin, 1)
            layout.addLayout(row)

        if need_pose_software:
            layout.addWidget(QLabel("Which software produced the pose / tracking file?"))
            row = QHBoxLayout()
            row.addWidget(QLabel("Source software:"))
            self._software_combo = QComboBox()
            self._software_combo.addItems(POSE_SOFTWARES)
            row.addWidget(self._software_combo, 1)
            layout.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def data_sr(self) -> int | None:
        return self._sr_spin.value() if self._sr_spin is not None else None

    def source_software(self) -> str | None:
        return self._software_combo.currentText() if self._software_combo is not None else None


class _DropList(QListWidget):
    """A QListWidget that accepts file drops and records the paths."""

    def __init__(self, parent=None, accent: str = "rgba(255,255,255,60)"):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.paths: list[str] = []
        self.setStyleSheet(
            f"QListWidget {{ border: 2px dashed {accent}; border-radius: 8px;"
            " min-height: 160px; }"
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
        # No Qt parent by default: the main window is still hidden at startup,
        # and a dialog parented to a hidden window gets no Windows taskbar
        # entry (it can vanish behind other windows with no way back).
        super().__init__(parent)
        self.shell = shell
        self.io_widget = io_widget
        self.app_state = io_widget.app_state
        self._drop_tmp_dir: Path | None = None
        self.setWindowTitle("ethograph — get started")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(16)

        intro = QLabel("Three ways to get data into ethograph — pick one:")
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 13pt; font-weight: 600;")
        outer.addWidget(intro)

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addWidget(self._build_template_card(), 1)
        body.addWidget(self._build_drop_card(), 1)
        body.addWidget(self._build_custom_card(), 2)
        outer.addLayout(body)

        outer.addWidget(self._build_supported_types_strip())

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_supported_types_strip() -> QFrame:
        """A one-line reference of what can be dragged & dropped (with examples)."""

        def _fmt(exts) -> str:
            return " ".join(f"<code>{e}</code>" for e in sorted(exts))

        pose_estimation = (
            "<b>DeepLabCut</b> (<code>.csv</code>, <code>.h5</code>)"
            "  ·  <b>SLEAP</b> (<code>.slp</code>, <code>.h5</code>)"
            "  ·  <b>LightningPose</b> (<code>.csv</code>)"
            "  ·  <b>Anipose</b> (<code>.csv</code>)"
            "  ·  <b>VIA-tracks</b> (<code>.csv</code>)"
        )

        ephys_docs = "https://akseli-ilmanen.github.io/ethograph/getting_started/loading_ephys.html"
        ephys = (
            "all Neo-supported formats, raw data (<code>.dat</code>, <code>.bin</code>, …) "
            "and Kilosort folders — "
            f"<a href='{ephys_docs}'>see docs</a>"
        )

        rows = [
            ("🎞 Video", _fmt(VIDEO_EXTENSIONS)),
            ("🔊 Audio", _fmt(AUDIO_EXTENSIONS)),
            ("📈 Pose estimation", pose_estimation),
            ("⚡ Ephys", ephys),
            ("🧠 Neurons", "Kilosort folder  ·  pynapple <code>.npz</code> / <code>.nwb</code>"),
            ("📊 Features", _fmt(NPY_EXTENSIONS) + "  ·  session <code>.nc</code> / <code>.nwb</code> / <code>.npz</code>"),
            ("🏷 Labels", _fmt(LABEL_EXTENSIONS)),
        ]
        items = "".join(f"<li><b>{name}</b>&nbsp;&nbsp;{exts}</li>" for name, exts in rows)
        cells = f"<ul style='margin:0; -qt-list-indent:1;'>{items}</ul>"

        frame = QFrame()
        frame.setObjectName("typesStrip")
        frame.setStyleSheet(
            "QFrame#typesStrip { border-top: 1px solid rgba(255,255,255,25);"
            " padding-top: 8px; }"
            " QFrame#typesStrip code { color: #81c784; }"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(4, 6, 4, 0)
        lay.setSpacing(2)
        heading = QLabel("Supported files — drag any of these onto the drop zone:")
        heading.setStyleSheet("color: rgba(255,255,255,150); font-size: 10pt;")
        lay.addWidget(heading)
        body = QLabel(cells)
        body.setTextFormat(Qt.RichText)
        body.setOpenExternalLinks(True)
        body.setWordWrap(True)
        body.setStyleSheet("font-size: 10pt;")
        lay.addWidget(body)
        return frame

    @staticmethod
    def _make_card(num: int, title: str, subtitle: str, accent: str) -> tuple[QFrame, QVBoxLayout]:
        """A numbered, accent-coloured card holding one entry point."""
        card = QFrame()
        card.setObjectName("coverCard")
        card.setStyleSheet(
            f"QFrame#coverCard {{ border: 1px solid rgba(255,255,255,35);"
            f" border-top: 3px solid {accent}; border-radius: 10px;"
            f" background-color: rgba(255,255,255,10); }}"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QLabel(
            f'<span style="color:{accent}; font-size:16pt; font-weight:700;">{num}</span>'
            f'&nbsp;&nbsp;<span style="font-size:12pt; font-weight:600;">{title}</span>'
        )
        layout.addWidget(header)

        sub = QLabel(subtitle)
        sub.setWordWrap(True)
        sub.setStyleSheet("color: rgba(255,255,255,150);")
        layout.addWidget(sub)
        return card, layout

    def _build_template_card(self) -> QFrame:
        card, layout = self._make_card(
            1,
            "Template datasets",
            "Fastest way to try the GUI: download a ready-made example dataset.",
            _ACCENTS["template"],
        )
        # BMP text glyph (not a colour emoji) — renders as a stable black
        # symbol on Windows / macOS / Linux system fonts.
        template_btn = QPushButton("🐦‍⬛  Browse templates…")
        template_btn.setMinimumHeight(48)
        template_btn.clicked.connect(self._on_template)
        layout.addWidget(template_btn)
        layout.addStretch()
        return card

    def _build_drop_card(self) -> QFrame:
        card, layout = self._make_card(
            2,
            "Drag &amp; drop",
            "Quick exploration: drop single, already-aligned media / feature / "
            "label files (single trial assumed).",
            _ACCENTS["drop"],
        )
        self._drop = _DropList(accent=_ACCENTS["drop"])
        layout.addWidget(self._drop, 1)

        self._video_motion_cb = QCheckBox("Compute video motion — pixel change  (video only)")
        self._video_motion_cb.setToolTip(
            "Adds a motion-energy feature with a (time, camera) shape, so you can "
            "pick a camera in the Feature controls or view all cameras as a heatmap.\n"
            "Does nothing if no video is dropped."
        )
        layout.addWidget(self._video_motion_cb)

        row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._drop.clear_paths)
        load_btn = QPushButton("Load")
        load_btn.setMinimumHeight(36)
        load_btn.clicked.connect(self._on_load_dropped)
        row.addWidget(clear_btn)
        row.addWidget(load_btn, 1)
        layout.addLayout(row)
        return card

    def _build_custom_card(self) -> QFrame:
        card, layout = self._make_card(
            3,
            "Custom set-up",
            "Your own multi-trial data: point the loader at a session file "
            "(.nc / .nwb / .npz / pynapple folder) plus media folders. "
            "Run the wizard first if your data is not yet aligned.",
            _ACCENTS["custom"],
        )
        wizard_btn = QPushButton("🧙  Data wizard — prepare my data")
        wizard_btn.setMinimumHeight(48)
        wizard_btn.clicked.connect(self._on_wizard)
        layout.addWidget(wizard_btn)

        # Host for the IOWidget load panel (session/media path fields + Load
        # button). Borrowed on show, returned to the IO tab on hide.
        self._load_panel_borrowed = False
        self._load_host = QWidget()
        self._load_host_layout = QVBoxLayout(self._load_host)
        self._load_host_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._load_host)
        layout.addStretch()
        return card

    # ------------------------------------------------------------------
    # Load panel borrow / return (pattern shared with top-bar popups)
    # ------------------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self._borrow_load_panel()

    def hideEvent(self, event):
        self._return_load_panel()
        super().hideEvent(event)

    def _borrow_load_panel(self):
        """Reparent the IOWidget load panel into the right column."""
        if self._load_panel_borrowed:
            return
        self._load_panel_borrowed = True
        io = self.io_widget
        io.load_buttons_row.hide()
        self._load_host_layout.addWidget(io.load_panel)
        io.load_panel.show()
        # The panel's Load button runs IOWidget._on_load_clicked (connected at
        # IOWidget init, so it fires first and blocks until the load finishes);
        # this closes the cover page afterwards so the main window can show.
        io.load_button.clicked.connect(self._close_if_loaded)

    def _return_load_panel(self):
        """Give the load panel back to the IO tab (it must outlive this dialog)."""
        if not self._load_panel_borrowed:
            return
        self._load_panel_borrowed = False
        io = self.io_widget
        io.load_button.clicked.disconnect(self._close_if_loaded)
        self._load_host_layout.removeWidget(io.load_panel)
        io.load_buttons_row.show()
        # Re-insert at the top of the IO widget (original position).
        io.layout().insertWidget(0, io.load_panel)

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
            details = self._collect_drop_details(buckets)
            if details is None:
                return  # user cancelled the follow-up prompt
            self._populate_io_from_buckets(buckets, details)
        except Exception as e:  # noqa: BLE001 - outermost GUI boundary
            logger.exception("Failed to prepare dropped files")
            notify_dialog(f"Could not prepare dropped files:\n{e}", "critical")
            return
        self.io_widget._on_load_clicked()
        self._close_if_loaded()

    # ------------------------------------------------------------------
    # Drag & drop → IO fields
    # ------------------------------------------------------------------

    def _collect_drop_details(self, buckets: dict[str, list[str]]) -> dict | None:
        """Ask once for the dropped values that cannot be inferred.

        Returns a dict of resolved details, or ``None`` if the user cancelled.
        Skips the prompt entirely when nothing needs asking.
        """
        need_npy_sr = bool(buckets["npy"])
        ambiguous_pose = any(Path(p).suffix.lower() in AMBIGUOUS_POSE_EXTENSIONS for p in buckets["pose"])
        if not need_npy_sr and not ambiguous_pose:
            return {"data_sr": None, "source_software": None}

        npy_name = Path(buckets["npy"][0]).name if need_npy_sr else None
        dlg = _DropDetailsDialog(need_npy_sr, npy_name, ambiguous_pose, parent=self)
        if not dlg.exec_():
            return None
        return {"data_sr": dlg.data_sr(), "source_software": dlg.source_software()}

    def _populate_io_from_buckets(self, buckets: dict[str, list[str]], details: dict):
        io = self.io_widget
        app_state = self.app_state

        videos = buckets["video"]
        poses = buckets["pose"]

        if buckets["npy"]:
            self._populate_io_from_npy(buckets, details)
            return

        no_media = not (buckets["session"] or videos or poses or buckets["audio"])
        if no_media and (buckets["ephys"] or buckets["neurons"]):
            self._populate_io_from_ephys(buckets)
            return

        cam_map: list[tuple[str, str | None]] = []  # (video, pose|None)
        # The camera-order dialog only matters when pose files must be paired
        # with videos; videos alone get arbitrary cam-1, cam-2, … names.
        if poses and (len(videos) > 1 or len(poses) > 1):
            dlg = _CamMatchDialog(videos, poses, parent=self)
            if not dlg.exec_():
                raise RuntimeError("Camera assignment cancelled.")
            videos = dlg.ordered_videos()
            poses = dlg.ordered_poses()
        for i, v in enumerate(videos):
            cam_map.append((v, poses[i] if i < len(poses) else None))

        session_files = buckets["session"]
        has_media = bool(cam_map or buckets["audio"])
        if has_media:
            # Fresh per-drop temp dir so throwaway files never share a
            # .ethograph/local_settings.yaml with a previous drop.
            self._drop_tmp_dir = self._prepare_drop_dir()
        if session_files:
            # A real session/feature file was provided — use it directly. Media
            # dropped alongside it still needs a synthesised alignment (the
            # session file's folder usually has no .ethograph sidecar); the
            # loader picks it up via nwb_file_path.
            app_state.nc_file_path = session_files[0]
            if has_media:
                nwb_path = self._build_tmp_alignment(cam_map, buckets["audio"])
                app_state.nwb_file_path = str(nwb_path)
        else:
            # Pure media: synthesise a single-trial alignment.tmp.nwb.
            nwb_path = self._build_tmp_alignment(cam_map, buckets["audio"])
            app_state.nwb_file_path = str(nwb_path)
            if cam_map and self._video_motion_cb.isChecked():
                # Video motion requested → the session is an xarray .nc holding a
                # (time, camera) motion feature; media still comes from the tmp
                # alignment above. Feature/camera dropdown + heatmap come for free.
                app_state.nc_file_path = str(self._compute_video_motion_nc(cam_map))
            else:
                app_state.nc_file_path = str(nwb_path)

        # Drag & drop never takes a metadata table — setting nc_file_path above
        # reloads local settings, which can restore a stale metadata_path (e.g.
        # a previous drop's tmp alignment NWB). Clear it so the drop's own
        # trial timing wins; sidecar TSV discovery via source_path still works.
        app_state.metadata_path = None

        # A drop defines a fresh single-trial session, so it is authoritative:
        # clear every media folder first, then set only the ones actually dropped.
        # Otherwise a stale pose/audio folder from a previous drop survives (these
        # are persisted SCOPE_LOCAL fields) and _validate_media_files warns that
        # the new alignment has no pose/audio media.
        app_state.video_folder = None
        app_state.audio_folder = None
        app_state.pose_folder = None

        if videos:
            app_state.video_folder = str(Path(videos[0]).parent)
            if len(cam_map) > 1:
                # Several videos dropped = one trial filmed by multiple cameras in
                # parallel, so show every camera as its own view (not just cam-1).
                # Devices match the `video_cam-{i+1}` streams synthesised above.
                app_state.primary_camera = "cam-1"
                app_state.extra_cameras = [f"cam-{i + 1}" for i in range(1, len(cam_map))]
        if poses:
            app_state.pose_folder = str(Path(poses[0]).parent)
            app_state.source_software = details.get("source_software")
        if buckets["audio"]:
            app_state.audio_folder = str(Path(buckets["audio"][0]).parent)
        if buckets["ephys"]:
            app_state.ephys_path = buckets["ephys"][0]
        if buckets["neurons"]:
            app_state.neurons_path = buckets["neurons"][0]
        if buckets["labels"] and hasattr(io, "import_labels_checkbox"):
            io.import_labels_checkbox.setChecked(True)

    def _populate_io_from_npy(self, buckets: dict[str, list[str]], details: dict):
        """Convert a dropped .npy into a .nc (the one case that must persist data).

        A numpy array carries no time axis, so its sample rate is asked for via a
        follow-up prompt; an optional dropped video supplies fps and alignment.
        Other buckets are ignored here — npy is a standalone feature source.
        """
        from ethograph.gui.video_manager import probe_video
        from ethograph.io.data_loader import wizard_single_from_npy_file

        npy_path = buckets["npy"][0]
        video_path = buckets["video"][0] if buckets["video"] else None
        fps = probe_video(video_path).fps if video_path else None

        output_path = str(Path(npy_path).with_suffix(".nc"))
        dt = wizard_single_from_npy_file(
            video_path=video_path,
            fps=fps,
            npy_path=npy_path,
            data_sr=details["data_sr"],
            output_nc_path=output_path,
        )
        dt.to_netcdf(output_path)

        app_state = self.app_state
        app_state.nc_file_path = output_path
        app_state.metadata_path = None
        if video_path:
            app_state.video_folder = str(Path(video_path).parent)

    def _populate_io_from_ephys(self, buckets: dict[str, list[str]]):
        """Build a bare session .nc for an ephys- and/or kilosort-only drop.

        Ephys traces and neurons load from their own paths; this only writes a
        minimal session (with a single-trial alignment) so the loader has an
        anchor, mirroring the old ephys wizard dialog.
        """
        from ethograph.io.data_loader import wizard_single_from_ephys

        ephys_path = buckets["ephys"][0] if buckets["ephys"] else None
        neurons_path = buckets["neurons"][0] if buckets["neurons"] else None
        anchor = ephys_path or neurons_path
        output_path = str(Path(anchor).with_suffix(".nc")) if ephys_path else str(Path(anchor) / "session.nc")

        dt = wizard_single_from_ephys(output_nc_path=output_path)
        dt.to_netcdf(output_path)

        app_state = self.app_state
        app_state.nc_file_path = output_path
        app_state.metadata_path = None
        if ephys_path:
            app_state.ephys_path = ephys_path
        if neurons_path:
            app_state.neurons_path = neurons_path

    def _compute_video_motion_nc(self, cam_map) -> Path:
        """Compute per-camera motion energy behind a busy dialog and return the .nc."""
        from ethograph.gui.dialog_busy_progress import BusyProgressDialog

        dlg = BusyProgressDialog("Computing video motion…", parent=self)
        nc_path, error = dlg.execute(self._build_video_motion_nc, cam_map, self._drop_tmp_dir)
        if error or nc_path is None:
            raise RuntimeError(f"Could not compute video motion: {error}")
        return nc_path

    @staticmethod
    def _build_video_motion_nc(cam_map, out_dir: Path) -> Path:
        """Write a ``(time, camera)`` video-motion feature to a throwaway .nc.

        One motion-energy trace per dropped video, stacked on a ``camera`` dim
        (values ``cam-1``, ``cam-2``, … matching the alignment's video streams),
        so the catalog offers a camera dropdown and a heatmap view for free.
        """
        import numpy as np
        import xarray as xr

        from ethograph.features.movement import extract_video_motion
        from ethograph.gui.video_manager import probe_video

        motions: list[np.ndarray] = []
        cam_names: list[str] = []
        fps_used: float | None = None
        for i, (video, _pose) in enumerate(cam_map):
            fps = probe_video(video).fps
            if not fps:
                raise RuntimeError(f"Could not read frame rate from {Path(video).name}.")
            da = extract_video_motion(video, fps=fps, verbose=False)
            motions.append(np.asarray(da.values, dtype=float))
            cam_names.append(f"cam-{i + 1}")
            fps_used = fps

        max_len = max(len(m) for m in motions)
        arr = np.full((max_len, len(motions)), np.nan)
        for j, m in enumerate(motions):
            arr[: len(m), j] = m
        time = np.arange(max_len) / fps_used

        ds = xr.Dataset(
            {"video_motion": (["time", "camera"], arr)},
            # ``individuals`` is required by TrialTree validation even though
            # motion energy is not per-individual.
            coords={"time": time, "camera": cam_names, "individuals": ["individual 1"]},
        )
        ds.attrs["fps"] = fps_used

        out_path = out_dir / f"video_motion-{uuid4().hex[:8]}.nc"
        ds.to_netcdf(out_path)
        return out_path

    def _build_tmp_alignment(self, cam_map, audio_files) -> Path:
        """Create a single-trial alignment.tmp.nwb from loose media files."""
        import pandas as pd

        from ethograph.gui.video_manager import probe_video
        from ethograph.io.nwb_alignment import align_media_from_streams

        if not cam_map and not audio_files:
            raise RuntimeError("No media files to build an alignment from.")

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
            logger.info(
                "Drop alignment cam-%d: assumed fps=%s (from %s)%s",
                i + 1,
                fps,
                Path(video).name,
                f"; pose {Path(pose).name} uses this fps" if pose else "",
            )
            if pose:
                # Pose shares the matching video's frame rate (per spec).
                streams.append({"name": f"pose_cam-{i + 1}", "files": [pose], "rate": fps})

        for j, audio in enumerate(audio_files):
            rate, duration = _audio_info(audio)
            stop_time = max(stop_time, duration)
            streams.append(
                {
                    "name": f"audio_mic-{j + 1}",
                    "files": [audio],
                    "rate": rate,
                    "starting_time": 0.0,
                }
            )

        if stop_time <= 0.0:
            raise RuntimeError("Could not determine session duration from the dropped media.")

        trials = pd.DataFrame({"trial": [1], "start_time": [0.0], "stop_time": [stop_time]})

        out_path = self._drop_tmp_dir / f"alignment-{uuid4().hex[:8]}.tmp.nwb"
        align_media_from_streams(trials, streams, out_path)
        return out_path

    @staticmethod
    def _prepare_drop_dir() -> Path:
        """Return a fresh, empty per-drop temp dir for throwaway alignment/.nc.

        Each drop gets its OWN subdirectory so its ``.ethograph/local_settings.yaml``
        starts empty — a shared directory would leak a previous drop's panel layout
        (e.g. one saved with no video panel) into the next drop, so dropped media
        would silently fail to appear. Older drop dirs are removed best-effort;
        a dir whose files are still open (Windows locks HDF5) is simply left.
        """
        base = Path(tempfile.gettempdir()) / "ethograph_tmp_alignment"
        base.mkdir(parents=True, exist_ok=True)
        for stale in base.iterdir():
            try:
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
                else:
                    stale.unlink()
            except OSError:
                pass
        drop_dir = base / uuid4().hex[:8]
        drop_dir.mkdir(parents=True, exist_ok=True)
        return drop_dir

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


def show_cover_page(shell) -> bool:
    """Run the start dialog before the (still hidden) main window is shown.

    Returns True if the GUI should open: a dataset was loaded (or is already
    loaded). Returns False when the user closed the dialog, meaning the app
    should exit without showing the main window.
    """
    meta = getattr(shell, "meta_widget", None)
    io_widget = getattr(meta, "io_widget", None)
    if io_widget is None:
        return True
    app_state = io_widget.app_state
    if getattr(app_state, "ready", False) or getattr(app_state, "dt", None) is not None:
        return True
    page = CoverPage(shell, io_widget)
    shell._cover_page = page
    # Size from the screen (the shell is still hidden, its geometry pending);
    # wide enough that the custom-loader path fields are readable.
    screen = QApplication.primaryScreen().availableGeometry()
    page.resize(int(screen.width() * 0.85), int(screen.height() * 0.75))
    return page.exec_() == QDialog.Accepted
