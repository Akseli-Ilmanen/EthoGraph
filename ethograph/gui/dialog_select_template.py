"""Dialog for selecting a template dataset to pre-fill IO paths."""

import logging
import traceback
import webbrowser
from pathlib import Path

logger = logging.getLogger(__name__)

from qtpy.QtCore import QSize, QThread, Qt, Signal
from qtpy.QtGui import QMovie, QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
)

from ethograph.datasets import (
    DATASETS,
    DOWNLOAD_BASE,
    dataset_dir,
    get_gui_assets,
    is_dataset_downloaded,
    resolve_dataset_paths,
)
from ethograph.gui.notify import notify_dialog
from ethograph.utils.download import (
    build_alignment_nwb,
    download_assets,
    ensure_default_configs,
    write_example_configs,
)

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "examples" / "assets"

# Backward-compat re-exports used by test code
_DOWNLOAD_BASE = DOWNLOAD_BASE
TEMPLATES = list(DATASETS.values())


def _template_dir(key_or_dict) -> Path:
    """Backward-compat helper — accepts a key or a legacy template dict."""
    if isinstance(key_or_dict, str):
        return dataset_dir(key_or_dict)
    return DOWNLOAD_BASE / key_or_dict["folder"]


def _template_downloaded(key_or_dict) -> bool:
    """Backward-compat helper — accepts a key or a legacy template dict."""
    if isinstance(key_or_dict, str):
        return is_dataset_downloaded(key_or_dict)
    return is_dataset_downloaded(key_or_dict["dataset_key"])


def _resolve_template_paths(key_or_dict) -> dict:
    """Backward-compat helper — accepts a key or a legacy template dict."""
    if isinstance(key_or_dict, str):
        return resolve_dataset_paths(key_or_dict)
    return resolve_dataset_paths(key_or_dict["dataset_key"])


def _build_alignment_nwb(key_or_dict) -> None:
    """Backward-compat helper — accepts a key or a legacy template dict."""
    if isinstance(key_or_dict, str):
        build_alignment_nwb(key_or_dict)
    else:
        build_alignment_nwb(key_or_dict["dataset_key"])


class _DownloadWorker(QThread):
    """Downloads template assets in a background thread."""

    progress = Signal(int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        info = DATASETS[self._key]
        try:
            download_assets(
                release_tag=info["release_tag"],
                assets=get_gui_assets(self._key),
                dest=dataset_dir(self._key),
                on_progress=self.progress.emit,
                cancelled=lambda: self._cancelled,
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        if not self._cancelled:
            self.finished.emit()


class TemplateDialog(QDialog):
    """Popup showing template datasets as clickable cards with images."""

    _CARDS_PER_ROW = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_template = None
        self.setWindowTitle("Select Templates")

        outer = QVBoxLayout()
        outer.setSpacing(12)
        self.setLayout(outer)

        for i, key in enumerate(DATASETS):
            if i % self._CARDS_PER_ROW == 0:
                row = QHBoxLayout()
                row.setSpacing(12)
                outer.addLayout(row)
            card = self._create_card(key)
            row.addWidget(card)

    def _create_card(self, key: str) -> QFrame:
        ds = DATASETS[key]
        card = QFrame()
        card.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        card.setCursor(Qt.PointingHandCursor)
        card.setStyleSheet(
            "QFrame:hover { background-color: palette(midlight); }"
        )

        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(8, 8, 8, 8)
        card.setLayout(card_layout)

        image_label = QLabel()
        image_path = _ASSETS_DIR / ds["image"]
        if image_path.exists():
            if image_path.suffix.lower() == ".gif":
                movie = QMovie(str(image_path))
                movie.setScaledSize(QSize(220, 160))
                image_label.setMovie(movie)
                movie.start()
            else:
                pixmap = QPixmap(str(image_path))
                image_label.setPixmap(
                    pixmap.scaled(220, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        else:
            image_label.setText("(image not found)")
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setFixedSize(220, 160)
        card_layout.addWidget(image_label, alignment=Qt.AlignCenter)

        text_label = QLabel(ds["name"])
        text_label.setWordWrap(True)
        text_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(text_label)

        link_url = ds.get("paper_url") or ds.get("dataset_url")
        if link_url:
            link_text = "Open dataset" if ds.get("dataset_url") and not ds.get("paper_url") else "Open paper"
            link = QPushButton(link_text)
            link.setFlat(True)
            link.setCursor(Qt.PointingHandCursor)
            link.setStyleSheet("color: palette(link); text-decoration: underline;")
            link.clicked.connect(lambda _checked, u=link_url: webbrowser.open(u))
            card_layout.addWidget(link, alignment=Qt.AlignCenter)

        status = QLabel()
        status.setAlignment(Qt.AlignCenter)
        if is_dataset_downloaded(key):
            status.setText("Downloaded")
            status.setStyleSheet("color: green; font-weight: bold;")
        else:
            status.setText(f"Click to download (~{ds['size_mb']} MB)")
            status.setStyleSheet("color: gray;")
        card_layout.addWidget(status)

        card.mousePressEvent = lambda event, k=key: self._on_card_clicked(k)
        return card

    def _on_card_clicked(self, key: str):
        if is_dataset_downloaded(key):
            self._finalize(key)
            return
        self._download_and_select(key)

    def _finalize(self, key: str):
        ds = DATASETS[key]
        if ds.get("nc_filename") is None and ds.get("audio_file"):
            self._generate_nc_from_audio(key)
            return
        ensure_default_configs()
        write_example_configs(key, dataset_dir(key))
        try:
            build_alignment_nwb(key)
        except Exception:
            logger.warning("Failed to build alignment NWB", exc_info=True)
        self.selected_template = resolve_dataset_paths(key)
        self.accept()

    def _generate_nc_from_audio(self, key: str):
        ds = DATASETS[key]
        dest = dataset_dir(key)
        audio_path = str(dest / ds["audio_file"])
        nc_path = str(dest / (Path(ds["audio_file"]).stem + ".nc"))

        if not Path(nc_path).exists():
            try:
                from ethograph.io.data_loader import wizard_single_from_audio
                from ethograph.utils.audio import get_audio_sr

                audio_sr = get_audio_sr(audio_path)
                dt = wizard_single_from_audio(
                    video_path=None, fps=30,
                    audio_path=audio_path, audio_sr=audio_sr,
                )
                dt.to_netcdf(nc_path)
            except Exception as e:
                traceback.print_exc()
                notify_dialog(f"Failed to generate .nc from audio:\n{e}", "error", "Error", self)
                return

        resolved = resolve_dataset_paths(key)
        resolved["nc_file_path"] = nc_path
        resolved["audio_folder"] = str(dest)
        logger.debug("Canary template resolved paths: %s", resolved)
        self.selected_template = resolved
        self.accept()

    def _download_and_select(self, key: str):
        assets = get_gui_assets(key)
        progress = QProgressDialog(
            "Downloading example data...", "Cancel", 0, len(assets), self
        )
        progress.setWindowTitle("Downloading")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = _DownloadWorker(key)

        def on_progress(count, name):
            if not progress.wasCanceled():
                progress.setValue(count)
                progress.setLabelText(f"Downloading {name}...")

        def on_finished():
            progress.close()
            worker.deleteLater()
            self._finalize(key)

        def on_error(msg):
            progress.close()
            worker.deleteLater()
            notify_dialog(msg, "warning", "Download Error", self)

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        progress.canceled.connect(worker.cancel)

        worker.start()
