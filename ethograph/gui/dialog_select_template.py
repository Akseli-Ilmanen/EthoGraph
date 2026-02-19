"""Dialog for selecting a template dataset to pre-fill IO paths."""

import webbrowser
from pathlib import Path

from qtpy.QtCore import QSize, QThread, Qt, Signal
from qtpy.QtGui import QMovie, QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
)

from ethograph.utils.download import (
    EXAMPLE_DATASETS,
    download_assets,
    is_downloaded,
)

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "tutorials" / "assets"
_DOWNLOAD_BASE = Path.home() / ".ethograph" / "example_data"

TEMPLATES = [
    {
        "name": "Moll et al., 2025 — Tool-using crows",
        "image": "moll1.png",
        "paper_url": "https://doi.org/10.1016/j.cub.2025.08.033",
        "dataset_key": "moll2025",
        "folder": "Moll2025",
        "nc_filename": "Trial_data.nc",
        "has_video": True,
        "has_audio": False,
        "has_pose": True,
        "import_labels": True,
    },
    {
        "name": "Rüttimann et al., 2025 — Zebra finches in BirdPark",
        "image": "birdpark0.png",
        "paper_url": "https://doi.org/10.7717/peerj.20203",
        "dataset_key": "birdpark",
        "folder": "BirdPark",
        "nc_filename": "copExpBP08_trim.nc",
        "has_video": True,
        "has_audio": True,
        "has_pose": False,
    },
    {
        "name": "Philodoptera — Motor control of sound production in crickets",
        "image": "cricket0.png",
        "paper_url": "",
        "dataset_key": "philodoptera",
        "folder": "Philodoptera",
        "nc_filename": "philodoptera.nc",
        "has_video": True,
        "has_audio": True,
        "has_pose": True,
    },
    {
        "name": "Reiske et al., 2025 — Mouse Lockbox",
        "image": "lockbox2.gif",
        "paper_url": "https://arxiv.org/abs/2505.15408",
        "dataset_key": "lockbox",
        "folder": "Lockbox",
        "nc_filename": "lockbox.nc",
        "has_video": True,
        "has_audio": False,
        "has_pose": True,
    },
]


def _template_dir(template: dict) -> Path:
    return _DOWNLOAD_BASE / template["folder"]


def _template_downloaded(template: dict) -> bool:
    return is_downloaded(template["dataset_key"], _template_dir(template))


def _resolve_template_paths(template: dict) -> dict:
    dest = str(_template_dir(template))
    nc = str(_template_dir(template) / template["nc_filename"])
    return {
        "name": template["name"],
        "dataset_key": template["dataset_key"],
        "nc_file_path": nc,
        "video_folder": dest if template.get("has_video") else "",
        "audio_folder": dest if template.get("has_audio") else "",
        "pose_folder": dest if template.get("has_pose") else "",
        "import_labels": template.get("import_labels", False),
    }


class _DownloadWorker(QThread):
    """Downloads template assets in a background thread."""

    progress = Signal(int, str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, template: dict):
        super().__init__()
        self._template = template
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        info = EXAMPLE_DATASETS[self._template["dataset_key"]]
        try:
            download_assets(
                release_tag=info["release_tag"],
                assets=info["assets_gui"],
                dest=_template_dir(self._template),
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_template = None
        self.setWindowTitle("Select Template Dataset")

        layout = QHBoxLayout()
        layout.setSpacing(12)
        self.setLayout(layout)

        for template in TEMPLATES:
            card = self._create_card(template)
            layout.addWidget(card)

    def _create_card(self, template: dict) -> QFrame:
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
        image_path = _ASSETS_DIR / template["image"]
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

        text_label = QLabel(template["name"])
        text_label.setWordWrap(True)
        text_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(text_label)

        if template.get("paper_url"):
            link = QPushButton("Open paper")
            link.setFlat(True)
            link.setCursor(Qt.PointingHandCursor)
            link.setStyleSheet("color: palette(link); text-decoration: underline;")
            url = template["paper_url"]
            link.clicked.connect(lambda _checked, u=url: webbrowser.open(u))
            card_layout.addWidget(link, alignment=Qt.AlignCenter)

        status = QLabel()
        status.setAlignment(Qt.AlignCenter)
        if _template_downloaded(template):
            status.setText("Downloaded")
            status.setStyleSheet("color: green; font-weight: bold;")
        else:
            size_mb = EXAMPLE_DATASETS[template["dataset_key"]]["size_mb"]
            status.setText(f"Click to download (~{size_mb} MB)")
            status.setStyleSheet("color: gray;")
        card_layout.addWidget(status)

        card.mousePressEvent = lambda event, t=template: self._on_card_clicked(t)
        return card

    def _on_card_clicked(self, template: dict):
        if _template_downloaded(template):
            self.selected_template = _resolve_template_paths(template)
            self.accept()
            return
        self._download_and_select(template)

    def _download_and_select(self, template: dict):
        info = EXAMPLE_DATASETS[template["dataset_key"]]
        assets = info["assets_gui"]
        progress = QProgressDialog(
            "Downloading example data...", "Cancel", 0, len(assets), self
        )
        progress.setWindowTitle("Downloading")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = _DownloadWorker(template)

        def on_progress(count, name):
            if not progress.wasCanceled():
                progress.setValue(count)
                progress.setLabelText(f"Downloading {name}...")

        def on_finished():
            progress.close()
            worker.deleteLater()
            self.selected_template = _resolve_template_paths(template)
            self.accept()

        def on_error(msg):
            progress.close()
            worker.deleteLater()
            QMessageBox.warning(self, "Download Error", msg)

        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        progress.canceled.connect(worker.cancel)

        worker.start()
