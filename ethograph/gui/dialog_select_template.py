"""Dialog for selecting a template dataset to pre-fill IO paths."""

import webbrowser
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "tutorials" / "assets"

TEMPLATES = [
    {
        "name": "Moll et al., 2025 — Tool-using crows",
        "image": "moll1.png",
        "paper_url": "https://doi.org/10.1016/j.cub.2025.08.033",
        "nc_file_path": r"D:\Akseli\Code\ethograph\data\Moll2025\Trial_data.nc",
        "video_folder": r"D:\Akseli\Code\ethograph\data\Moll2025",
        "audio_folder": "",
        "pose_folder": r"D:\Akseli\Code\ethograph\data\Moll2025",
        "import_labels": True,
    },
    {
        "name": "Rüttimann et al., 2025 — Zebra finches in BirdPark",
        "image": "birdpark0.png",
        "paper_url": "https://doi.org/10.7717/peerj.20203",
        "nc_file_path": "",
        "video_folder": "",
        "audio_folder": "",
        "pose_folder": "",
    },
    {
        "name": "Philodoptera — Motor control of sound production in crickets",
        "image": "cricket0.png",
        "paper_url": "",
        "nc_file_path": "",
        "video_folder": "",
        "audio_folder": "",
        "pose_folder": "",
    },
]


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

        card.mousePressEvent = lambda event, t=template: self._on_card_clicked(t)
        return card

    def _on_card_clicked(self, template: dict):
        self.selected_template = template
        self.accept()
