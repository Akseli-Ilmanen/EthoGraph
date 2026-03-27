"""Widget for manually pairing video camera files with pose files.

The two folder panels (video / pose) show unique file *stems* after stripping
any trial-specific part from each filename.  The user aligns rows so that
row N in the video column corresponds to row N in the pose column.

Trial-prefix detection
----------------------
Given ``dt.trials = [1, 2, 3]`` and files such as ``leftcam_1.mp4``,
``leftcam_2.mp4``, … the function :func:`get_trial_stem` strips the trial
suffix/prefix to return ``"leftcam"``.  The result is de-duplicated so only
one representative stem per camera appears in the list.

Integration
-----------
Open as a dialog from widgets_io.py next to the pose-folder selector::

    dialog = PoseVideoMatcherDialog(
        video_folder=app_state.video_folder,
        pose_folder=app_state.pose_folder,
        trial_ids=app_state.trials,
        parent=self,
    )
    if dialog.exec_():
        app_state.camera_pose_stem_map = dict(dialog.get_mapping())
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.utils.validation import POSE_EXTENSIONS, VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Trial-stem utilities
# ---------------------------------------------------------------------------

def get_trial_stem(filename: str, trial_ids: list) -> str:
    """Strip the trial-specific substring from a filename stem.

    Tries common separator patterns (``_``, ``-``, ``.``) and both suffix
    and prefix positions.  Falls back to the bare stem if nothing matches.

    Examples
    --------
    >>> get_trial_stem("leftcam_1.mp4", [1, 2, 3])
    'leftcam'
    >>> get_trial_stem("trial2_rightpose.h5", [1, 2, 3])
    'rightpose'
    """
    stem = Path(filename).stem
    for trial_id in trial_ids:
        s = str(trial_id)
        for sep in ("_", "-", "."):
            # Suffix: leftcam_1
            candidate = stem[: -(len(sep) + len(s))]
            if stem.endswith(f"{sep}{s}") and candidate:
                return candidate.rstrip("_-.")
            # Prefix: 1_leftcam
            if stem.startswith(f"{s}{sep}"):
                return stem[len(s) + len(sep) :].lstrip("_-.")
        # Bare numeric suffix without separator: leftcam1
        if s.isdigit() and stem.endswith(s) and len(stem) > len(s):
            candidate = stem[: -len(s)].rstrip("_-.")
            if candidate:
                return candidate
    return stem


def get_unique_stems(
    folder: str,
    trial_ids: list,
    extensions: set[str],
) -> list[str]:
    """Return sorted unique stems from *folder* after stripping trial IDs."""
    if not folder or not os.path.isdir(folder):
        return []
    stems: set[str] = set()
    for fname in os.listdir(folder):
        if Path(fname).suffix.lower() in extensions:
            stems.add(get_trial_stem(fname, trial_ids))
    return sorted(stems)



# ---------------------------------------------------------------------------
# Reorderable list widget
# ---------------------------------------------------------------------------

class ReorderableList(QWidget):
    """A QListWidget with Up / Down buttons and keyboard-drag support."""

    order_changed = Signal()

    def __init__(self, title: str, editable: bool = True, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel(f"<b>{title}</b>"))

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        if editable:
            self._list.setDragDropMode(QAbstractItemView.InternalMove)
            self._list.model().rowsMoved.connect(self.order_changed)
        else:
            self._list.setDragDropMode(QAbstractItemView.NoDragDrop)
        layout.addWidget(self._list)

        if editable:
            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(0, 0, 0, 0)
            up_btn = QPushButton("↑ Up")
            down_btn = QPushButton("↓ Down")
            up_btn.setFixedHeight(22)
            down_btn.setFixedHeight(22)
            up_btn.clicked.connect(self._move_up)
            down_btn.clicked.connect(self._move_down)
            btn_row.addWidget(up_btn)
            btn_row.addWidget(down_btn)
            layout.addLayout(btn_row)

    def set_items(self, items: list[str]) -> None:
        self._list.clear()
        for item in items:
            self._list.addItem(QListWidgetItem(item))

    def get_items(self) -> list[str]:
        return [self._list.item(i).text() for i in range(self._list.count())]

    def _move_up(self) -> None:
        row = self._list.currentRow()
        if row > 0:
            item = self._list.takeItem(row)
            self._list.insertItem(row - 1, item)
            self._list.setCurrentRow(row - 1)
            self.order_changed.emit()

    def _move_down(self) -> None:
        row = self._list.currentRow()
        if row < self._list.count() - 1:
            item = self._list.takeItem(row)
            self._list.insertItem(row + 1, item)
            self._list.setCurrentRow(row + 1)
            self.order_changed.emit()


# ---------------------------------------------------------------------------
# Matcher widget
# ---------------------------------------------------------------------------

class PoseVideoMatcherWidget(QWidget):
    """Side-by-side video / pose stem lists for manual camera↔pose pairing.

    Row N in the video column is paired with row N in the pose column.
    Both lists support drag-drop and Up/Down buttons for reordering.

    Signals
    -------
    mapping_changed : emitted with ``list[tuple[video_stem, pose_stem]]``
        whenever the user changes the row order in either list.
    """

    mapping_changed = Signal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        hint = QLabel(
            "Reorder the video column (right) so that each row matches the "
            "pose file on the same row (left).\n"
            "This ensures keypoint markers are shown on top of the correct video.\n"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        cols = QHBoxLayout()
        self._pose_list = ReorderableList("Pose files", editable=False)
        self._video_list = ReorderableList("Video files")
        self._video_list.order_changed.connect(self._on_order_changed)
        cols.addWidget(self._pose_list)

        arrow_col = QVBoxLayout()
        arrow_col.setAlignment(Qt.AlignVCenter)
        arrow_lbl = QLabel("↔")
        arrow_lbl.setAlignment(Qt.AlignCenter)
        arrow_col.addWidget(arrow_lbl)
        cols.addLayout(arrow_col)

        cols.addWidget(self._video_list)
        layout.addLayout(cols)

    def set_folders(
        self,
        video_folder: str,
        pose_folder: str,
        trial_ids: list,
    ) -> None:
        """Populate both lists from the given folders and auto-match."""
        video_stems = get_unique_stems(video_folder, trial_ids, VIDEO_EXTENSIONS)
        pose_stems = get_unique_stems(pose_folder, trial_ids, POSE_EXTENSIONS)
        self._pose_list.set_items(pose_stems)
        self._video_list.set_items(video_stems)
        self._on_order_changed()

    def set_items_direct(self, video_items: list[str], pose_items: list[str]) -> None:
        """Populate both lists directly without folder scanning."""
        self._pose_list.set_items(pose_items)
        self._video_list.set_items(video_items)
        self._on_order_changed()

    def get_mapping(self) -> list[tuple[str, str]]:
        """Return current ``[(video_stem, pose_stem), …]`` pairs."""
        return list(zip(self._video_list.get_items(), self._pose_list.get_items()))

    def _on_order_changed(self) -> None:
        self.mapping_changed.emit(self.get_mapping())



# ---------------------------------------------------------------------------
# Convenience dialog
# ---------------------------------------------------------------------------

class PoseVideoMatcherDialog(QDialog):
    """Wraps PoseVideoMatcherWidget in a modal dialog with OK / Cancel.

    When ``nwb_registry`` is provided the pose column is populated from NWB
    processing module entries instead of scanning for pose files on disk.
    """

    def __init__(
        self,
        video_folder: str,
        pose_folder: str,
        trial_ids: list,
        parent: QWidget | None = None,
        nwb_registry: dict[str, dict[str, Any]] | None = None,
        pose_items: list = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Match video cameras to pose files")
        self.setMinimumWidth(520)
        self.setMinimumHeight(400)

        layout = QVBoxLayout(self)
        self._matcher = PoseVideoMatcherWidget()

        if nwb_registry is not None:
            video_stems = get_unique_stems(video_folder, trial_ids, VIDEO_EXTENSIONS)
            self._matcher.set_items_direct(video_stems, pose_items)
        else:
            self._matcher.set_folders(video_folder, pose_folder, trial_ids)

        layout.addWidget(self._matcher)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_mapping(self) -> list[tuple[str, str]]:
        """Return the confirmed ``[(video_stem, pose_stem), …]`` mapping."""
        return self._matcher.get_mapping()
