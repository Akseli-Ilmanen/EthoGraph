"""Keypoint labelling dialog: label a few frames, let a tracker fill the rest.

Three tabs, one per stage of the work — **Keypoints** (who is labelled and with
which keypoints), **Label** (the modes, the points table, frame suggestions) and
**Fill and export**. One column holding every group at once grew taller than a
screen; the split is by stage, so nothing a stage needs sits on another tab.

Non-modal, because the whole point is to keep navigating frames with the normal
playhead while labelling. It owns a :class:`~ethograph.gui.pose_annotate.KeypointStore`,
attaches a :class:`~ethograph.gui.pose_edit_mixin.KeypointLabelMode` to the
primary camera view, and renders fill results back through the ordinary pose
overlay so filled keypoints are indistinguishable from imported predictions.

The panel is hierarchical, following SLEAP: one keypoint schema (the skeleton)
shared by every individual, and one branch per individual whose children are
that individual's keypoints on the current frame. Clicking a branch selects the
individual to label; clicking a leaf selects the keypoint. Labelling one
individual is simply the case where there is one branch, and no branches at all
is a legal state too — the tree is empty until an individual is added.

Unticking "Individuals share the same keypoints" gives each individual its own
subset of the schema, for animals that cannot be labelled symmetrically. The
keypoint buttons then edit the selected individual's set instead of the schema
everyone shares.

The canvas has two modes, armed by their buttons (pressing the active one
again turns labelling off), after napari-deeplabcut: **Sequential** labels every
keypoint on one frame and never navigates by itself, **Loop** sweeps one
keypoint across frames. What Loop does after each click is the "Between clicks"
dropdown — stay put, step a frame, or jump to the next suggested frame — rather
than being inferred from whether a suggestion list happens to exist. Editing
needs no mode — clicking an existing point always drags it, ``Backspace``
deletes the selected point and ``Ctrl+Z`` undoes. Clicking a *filled* point pins
it as a label, which is how a prediction is accepted or corrected. See
:mod:`~ethograph.gui.pose_edit_mixin`.

Navigation splits by modifier: plain ``←``/``→`` step one frame (the main
window's own binding, untouched), ``Shift+←``/``Shift+→`` jump between the
suggested frames — the ones worth labelling, which is what you actually move
through while annotating.

The points table has one row per ``(frame, individual)`` and an ``x``/``y``
column pair per keypoint, so everything on a frame is visible at once. The
keypoint name is painted once *above* its pair by :class:`PairedHeaderView`.
Clicking a cell seeks the playhead to that frame and makes the clicked keypoint
active; conversely the playhead's own row is selected and scrolled to. Rows are
multi-selectable and right-click deletes their labels — or pins their
predictions.

Human labels and filled predictions live in the same table. A ``Source`` column
says which a row is (one hand-placed point makes the row the user's), predicted
coordinates are dimmed, and the ``Frame``, ``Individual`` and ``Source`` headers
carry the funnel filters of :mod:`~ethograph.gui.table_filter` — so "show me
only what I labelled", or "only the frames the fill invented", is one click.
Before a fill the rows are the labelled frames; afterwards they are every frame,
which is why the table is a virtual model (:class:`PointTableModel`) rather than
a widget grid.

"Load into the GUI" turns the keypoints — and optionally velocity, speed and
acceleration derived from them — into ordinary plottable features, so a fill can
be inspected without writing and reloading a file. The keypoints are themselves a
complete dataset: the time axis is the video's own frame grid (frame index ×
1/fps) and the keypoint/individual names come from this dialog, so it works with
no dataset loaded at all — one is created.

Anchors are project data, not settings: they are persisted to a sidecar next to
the video (``<video>.keypoints.json``), never into ``app_state``. A fill is not
persisted at all, and never feeds the next fill — see
:mod:`~ethograph.gui.pose_annotate`.
"""

from __future__ import annotations

import html
import logging
from contextlib import contextmanager

import numpy as np
from qtpy.QtCore import QAbstractTableModel, QEvent, QModelIndex, QRect, Qt
from qtpy.QtGui import QBrush, QColor, QPalette, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableView,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.dialog_busy_progress import BusyProgressDialog
from ethograph.gui.notify import notify
from ethograph.gui.pose_annotate import (
    DEFAULT_INDIVIDUAL,
    KINEMATICS,
    RECOMMENDED_ANCHORS,
    KeypointStore,
    KeypointStoreError,
    sidecar_path,
    store_to_dlc_h5,
    store_to_kinematics,
    store_to_movement_ds,
)
from ethograph.gui.pose_edit_mixin import (
    LOOP_MODE,
    SEQUENTIAL_MODE,
    KeypointLabelMode,
    glyph_for_individual,
    keypoint_colors,
)
from ethograph.gui.pose_fill import VideoFrameSource, available_backends, build_backend
from ethograph.gui.pose_render import movement_ds_to_pose_render
from ethograph.gui.pose_suggest import suggest_frames
from ethograph.gui.table_filter import (
    SORT_ROLE,
    CategoryFilterDialog,
    FilterHeaderView,
    MultiColumnFilterProxy,
    NumericFilterDialog,
)

logger = logging.getLogger(__name__)


@contextmanager
def _blocked(widget):
    """Mute a widget's signals — these combos are readouts as well as inputs."""
    previous = widget.blockSignals(True)
    try:
        yield widget
    finally:
        widget.blockSignals(previous)


#: Longest image side fed to the tracking backends. Downscaling is the single
#: biggest CPU speedup and costs almost nothing in accuracy at this anchor density.
MAX_SIDE = 512

_LABELLED_MARK = "●"
_UNLABELLED_MARK = "·"

#: Keeps the points table from eating the dialog — it scrolls past this.
#: Sized so the dialog's default height shows a useful run of frames rather
#: than capping the table early and leaving the extra space blank.
TABLE_MAX_HEIGHT = 380

#: Columns before the per-keypoint ``x``/``y`` pairs.
_FIXED_COLUMNS = ("Frame", "Individual", "Source", "Confidence")

#: Column indices of :data:`_FIXED_COLUMNS`, since they read badly as bare
#: numbers in the filter wiring.
FRAME_COLUMN, INDIVIDUAL_COLUMN, SOURCE_COLUMN, CONFIDENCE_COLUMN = range(len(_FIXED_COLUMNS))

#: Provenance of a ``(frame, individual)`` row — see ``KeypointStore.is_human``.
HUMAN_SOURCE = "Human"
FILL_SOURCE = "Fill"

_FIXED_COLUMN_TOOLTIPS = (
    "Video frame. Click a cell to jump the playhead there.",
    "Which individual this row's points belong to.",
    f"{HUMAN_SOURCE} — you placed or corrected at least one point on this row.\n"
    f"{FILL_SOURCE} — every point here came from the fill backend.\n\n"
    "Filling always rebuilds from the human points alone, so a filled point\n"
    "only survives a re-fill once you pin it (click it, or use the right-click\n"
    "menu here).",
    "How much the fill trusts this row, averaged over its keypoints.\n"
    "1.00 means you labelled it by hand; low means the fill was lost.\n\n"
    "Spline: decays with distance from the nearest labelled frame.\n"
    "Optical flow and CoTracker3: each gap is tracked twice, forwards from\n"
    "the label on its left and backwards from the one on its right — the\n"
    "score falls as the two tracks disagree, and drops to zero where either\n"
    "tracker reports the point as lost.\n\n"
    "Filter this column to find the frames worth correcting; the 'Lowest\n"
    "fill confidence' suggestion method ranks by the same number.",
)

#: Thumbnails are enough to score frames for suggestions — DeepLabCut clusters
#: at ~30 px wide. Far cheaper to decode than the fill backends' input.
SUGGEST_MAX_SIDE = 64

#: Smallest share of a video the suggestion spin box offers. A tenth of a
#: percent is 60 frames of a 60k-frame recording — already more than anyone
#: labels by hand — and a floor keeps the box from resolving to zero frames.
MIN_SUGGEST_PERCENT = 0.1

#: Each method names the workflow it suits, because they are NOT
#: interchangeable — see the module docstring of ``pose_suggest``. Ordered by
#: **when in the workflow they apply**: the first three need nothing but the
#: video and so are where a new project starts; ``uncertain`` comes last
#: because it can only rank what a fill has already scored.
_SUGGEST_METHODS = (
    (
        "uniform",
        "Evenly spaced  (before fill)",
        "Equally spaced frames, no video scan — instant.\nHard to beat on a short clip of one behaviour.",
    ),
    (
        "motion",
        "Biggest pixel change  (before fill)",
        "Frames whose pixels differ most from the frame before —\n"
        "where the animal moves fastest. Fast motion is where optical\n"
        "flow loses the point and where a spline cuts the corner.",
    ),
    (
        "diverse",
        "Most different frames  (before fill)",
        "Groups the frames by how they look and takes one per group\n"
        "(DeepLabCut's k-means), so the picks are as unlike each\n"
        "other as possible.",
    ),
    (
        "uncertain",
        "Lowest fill confidence  (after fill)",
        "Frames the last fill scored lowest, where tracking forwards\n"
        "and backwards disagreed most — the ones worth correcting.\n"
        "The backends are frozen, so extra labels reset drift rather\n"
        "than teach anything.\n\n"
        "Needs a fill to have run.",
    ),
)

#: What Loop mode does after each click — the "Between clicks" dropdown.
AFTER_CLICK_FRAME = "frame"
AFTER_CLICK_SUGGESTION = "suggestion"
AFTER_CLICK_STAY = "stay"

_AFTER_CLICK_CHOICES = (
    (
        AFTER_CLICK_FRAME,
        "Jump +1 frame",
        "Step to the very next frame after each click — a dense sweep of one\nkeypoint through a short clip.",
    ),
    (
        AFTER_CLICK_SUGGESTION,
        "Jump to next suggested frame",
        "Follow the list from 'Which frames to label' below, so each click lands\n"
        "on a frame that was chosen deliberately rather than on the near-identical\n"
        "neighbour. Suggest some frames first.",
    ),
    (
        AFTER_CLICK_STAY,
        "Stay on this frame",
        "Never move the playhead. Navigate yourself with ← / → (single frames)\n"
        "and Shift+← / Shift+→ (suggested frames).",
    ),
)


class PairedHeaderView(FilterHeaderView):
    """Two-row horizontal header: a group name spanning each ``x``/``y`` pair.

    ``QHeaderView`` has no multi-level support, and repeating the keypoint name
    in both column labels ("beak x", "beak y") is what made the table so wide.
    Each of a pair's two sections therefore paints the *same* group name across
    their union rect and only ``x`` or ``y`` beneath its own half — the two
    halves join into one centred label, and painting it twice is idempotent, so
    it survives a partial repaint of either section.

    The first *fixed_columns* sections are ungrouped, painted normally, and are
    the ones that carry the inherited filter funnels.
    """

    #: Slack around a group name, so ResizeToContents never elides it.
    PADDING = 12

    def __init__(self, fixed_columns: int, parent=None):
        super().__init__(parent=parent)
        self._fixed = int(fixed_columns)
        self._groups: list[str] = []
        self._brushes: list[QBrush] = []
        self.setSectionsClickable(True)
        self.setHighlightSections(False)

    def groups(self) -> list[str]:
        """The group name over each column pair, left to right."""
        return list(self._groups)

    def set_groups(self, groups: list[str], brushes: list[QBrush]) -> None:
        self._groups = list(groups)
        self._brushes = list(brushes)
        self.viewport().update()

    def _group_index(self, section: int) -> int | None:
        """Which group a section belongs to, or ``None`` for a fixed column."""
        if section < self._fixed:
            return None
        index = (section - self._fixed) // 2
        return index if 0 <= index < len(self._groups) else None

    def sectionSizeFromContents(self, section: int):
        size = super().sectionSizeFromContents(section)
        # The funnel is drawn over the section, so its zone has to be paid for
        # in width or ResizeToContents elides the label underneath it.
        if section in self.filterable:
            size.setWidth(size.width() + self.FILTER_ZONE_W)
        if not self._groups:
            return size
        size.setHeight(size.height() * 2)
        index = self._group_index(section)
        if index is not None:
            half = self.fontMetrics().horizontalAdvance(self._groups[index]) // 2
            size.setWidth(max(size.width(), half + self.PADDING))
        return size

    def paintSection(self, painter, rect, section: int) -> None:
        index = self._group_index(section)
        if index is None:
            super().paintSection(painter, rect, section)  # draws the filter funnel too
            return
        painter.save()
        super().paintSection(painter, rect, section)  # background; the label is empty
        half = rect.height() // 2
        first = self._fixed + 2 * index
        span = QRect(
            self.sectionViewportPosition(first),
            rect.top(),
            self.sectionSize(first) + self.sectionSize(first + 1),
            half,
        )
        painter.setPen(QPen(self._brushes[index].color()))
        painter.drawText(span, Qt.AlignCenter, self._groups[index])
        painter.setPen(QPen(self.palette().color(QPalette.ButtonText)))
        axis = "x" if section == first else "y"
        painter.drawText(QRect(rect.left(), rect.top() + half, rect.width(), half), Qt.AlignCenter, axis)
        painter.restore()


class PointTableModel(QAbstractTableModel):
    """A :class:`KeypointStore` as rows of ``(frame, individual)``.

    Columns are ``Frame | Individual | Source`` then an ``x``/``y`` pair per
    keypoint. Values are read from the store on demand rather than copied into
    cells, for two reasons: once a fill exists there is a row for *every* frame
    of the video, which no item-based table can hold; and a view that reads the
    store cannot disagree with it, which the previous diffing item table could.

    Provenance is shown twice over. The ``Source`` cell states whether the row
    is the user's or the backend's — one human point anywhere in the row is
    enough, since correcting a single keypoint is what makes a frame yours — and
    the individual ``x``/``y`` pairs are dimmed wherever the coordinate came
    from the fill, so a mixed row still reads correctly.
    """

    def __init__(self, store: KeypointStore, parent=None):
        super().__init__(parent)
        self._store = store
        self._rows: list[tuple[int, str]] = []
        self._columns: list[str] = []
        self._row_of: dict[tuple[int, str], int] = {}
        #: ``frame -> (first row, last row)``; rows of a frame are contiguous, so
        #: a drag can repaint one frame instead of the whole video.
        self._frame_rows: dict[int, tuple[int, int]] = {}
        #: One frame's ``(positions, human mask, confidence)``, so a row costs
        #: one lookup rather than one per cell.
        self._cache: tuple[int, np.ndarray, np.ndarray, np.ndarray | None] | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @property
    def rows(self) -> list[tuple[int, str]]:
        return self._rows

    @property
    def keypoint_columns(self) -> list[str]:
        return self._columns

    def set_layout(self, rows: list[tuple[int, str]], columns: list[str]) -> None:
        """Replace the row keys and the keypoints carrying an ``x``/``y`` pair."""
        self.beginResetModel()
        self._rows = list(rows)
        self._columns = list(columns)
        self._row_of = {key: index for index, key in enumerate(self._rows)}
        self._frame_rows = {}
        for index, (frame, _individual) in enumerate(self._rows):
            first, _last = self._frame_rows.get(frame, (index, index))
            self._frame_rows[frame] = (first, index)
        self._cache = None
        self.endResetModel()

    def row_of(self, key: tuple[int, str]) -> int | None:
        return self._row_of.get(key)

    def key_at(self, row: int) -> tuple[int, str] | None:
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def keypoint_at(self, column: int) -> tuple[str, str] | None:
        """``(keypoint, axis)`` a column belongs to, or ``None`` for a fixed one."""
        index = (column - len(_FIXED_COLUMNS)) // 2
        if not 0 <= index < len(self._columns):
            return None
        return self._columns[index], "xy"[(column - len(_FIXED_COLUMNS)) % 2]

    def refresh_frame(self, frame: int | None) -> None:
        """Repaint one frame's rows — what a drag or a single placement changes."""
        self._cache = None
        span = self._frame_rows.get(int(frame)) if frame is not None else None
        if span is None:
            return
        first, last = span
        self.dataChanged.emit(self.index(first, 0), self.index(last, self.columnCount() - 1))

    def refresh_all(self) -> None:
        """Repaint everything — after a fill, which rewrites every row."""
        self._cache = None
        if self._rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._rows) - 1, self.columnCount() - 1))

    # ------------------------------------------------------------------
    # Qt model interface
    # ------------------------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(_FIXED_COLUMNS) + 2 * len(self._columns)

    def _frame_data(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        return self._frame_cache(frame)[:2]

    def _frame_cache(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self._cache is None or self._cache[0] != frame:
            confidence = self._store.confidence
            self._cache = (
                frame,
                self._store.positions(frame),
                self._store.human_mask(frame),
                confidence[frame] if confidence is not None and 0 <= frame < len(confidence) else None,
            )
        return self._cache[1:]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        frame, individual = self._rows[index.row()]
        column = index.column()

        if role == Qt.TextAlignmentRole:
            numeric = column == CONFIDENCE_COLUMN or column >= len(_FIXED_COLUMNS)
            return int(Qt.AlignRight | Qt.AlignVCenter) if numeric else None

        if column == FRAME_COLUMN:
            if role == Qt.DisplayRole:
                return str(frame)
            if role == SORT_ROLE:
                return float(frame)
            return None
        if column == INDIVIDUAL_COLUMN:
            return individual if role == Qt.DisplayRole else None
        if column == SOURCE_COLUMN:
            return self._source_data(frame, individual, role)
        if column == CONFIDENCE_COLUMN:
            return self._confidence_data(frame, individual, role)
        return self._point_data(frame, individual, column, role)

    def _confidence_data(self, frame: int, individual: str, role):
        """How much the fill trusts this row, averaged over its keypoints.

        ``nanmean``, matching ``pose_suggest.frame_confidence`` — a keypoint an
        asymmetric schema leaves out sits at NaN, and taking the minimum would
        pin every row of that individual to it.
        """
        if role not in (Qt.DisplayRole, Qt.ForegroundRole, Qt.ToolTipRole, SORT_ROLE):
            return None
        positions, human, confidence = self._frame_cache(frame)
        index = self._store.individual_index(individual)
        if confidence is None:
            return None
        # A point placed *after* the fill still carries the fill's old score in
        # the store — the array is a snapshot. Show it as the 1.0 the next fill
        # will write, rather than a number the user has already superseded.
        scores = np.where(human[index], 1.0, confidence[index])
        if np.all(np.isnan(scores)):
            return None
        if role == Qt.ForegroundRole:
            return None if human[index].any() else self._dim_brush()
        if role == Qt.ToolTipRole:
            worst = int(np.nanargmin(scores))
            return f"Lowest: {self._store.keypoint_names[worst]} {scores[worst]:.2f}"
        value = float(np.nanmean(scores))
        return f"{value:.2f}" if role == Qt.DisplayRole else value

    def _source_data(self, frame: int, individual: str, role):
        index = self._store.individual_index(individual)
        positions, human_mask = self._frame_data(frame)
        human = human_mask[index]
        # An empty row says nothing rather than "Fill": with the predictions
        # deleted there is no source to name.
        empty = not np.any(~np.isnan(positions[index][:, 0]))
        if role == Qt.DisplayRole:
            if human.any():
                return HUMAN_SOURCE
            return "" if empty else FILL_SOURCE
        if role == Qt.ForegroundRole and not human.any():
            return self._dim_brush()
        if role == Qt.ToolTipRole:
            total = len(self._store.keypoints_for(individual))
            return f"{int(human.sum())} of {total} keypoints placed by hand on frame {frame}."
        return None

    def _point_data(self, frame: int, individual: str, column: int, role):
        if role not in (Qt.DisplayRole, Qt.ForegroundRole, SORT_ROLE):
            return None
        target = self.keypoint_at(column)
        if target is None:
            return None
        keypoint, axis = target
        i, k = self._store.individual_index(individual), self._store.keypoint_index(keypoint)
        positions, human = self._frame_data(frame)
        value = positions[i, k, "xy".index(axis)]
        if np.isnan(value):
            return None
        if role == Qt.ForegroundRole:
            return None if human[i, k] else self._dim_brush()
        return f"{value:.1f}" if role == Qt.DisplayRole else float(value)

    @staticmethod
    def _dim_brush() -> QBrush:
        """The palette's disabled text colour — a predicted value, not a label."""
        return QBrush(QApplication.palette().color(QPalette.Disabled, QPalette.Text))

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal:
            return None
        fixed = section < len(_FIXED_COLUMNS)
        if role == Qt.DisplayRole:
            # The keypoint labels stay empty: PairedHeaderView paints the name
            # over each pair itself, and repeating it in both would double it.
            return _FIXED_COLUMNS[section] if fixed else ""
        if role == Qt.ToolTipRole:
            if fixed:
                return _FIXED_COLUMN_TOOLTIPS[section]
            target = self.keypoint_at(section)
            return None if target is None else f"{target[0]} {target[1]}"
        return None


class PoseLabellingDialog(QDialog):
    """Individual/keypoint tree, canvas labelling, backend choice, fill and export."""

    def __init__(self, data_widget, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keypoint labelling")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)

        self._data_widget = data_widget
        self.app_state = data_widget.app_state
        #: The main window. Key events land there whenever the user is looking
        #: at the video rather than at this dialog, so it is a key-filter target.
        self._shell = data_widget.shell
        self._view = self._shell.video_area.primary
        self._mode: KeypointLabelMode | None = None
        #: Frames proposed by pose_suggest, and where the user is within them.
        self._suggestions: list[int] = []
        self._suggestion_index = 0
        #: What the table's rows and columns were built from — recomputing the
        #: layout on every drag is what this avoids.
        self._table_signature: tuple | None = None

        self.store = self._load_store()
        self._build_ui()
        self._rebuild_tree()
        self._refresh_point_table()

        self._install_key_filter(True)
        self.app_state.current_frame_changed.connect(self._on_frame_changed)

    # ------------------------------------------------------------------
    # Store lifecycle
    # ------------------------------------------------------------------

    def _video_path(self) -> str | None:
        return self.app_state.video_path

    def _fps(self) -> float | None:
        fps = self._view.fps or self.app_state.video_fps
        return float(fps) if fps else None

    def _n_frames(self) -> int:
        return self._view.n_frames or int(self.app_state.num_frames or 0)

    def _load_store(self) -> KeypointStore:
        """Reuse the sidecar next to the video when one exists."""
        video = self._video_path()
        n_frames = self._n_frames()
        if video:
            path = sidecar_path(video)
            if path.exists():
                # A damaged sidecar is a runtime condition, not a bug: warn and
                # start empty rather than making the dialog un-openable. The
                # file is left alone so the user can inspect or repair it.
                try:
                    store = KeypointStore.load(path)
                except (KeypointStoreError, ValueError, KeyError, OSError) as e:
                    notify(f"Could not read {path.name}: {e}. Starting with no labels.", "warning")
                else:
                    store.n_frames = n_frames or store.n_frames
                    return store
        return KeypointStore(
            keypoint_names=list(self.app_state.keypoints or []),
            n_frames=n_frames,
            individual_names=list(self.app_state.labelling_individuals or [DEFAULT_INDIVIDUAL]),
        )

    def _save_store(self) -> None:
        video = self._video_path()
        if video:
            self.store.save(sidecar_path(video))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Three tabs, in the order the work happens: schema, labelling, output.

        One column of every group at once made the dialog taller than most
        screens; the split is by *stage*, so nothing a stage needs lives on
        another tab.
        """
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._label_page = self._build_label_page()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_schema_page(), "Define keypoints")
        self.tabs.addTab(self._label_page, "Label && Edit")  # && escapes the mnemonic
        self.tabs.addTab(self._build_output_page(), "Fill and export")
        # Opening the tab IS the intent to label, so arm Sequential rather than
        # making the first click a mode choice.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs, stretch=1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        # Taller by default: the points table is the thing you read most, and a
        # short dialog turns it into a 4-row peephole. Clamped to the screen so
        # it never opens taller than the display it lands on.
        width, height = 480, 800
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, available.width() - 80)
            height = min(height, available.height() - 80)
        self.resize(width, height)

    def _build_label_page(self) -> QWidget:
        """Labelling itself: the modes, the points table and frame suggestions."""
        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(self._build_mode_controls())
        box.addWidget(self._build_table_group(), stretch=1)
        box.addWidget(self._build_suggest_group())
        return page

    def _build_output_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.addWidget(self._build_fill_group())
        box.addWidget(self._build_export_group())
        box.addStretch()
        return page

    def _build_schema_page(self) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)

        self.shared_toggle = QCheckBox("Individuals share the same keypoints")
        self.shared_toggle.setToolTip(
            "On: every individual is an instance of one schema (SLEAP's skeleton).\n"
            "Off: each individual carries its own keypoints, so one animal can be\n"
            "labelled with keypoints another does not have. The keypoint buttons\n"
            "below then edit the selected individual's set."
        )
        self.shared_toggle.setChecked(self.store.shared_keypoints)
        self.shared_toggle.toggled.connect(self._on_shared_toggled)
        box.addWidget(self.shared_toggle)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Name", "This frame"])
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setRootIsDecorated(True)
        self.tree.setMinimumHeight(180)
        self.tree.header().setStretchLastSection(False)
        self.tree.setColumnWidth(0, 220)
        self.tree.currentItemChanged.connect(self._on_tree_item_changed)
        box.addWidget(self.tree)

        individual_row = QHBoxLayout()
        add_individual = QPushButton("Add individual…")
        add_individual.setToolTip("Label a second animal with the same keypoints (SLEAP's 'add instance').")
        add_individual.clicked.connect(self._on_add_individual)
        individual_row.addWidget(add_individual)
        remove_individual = QPushButton("Remove individual")
        remove_individual.setToolTip(
            "Remove the selected individual, including the last one — labelling\n"
            "resumes once an individual is added back."
        )
        remove_individual.clicked.connect(self._on_remove_individual)
        individual_row.addWidget(remove_individual)
        box.addLayout(individual_row)

        keypoint_row = QHBoxLayout()
        self._add_keypoint_btn = QPushButton("Add keypoint…")
        self._add_keypoint_btn.clicked.connect(self._on_add_keypoint)
        keypoint_row.addWidget(self._add_keypoint_btn)
        self._remove_keypoint_btn = QPushButton("Remove keypoint")
        self._remove_keypoint_btn.clicked.connect(self._on_remove_keypoint)
        keypoint_row.addWidget(self._remove_keypoint_btn)
        box.addLayout(keypoint_row)
        self._refresh_keypoint_hints()
        return page

    def _refresh_keypoint_hints(self) -> None:
        """Say who the keypoint buttons act on — everyone, or one individual."""
        if self.store.shared_keypoints:
            scope = "the schema every individual shares"
        else:
            scope = "the selected individual only"
        self._add_keypoint_btn.setToolTip(f"Add a keypoint to {scope}.")
        self._remove_keypoint_btn.setToolTip(f"Remove the selected keypoint from {scope}.")

    def _build_mode_controls(self) -> QWidget:
        """One compact row — modes plus the target pickers — over the status chip.

        Everything shares a single row so adding the pickers costs no vertical
        space: the table below is what the tab is for.

        Editing needs no mode of its own: clicking an existing point always
        selects and drags it, ``Backspace`` deletes the selected point and
        ``Ctrl+Z`` undoes.
        """
        widget = QWidget()
        box = QVBoxLayout(widget)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.sequential_btn = QPushButton("Sequential")
        self.sequential_btn.setCheckable(True)
        self.sequential_btn.setToolTip(
            "Label every keypoint on one frame. Each click places the active\n"
            "keypoint, then moves to the next one this individual still lacks,\n"
            "filling the table left to right; the playhead never moves on its own.\n\n"
            "Click an existing point to drag it. Tab cycles keypoints, 1-9 pick\n"
            "the individual, Backspace deletes the selected point, Ctrl+Z undoes,\n"
            "Shift+drag pans. Press the button again to stop."
        )
        self.sequential_btn.clicked.connect(
            lambda checked: self.set_interaction_mode(SEQUENTIAL_MODE if checked else None)
        )
        row.addWidget(self.sequential_btn)

        self.loop_btn = QPushButton("Loop")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setToolTip(
            "Sweep ONE keypoint across frames. Each click places it and then does\n"
            "whatever 'Between clicks' says — step a frame, jump to the next\n"
            "suggested frame, or stay put and let you navigate.\n\n"
            "This is what the fill backends want: each keypoint is interpolated\n"
            "over its own anchor set, so temporal density per keypoint matters.\n"
            "Press the button again to stop."
        )
        self.loop_btn.clicked.connect(lambda checked: self.set_interaction_mode(LOOP_MODE if checked else None))
        row.addWidget(self.loop_btn)

        self.individual_combo = QComboBox()
        self.individual_combo.setToolTip("Which individual the next click labels (1-9 also select it).")
        self.individual_combo.currentIndexChanged.connect(self._on_individual_combo_changed)
        row.addWidget(self.individual_combo, stretch=1)

        # Only Loop mode holds one keypoint still, so only Loop mode has a
        # keypoint to pick; Sequential advances through them on its own.
        self.keypoint_combo = QComboBox()
        self.keypoint_combo.setToolTip("Which keypoint to sweep across frames (Tab also cycles).")
        self.keypoint_combo.currentIndexChanged.connect(self._on_keypoint_combo_changed)
        self.keypoint_combo.hide()
        row.addWidget(self.keypoint_combo, stretch=1)

        box.addLayout(row)

        # Its own row, shown only in Loop mode: the mode row is already full,
        # and this choice is the whole substance of Loop mode.
        self.after_click_row = QWidget()
        after_click = QHBoxLayout(self.after_click_row)
        after_click.setContentsMargins(0, 0, 0, 0)
        after_click.setSpacing(4)
        after_click.addWidget(QLabel("Between clicks:"))
        self.after_click_combo = QComboBox()
        for key, label, tip in _AFTER_CLICK_CHOICES:
            self.after_click_combo.addItem(label, key)
            self.after_click_combo.setItemData(self.after_click_combo.count() - 1, tip, Qt.ToolTipRole)
        self.after_click_combo.setToolTip("Where the playhead goes after each click in Loop mode.")
        after_click.addWidget(self.after_click_combo, stretch=1)
        self.after_click_row.hide()
        box.addWidget(self.after_click_row)

        # What the next click will place, in the canvas's own visual language:
        # the individual's marker glyph, and the keypoint in its marker colour.
        self.active_label = QLabel()
        self.active_label.setTextFormat(Qt.RichText)
        self.active_label.setStyleSheet(
            "QLabel { font-size: 14px; padding: 3px 8px; border-radius: 4px; background: rgba(127,127,127,0.18); }"
        )
        self.active_label.hide()

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        status_row.addWidget(self.active_label)
        status_row.addStretch()

        # Marker size is in SCREEN pixels, so it stays put when the canvas is
        # zoomed; big markers are easier to hit, small ones let you see the
        # pixel you are aiming at.
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(4, 60)
        self.point_size_spin.setValue(int(self.app_state.labelling_point_size))
        self.point_size_spin.setSuffix(" px")
        self.point_size_spin.setToolTip(
            "Size of the labelling markers, in screen pixels — constant as you\n"
            "zoom. Shrink it to see the pixel under the point you are placing."
        )
        self.point_size_spin.valueChanged.connect(self._on_point_size_changed)
        self.point_size_spin.hide()
        status_row.addWidget(self.point_size_spin)
        box.addLayout(status_row)
        return widget

    def _on_point_size_changed(self, value: int) -> None:
        self.app_state.labelling_point_size = float(value)
        if self._mode is not None:
            self._mode.set_point_size(float(value))

    # ------------------------------------------------------------------
    # Target pickers
    # ------------------------------------------------------------------

    def _refresh_target_combos(self) -> None:
        """Rebuild and sync the individual/keypoint pickers from the store.

        Signals stay blocked throughout: each combo is both an input and a
        readout, and the mode writes back whenever a number key, a Tab or a
        click on an existing point moves the target.
        """
        individuals = list(self.store.individual_names)
        wanted = self._mode.active_individual if self._mode else self._combo_individual()
        with _blocked(self.individual_combo):
            if self._combo_items(self.individual_combo) != individuals:
                self.individual_combo.clear()
                for index, name in enumerate(individuals):
                    self.individual_combo.addItem(f"{glyph_for_individual(index)}  {name}", name)
            position = self.individual_combo.findData(wanted)
            self.individual_combo.setCurrentIndex(position if position >= 0 else 0)
        self.individual_combo.setEnabled(bool(individuals))

        loop = self.interaction_mode == LOOP_MODE
        self.keypoint_combo.setVisible(loop)
        self.after_click_row.setVisible(loop)
        if not loop or self._mode is None:
            return
        keypoints = self._mode.active_keypoints
        with _blocked(self.keypoint_combo):
            if self._combo_items(self.keypoint_combo) != keypoints:
                self.keypoint_combo.clear()
                for name in keypoints:
                    self.keypoint_combo.addItem(name, name)
                    self.keypoint_combo.setItemData(
                        self.keypoint_combo.count() - 1, self._keypoint_brush(name), Qt.ForegroundRole
                    )
            position = self.keypoint_combo.findData(self._mode.active_keypoint)
            self.keypoint_combo.setCurrentIndex(position if position >= 0 else 0)

    @staticmethod
    def _combo_items(combo: QComboBox) -> list:
        return [combo.itemData(i) for i in range(combo.count())]

    def _combo_individual(self) -> str | None:
        """The picker's individual, used to seed the mode when it is armed."""
        return self.individual_combo.currentData() if self.individual_combo.count() else None

    def _on_individual_combo_changed(self, _index: int) -> None:
        name = self.individual_combo.currentData()
        if self._mode is not None and name is not None:
            self._mode.set_active_individual(name)
            self._refresh_active_label()
            self._sync_tree_selection()

    def _on_keypoint_combo_changed(self, _index: int) -> None:
        name = self.keypoint_combo.currentData()
        if self._mode is not None and name is not None:
            self._mode.set_active(name, self._mode.active_individual)
            self._refresh_active_label()
            self._sync_tree_selection()

    def _refresh_active_label(self) -> None:
        """Show the marker the next click will drop, or hide the line when idle."""
        self._refresh_target_combos()
        if self._mode is None:
            self.active_label.hide()
            self.point_size_spin.hide()
            return
        individual = self._mode.active_individual
        keypoint = self._mode.active_keypoint
        if individual is None or keypoint is None:
            self.active_label.hide()
            return

        glyph = glyph_for_individual(self.store.individual_index(individual))
        colour = self._keypoint_brush(keypoint).color().name()
        mode = "Loop" if self._mode.mode == LOOP_MODE else "Sequential"
        self.active_label.setText(
            f'<span style="color:{colour}; font-size:17px;">{glyph}</span>&nbsp;'
            f"<b>{html.escape(individual)}</b>"
            f'&nbsp;·&nbsp;<b style="color:{colour};">{html.escape(keypoint)}</b>'
            f'&nbsp;&nbsp;<span style="opacity:0.6;">— {mode}</span>'
        )
        self.active_label.show()
        self.point_size_spin.show()

    def _build_table_group(self) -> QTableView:
        """The points table — seeks the video when clicked, right-click edits.

        A view over a virtual model rather than a widget table: once a fill
        exists the row set spans the whole video, which is far more rows than
        cell widgets can be made for.
        """
        self.point_model = PointTableModel(self.store, self)
        self.point_proxy = MultiColumnFilterProxy(self)
        self.point_proxy.setSourceModel(self.point_model)

        self.point_table = QTableView()
        self.point_table.setModel(self.point_proxy)
        header = PairedHeaderView(len(_FIXED_COLUMNS), self.point_table)
        self.point_table.setHorizontalHeader(header)
        header.set_filterable({INDIVIDUAL_COLUMN, SOURCE_COLUMN}, {FRAME_COLUMN, CONFIDENCE_COLUMN})
        header.filter_requested.connect(self._on_filter_requested)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        # ResizeToContents measures rows to size a column; with a fill loaded
        # there are as many rows as frames, so cap what it looks at.
        header.setResizeContentsPrecision(50)

        self.point_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.point_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # Extended selection so a run of frames can be discarded in one go.
        self.point_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.point_table.verticalHeader().setVisible(False)
        self.point_table.setMaximumHeight(TABLE_MAX_HEIGHT)
        self.point_table.setToolTip(
            "Click a cell to jump to that frame and make its keypoint active.\n"
            "Right-click to delete — or pin — the selected rows' points.\n"
            "The funnels in the Frame, Individual and Source headers filter the rows."
        )
        # clicked, not the selection signal: the table also selects itself when
        # the playhead moves, and that must not seek back.
        self.point_table.clicked.connect(self._on_table_clicked)
        self.point_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.point_table.customContextMenuRequested.connect(self._on_table_context_menu)
        return self.point_table

    # ------------------------------------------------------------------
    # Column filters
    # ------------------------------------------------------------------

    def _filter_values(self, column: int) -> list[str]:
        """The categories offered for a categorical column."""
        if column == INDIVIDUAL_COLUMN:
            return list(self.store.individual_names)
        return [HUMAN_SOURCE, FILL_SOURCE]

    def _on_filter_requested(self, column: int) -> None:
        """A funnel was clicked: edit that column's filter."""
        header = self.point_table.horizontalHeader()
        if header.is_categorical(column):
            dialog = CategoryFilterDialog(
                column, self._filter_values(column), self.point_proxy.cat_filter(column), self
            )
            if dialog.exec_():
                self.point_proxy.set_cat_filter(column, dialog.get_allowed())
        elif header.is_numeric(column):
            dialog = NumericFilterDialog(column, self.point_proxy.num_filter(column), self)
            if dialog.exec_():
                criterion = dialog.get_filter()
                self.point_proxy.set_numeric_filter(column, *(criterion or (None, None)))
        header.set_active_filters(self.point_proxy.active_filters())
        self._select_table_row_for_frame()

    def _clear_filters(self) -> None:
        """Drop every filter — schema edits can make the categories meaningless."""
        self.point_proxy.clear_filters()
        self.point_table.horizontalHeader().set_active_filters(set())

    # ------------------------------------------------------------------

    def _selected_table_keys(self) -> list[tuple[int, str]]:
        """The ``(frame, individual)`` behind each selected row, in view order."""
        rows = self.point_table.selectionModel().selectedRows()
        keys = [self.point_model.key_at(self.point_proxy.mapToSource(index).row()) for index in rows]
        return [key for key in keys if key is not None]

    def _on_table_context_menu(self, position) -> None:
        """Right-click → delete the selected rows' labels, or pin their fill."""
        clicked = self.point_table.indexAt(position)
        if clicked.isValid() and clicked.row() not in {
            index.row() for index in self.point_table.selectionModel().selectedRows()
        }:
            # Right-clicking outside the selection acts on the row under the
            # cursor, which is what every file manager does.
            self.point_table.selectRow(clicked.row())
        keys = self._selected_table_keys()
        if not keys:
            return
        menu = QMenu(self.point_table)
        suffix = "this frame" if len(keys) == 1 else f"{len(keys)} frames"
        # Each action appears only when the selection holds something for it to
        # act on: "Delete labels" on a row that is entirely a prediction did
        # nothing at all, which reads as a broken menu rather than an empty one.
        if any(self.store.is_human(frame, individual) for frame, individual in keys):
            menu.addAction(f"Delete labels on {suffix}", lambda: self._delete_table_rows(keys))
        if any(self.store.has_fill(frame, individual) for frame, individual in keys):
            clear = menu.addAction(f"Delete filled points on {suffix}", lambda: self._clear_fill_rows(keys))
            clear.setToolTip(
                "Throw these predictions away — for frames where the animal is\n"
                "occluded or out of shot and the backend placed a point anyway.\n"
                "Your labels are kept; a later fill will predict them again."
            )
            pin = menu.addAction(f"Pin filled points as labels on {suffix}", lambda: self._pin_table_rows(keys))
            pin.setToolTip(
                "Keep these predicted positions as your own labels, so the next\n"
                "fill treats them as ground truth instead of re-deriving them."
            )
        if menu.isEmpty():
            return
        menu.exec_(self.point_table.viewport().mapToGlobal(position))

    def _delete_table_rows(self, keys: list[tuple[int, str]]) -> None:
        """Drop every labelled point of the given ``(frame, individual)`` rows."""
        for frame, individual in keys:
            self.store.clear_individual(frame, individual)
        self._after_table_edit()

    def _clear_fill_rows(self, keys: list[tuple[int, str]]) -> None:
        """Throw away the given rows' predictions, keeping their labels."""
        removed = sum(self.store.clear_fill_for(frame, individual) for frame, individual in keys)
        if not removed:
            return
        self._push_pose_override()
        self._after_table_edit()
        notify(f"Removed {removed} filled point(s).", "info")

    def _pin_table_rows(self, keys: list[tuple[int, str]]) -> None:
        """Promote the given rows' filled points to labels ("accept the fill")."""
        pinned = sum(self.store.promote_fill(frame, individual) for frame, individual in keys)
        if not pinned:
            notify("Nothing to pin — those points are already labelled.", "info")
            return
        self._after_table_edit()
        notify(f"Pinned {pinned} filled point(s) as labels.", "info")

    def _after_table_edit(self) -> None:
        """A bulk edit touched several frames, so nothing narrow can be repainted."""
        if self._mode is not None:
            self._mode.refresh()
        self._on_store_changed(full=True)
        self._save_store()

    def _build_suggest_group(self) -> QGroupBox:
        """Which frames to label — see :mod:`~ethograph.gui.pose_suggest`."""
        group = QGroupBox("Which frames to label")
        box = QVBoxLayout(group)

        row = QHBoxLayout()
        self.suggest_method_combo = QComboBox()
        for key, label, tip in _SUGGEST_METHODS:
            self.suggest_method_combo.addItem(label, key)
            self.suggest_method_combo.setItemData(self.suggest_method_combo.count() - 1, tip, Qt.ToolTipRole)
        # "uncertain" sits last because it can only rank what a fill has already
        # scored: opening on it would mean the first press of the button can
        # only warn. Before a fill the combo starts at the top of the list.
        opening = "uncertain" if self.store.confidence is not None else _SUGGEST_METHODS[0][0]
        self.suggest_method_combo.setCurrentIndex(self.suggest_method_combo.findData(opening))
        row.addWidget(self.suggest_method_combo, stretch=1)

        # A share of the video, not an absolute count: how many frames are worth
        # labelling scales with how long the clip is, and the resolved count is
        # spelled out beside it so the percentage is never abstract.
        self.suggest_percent_spin = QDoubleSpinBox()
        self.suggest_percent_spin.setRange(MIN_SUGGEST_PERCENT, 100.0)
        self.suggest_percent_spin.setDecimals(1)
        self.suggest_percent_spin.setSingleStep(0.5)
        self.suggest_percent_spin.setSuffix(" %")
        self.suggest_percent_spin.setValue(self._default_suggest_percent())
        self.suggest_percent_spin.setToolTip("What share of the video to propose for labelling.")
        self.suggest_percent_spin.valueChanged.connect(lambda _v: self._refresh_suggest_count_label())
        row.addWidget(self.suggest_percent_spin)
        box.addLayout(row)

        self.suggest_count_label = QLabel()
        self.suggest_count_label.setToolTip("How many frames that percentage works out to.")
        box.addWidget(self.suggest_count_label)

        suggest_btn = QPushButton("Suggest frames")
        suggest_btn.setToolTip(
            "Propose frames spread across the video.\nLabelling neighbouring frames teaches a tracker very little."
        )
        suggest_btn.clicked.connect(self._on_suggest)
        box.addWidget(suggest_btn)

        self.suggestion_label = QLabel("No suggested frames yet.")
        box.addWidget(self.suggestion_label)

        nav = QHBoxLayout()
        prev_btn = QPushButton("← Previous  (Shift+←)")
        prev_btn.setToolTip("Jump to the previous suggested frame. Plain ← / → step one frame at a time.")
        prev_btn.clicked.connect(lambda: self._step_suggestion(-1))
        nav.addWidget(prev_btn)
        next_btn = QPushButton("(Shift+→)  Next →")
        next_btn.setToolTip("Jump to the next suggested frame. Plain ← / → step one frame at a time.")
        next_btn.clicked.connect(lambda: self._step_suggestion(1))
        nav.addWidget(next_btn)
        box.addLayout(nav)
        self._refresh_suggest_count_label()
        return group

    def _default_suggest_percent(self) -> float:
        """The share that works out to :data:`RECOMMENDED_ANCHORS` frames.

        The recommendation is a *count* — about twenty labelled frames is what
        the fill backends need — so it is converted here rather than picking an
        arbitrary percentage that would mean twenty frames on one clip and two
        thousand on another.
        """
        n_frames = self._n_frames()
        if not n_frames:
            return 10.0
        return min(100.0, max(MIN_SUGGEST_PERCENT, round(RECOMMENDED_ANCHORS / n_frames * 100, 1)))

    def _suggest_count(self) -> int:
        """Frames the requested percentage works out to, at least one."""
        return max(1, round(self._n_frames() * self.suggest_percent_spin.value() / 100))

    def _refresh_suggest_count_label(self) -> None:
        n_frames = self._n_frames()
        if not n_frames:
            self.suggest_count_label.setText("Frame count unknown — load a video.")
            return
        self.suggest_count_label.setText(f"{self._suggest_count()} of {n_frames} frames")

    def _build_fill_group(self) -> QGroupBox:
        group = QGroupBox("Fill")
        box = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        for info in available_backends():
            self.backend_combo.addItem(info.label, info.key)
            index = self.backend_combo.count() - 1
            if not info.available:
                item = self.backend_combo.model().item(index)
                item.setEnabled(False)
                self.backend_combo.setItemData(index, f"Not installed — {info.hint}", Qt.ToolTipRole)
        saved = self.app_state.labelling_backend
        position = self.backend_combo.findData(saved)
        self.backend_combo.setCurrentIndex(position if position >= 0 else 0)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row.addWidget(self.backend_combo, stretch=1)
        box.addLayout(row)

        fill_btn = QPushButton("Fill remaining frames")
        fill_btn.clicked.connect(self._on_fill)
        box.addWidget(fill_btn)

        clear_btn = QPushButton("Clear fill")
        clear_btn.setToolTip("Discard the filled frames; labelled anchors are kept.")
        clear_btn.clicked.connect(self._on_clear_fill)
        box.addWidget(clear_btn)
        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("Export")
        box = QVBoxLayout(group)

        self.invert_y_check = QCheckBox("Flip y so plots are not upside down")
        self.invert_y_check.setChecked(True)
        self.invert_y_check.setToolTip(
            "Pose coordinates count y DOWNWARD from the top-left corner; plots count\n"
            "it upward. Without the flip a keypoint at the top of the video is drawn\n"
            "at the bottom of a space plot.\n\n"
            "Applies to 'Load into the GUI' and the NetCDF export. The video overlay\n"
            "and the DeepLabCut export always keep raw image coordinates."
        )
        box.addWidget(self.invert_y_check)

        box.addWidget(QLabel("Derive from the labelled + filled trajectories:"))
        derive_row = QHBoxLayout()
        self.kinematic_checks: dict[str, QCheckBox] = {}
        for name in KINEMATICS:
            check = QCheckBox(name.capitalize())
            check.setChecked(True)
            check.setToolTip(f"Compute {name} per keypoint (movement.kinematics.compute_{name}).")
            derive_row.addWidget(check)
            self.kinematic_checks[name] = check
        derive_row.addStretch()
        box.addLayout(derive_row)

        load_btn = QPushButton("Load into the GUI")
        load_btn.setToolTip(
            "Add the keypoints — and whichever kinematics are ticked — to the\n"
            "current trial as features, so they can be plotted straight away.\n"
            "Filled frames are included, which is the point: it is how you see\n"
            "what the fill actually did. No file is written."
        )
        load_btn.clicked.connect(self._on_load_into_gui)
        box.addWidget(load_btn)

        movement_btn = QPushButton("Export poses (NetCDF)…")
        movement_btn.setToolTip("Write a movement-compatible poses dataset covering every frame.")
        movement_btn.clicked.connect(self._on_export_movement)
        box.addWidget(movement_btn)

        dlc_btn = QPushButton("Export DeepLabCut CollectedData…")
        dlc_btn.setToolTip("Write labelled frames only, for training a DLC model elsewhere.")
        dlc_btn.clicked.connect(self._on_export_dlc)
        box.addWidget(dlc_btn)
        return group

    def _export_image_height(self) -> float | None:
        """Image height for the y-flip, or ``None`` to keep image coordinates."""
        if not self.invert_y_check.isChecked():
            return None
        height = self._view.image_height()
        if not height:
            notify("Video height is unknown — exporting in image coordinates (y not flipped).", "warning")
            return None
        return float(height)

    def _selected_kinematics(self) -> list[str]:
        return [name for name, check in self.kinematic_checks.items() if check.isChecked()]

    def _on_load_into_gui(self) -> None:
        """Add the keypoints (and ticked kinematics) to the trial as features."""
        fps = self._fps()
        if not fps:
            notify("Video frame rate is unknown — cannot build a poses dataset.", "warning")
            return
        if not self.store.anchor_frames():
            notify("Nothing to load — no frames are labelled.", "warning")
            return
        if self.store.filled is None:
            notify("Only the labelled frames will be loaded — run Fill to cover the rest.", "info")

        ds = store_to_movement_ds(self.store, fps, self._export_image_height())
        try:
            # Kinematics come from the flipped positions, so velocity's y sign
            # matches the trajectory the user is looking at.
            arrays = store_to_kinematics(ds, self._selected_kinematics())
        except (KeypointStoreError, ImportError, ValueError) as e:
            notify(f"Could not derive kinematics: {e}", "error")
            return

        added = self._data_widget.add_keypoint_features(arrays)
        if added:
            notify(f"Loaded {', '.join(added)} — add a panel to plot them.", "info")

    # ------------------------------------------------------------------
    # Individual / keypoint tree
    # ------------------------------------------------------------------

    def _keypoint_brush(self, keypoint: str) -> QBrush:
        """The colour the canvas draws *keypoint* in."""
        rgba = keypoint_colors(self.store.n_keypoints)[self.store.keypoint_index(keypoint)]
        return QBrush(QColor.fromRgbF(*(float(c) for c in rgba)))

    def _rebuild_tree(self) -> None:
        """Recreate the branches; call whenever the schema changes.

        The branch text carries the individual's marker glyph and each leaf's
        mark is drawn in its keypoint colour, so the tree reads like the canvas:
        shape = individual, colour = keypoint.
        """
        self.tree.blockSignals(True)
        self.tree.clear()
        for index, individual in enumerate(self.store.individual_names):
            branch = QTreeWidgetItem([f"{glyph_for_individual(index)}  {individual}", ""])
            branch.setData(0, Qt.UserRole, (individual, None))
            for keypoint in self.store.keypoints_for(individual):
                leaf = QTreeWidgetItem([keypoint, ""])
                leaf.setData(0, Qt.UserRole, (individual, keypoint))
                leaf.setForeground(1, self._keypoint_brush(keypoint))
                branch.addChild(leaf)
            self.tree.addTopLevelItem(branch)
            branch.setExpanded(True)
        self.tree.blockSignals(False)
        self._refresh_keypoint_hints()
        self._refresh_tree_marks()
        self._sync_tree_selection()
        self._refresh_active_label()
        self.app_state.labelling_keypoints = list(self.store.keypoint_names)
        self.app_state.labelling_individuals = list(self.store.individual_names)

    def _refresh_tree_marks(self) -> None:
        """Update the per-item "labelled on this frame" column."""
        frame = int(self.app_state.current_frame or 0)
        placed = self.store.anchor_positions(frame)
        for i in range(self.tree.topLevelItemCount()):
            branch = self.tree.topLevelItem(i)
            labelled = 0
            for row in range(branch.childCount()):
                leaf = branch.child(row)
                # A leaf's row is not the schema index once individuals carry
                # different keypoints — always resolve through the name.
                keypoint = leaf.data(0, Qt.UserRole)[1]
                is_set = not np.isnan(placed[i, self.store.keypoint_index(keypoint), 0])
                leaf.setText(1, _LABELLED_MARK if is_set else _UNLABELLED_MARK)
                labelled += int(is_set)
            branch.setText(1, f"{labelled}/{branch.childCount()}")

    def _sync_tree_selection(self) -> None:
        """Highlight the item the canvas is currently writing to."""
        if self._mode is None:
            return
        target = (self._mode.active_individual, self._mode.active_keypoint)
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            branch = self.tree.topLevelItem(i)
            for k in range(branch.childCount()):
                leaf = branch.child(k)
                if leaf.data(0, Qt.UserRole) == target:
                    self.tree.setCurrentItem(leaf)
                    self.tree.blockSignals(False)
                    return
        self.tree.blockSignals(False)

    def _on_tree_item_changed(self, item: QTreeWidgetItem | None, _previous=None) -> None:
        if item is None or self._mode is None:
            return
        individual, keypoint = item.data(0, Qt.UserRole)
        if keypoint is None:
            self._mode.set_active_individual(individual)
        else:
            self._mode.set_active(keypoint, individual)
        self._refresh_active_label()

    # ------------------------------------------------------------------
    # Schema editing
    # ------------------------------------------------------------------

    def _on_add_individual(self) -> None:
        suggestion = f"individual_{self.store.n_individuals}"
        name, ok = QInputDialog.getText(self, "Add individual", "Individual name:", text=suggestion)
        name = name.strip()
        if not ok or not name:
            return
        if name in self.store.individual_names:
            notify(f"Individual {name!r} already exists.", "warning")
            return
        self._apply_schema(individuals=[*self.store.individual_names, name])

    def _on_remove_individual(self) -> None:
        """Remove the selected individual — including the last one.

        Deleting every individual is allowed: with none left nothing can be
        labelled, which is exactly the state to start renaming from.
        """
        individual = self._selected_individual()
        if individual is None:
            notify("There are no individuals to remove.", "warning")
            return
        labelled = self.store.anchor_frames_for_individual(individual)
        if labelled:
            confirm = QMessageBox.question(
                self,
                "Remove individual",
                f"{individual!r} is labelled on {len(labelled)} frame(s).\n"
                "Removing it discards those points. Continue?",
            )
            if confirm != QMessageBox.Yes:
                return
        self._apply_schema(individuals=[n for n in self.store.individual_names if n != individual])

    def _on_shared_toggled(self, shared: bool) -> None:
        self._apply_schema(shared=shared)

    def _on_add_keypoint(self) -> None:
        name, ok = QInputDialog.getText(self, "Add keypoint", "Keypoint name:")
        name = name.strip()
        if not ok or not name:
            return
        if self.store.shared_keypoints:
            if name in self.store.keypoint_names:
                notify(f"Keypoint {name!r} already exists.", "warning")
                return
            self._apply_schema(keypoints=[*self.store.keypoint_names, name])
            return
        individual = self._require_individual()
        if individual is None:
            return
        if name in self.store.keypoints_for(individual):
            notify(f"{individual!r} already has keypoint {name!r}.", "warning")
            return
        self._apply_schema(individual_keypoints=(individual, [*self.store.keypoints_for(individual), name]))

    def _on_remove_keypoint(self) -> None:
        keypoint = self._selected_keypoint()
        if keypoint is None:
            notify("Select a keypoint to remove.", "warning")
            return
        if self.store.shared_keypoints:
            self._apply_schema(keypoints=[n for n in self.store.keypoint_names if n != keypoint])
            return
        individual = self._selected_individual()
        remaining = [n for n in self.store.keypoints_for(individual) if n != keypoint]
        self._apply_schema(individual_keypoints=(individual, remaining))

    def _require_individual(self) -> str | None:
        """The individual per-individual edits apply to, warning when there is none."""
        individual = self._selected_individual()
        if individual is None:
            notify("Add an individual first — keypoints belong to one when they are not shared.", "warning")
        return individual

    def _selected_individual(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return self.store.individual_names[0] if self.store.individual_names else None
        return item.data(0, Qt.UserRole)[0]

    def _selected_keypoint(self) -> str | None:
        item = self.tree.currentItem()
        return None if item is None else item.data(0, Qt.UserRole)[1]

    def _apply_schema(
        self,
        keypoints: list[str] | None = None,
        individuals: list[str] | None = None,
        shared: bool | None = None,
        individual_keypoints: tuple[str, list[str]] | None = None,
    ) -> None:
        """Change the schema; any existing fill is invalidated by the store."""
        if shared is not None:
            self.store.set_shared_keypoints(shared)
        if individuals is not None:
            self.store.set_individual_names(individuals)
        if keypoints is not None:
            self.store.set_keypoint_names(keypoints)
        if individual_keypoints is not None:
            self.store.set_keypoints_for(*individual_keypoints)
        self._push_pose_override()
        if self._mode is not None:
            if self.store.n_individuals:
                self._restart_mode()
            else:
                # Nothing left to label — drop out of labelling mode rather than
                # leaving a canvas that silently swallows clicks.
                self.set_interaction_mode(None)
        self._rebuild_tree()
        # The individuals a filter names may not exist any more, and a filter
        # nobody can see the cause of looks like a broken table.
        self._clear_filters()
        self._refresh_point_table(full=True)
        self._save_store()

    # ------------------------------------------------------------------
    # Labelling mode
    # ------------------------------------------------------------------

    @property
    def interaction_mode(self) -> str | None:
        """``"label"``, ``"edit"``, or ``None`` when the canvas is not armed."""
        return None if self._mode is None else self._mode.mode

    def set_interaction_mode(self, mode: str | None) -> None:
        """Arm the canvas for labelling or editing, or disarm it (``None``)."""
        if mode is not None and not self._can_label():
            mode = None
        if mode is None:
            self._detach_mode()
        elif self._mode is None:
            self._attach_mode(mode)
        else:
            self._mode.set_mode(mode)
        if mode is not None:
            # Arming from the Keypoints tab would otherwise hide the line that
            # says what the next click places.
            self.tabs.setCurrentWidget(self._label_page)
        self._sync_mode_buttons()
        self._refresh_active_label()

    def _toggle_interaction_mode(self, mode: str) -> None:
        """Arm *mode*, or disarm when it is already the one running."""
        self.set_interaction_mode(None if mode == self.interaction_mode else mode)

    def _can_label(self, quiet: bool = False) -> bool:
        """Whether the canvas can be armed, warning about whatever is missing.

        ``quiet`` suppresses the warnings: arming happens automatically when the
        labelling tab is opened, and a user who has not defined a schema yet
        should not be scolded for looking at the tab.
        """
        if not self.store.n_individuals:
            if not quiet:
                notify("Add an individual before labelling.", "warning")
            return False
        if not self.store.keypoint_names:
            if not quiet:
                notify("Add at least one keypoint before labelling.", "warning")
            return False
        if self._view.scene() is None:
            if not quiet:
                notify("Load a video (or a still frame) before labelling.", "warning")
            return False
        return True

    def _on_tab_changed(self, index: int) -> None:
        """Arm Sequential when the labelling tab is opened with nothing armed."""
        if self.tabs.widget(index) is not self._label_page or self._mode is not None:
            return
        if self._can_label(quiet=True):
            self.set_interaction_mode(SEQUENTIAL_MODE)

    def _sync_mode_buttons(self) -> None:
        mode = self.interaction_mode
        self.sequential_btn.setChecked(mode == SEQUENTIAL_MODE)
        self.loop_btn.setChecked(mode == LOOP_MODE)

    def _attach_mode(self, mode: str = SEQUENTIAL_MODE) -> None:
        self.store.n_frames = self._n_frames() or self.store.n_frames
        self._mode = KeypointLabelMode(
            self._view,
            self.store,
            on_changed=self._on_store_changed,
            mode=mode,
            on_advance_frame=self._advance_frame,
            point_size=float(self.app_state.labelling_point_size),
            on_released=self._on_store_changed,
        )
        self._mode.set_frame(int(self.app_state.current_frame or 0))
        self._install_key_filter(True)
        self._sync_tree_selection()
        self._refresh_active_label()

    def _detach_mode(self) -> None:
        if self._mode is None:
            return
        # The key filter stays: Backspace and Ctrl+Z go on working on whatever
        # the Keypoints tree has selected once labelling is disarmed.
        self._mode.detach()
        self._mode = None
        self._save_store()
        self._sync_mode_buttons()
        self._refresh_active_label()

    def _restart_mode(self) -> None:
        mode = self.interaction_mode or SEQUENTIAL_MODE
        keypoint = self._mode.active_keypoint if self._mode else None
        individual = self._mode.active_individual if self._mode else None
        self._detach_mode()
        self._attach_mode(mode)
        self._sync_mode_buttons()
        if individual not in self.store.individual_names:
            individual = None
        if keypoint is not None and self.store.has_keypoint(keypoint, individual):
            self._mode.set_active(keypoint, individual)

    def _on_store_changed(self, full: bool = False, frame: int | None = None) -> None:
        self._refresh_tree_marks()
        self._sync_tree_selection()
        self._refresh_active_label()
        self._refresh_point_table(full=full, frame=frame)
        # A correction has to reach the pose overlay too, or the prediction it
        # replaced stays on screen next to it. Not while a drag is running: the
        # poses dataset is rebuilt whole, which is far too much per mouse move,
        # and the release fires this again once the point has settled.
        if self.store.filled is not None and not (self._mode is not None and self._mode.dragging):
            self._push_pose_override()

    def _on_frame_changed(self, frame: int) -> None:
        if self._mode is not None:
            self._mode.set_frame(int(frame))
        self._refresh_tree_marks()
        self._select_table_row_for_frame()

    def _on_undo(self) -> None:
        # An undo can land on any frame, not the one being viewed, so the table
        # is repainted where the change actually happened. Through the common
        # path, so an undone correction also reaches the pose overlay rather
        # than leaving the point it replaced on screen.
        frame = self.store.undo()
        if self._mode is not None:
            self._mode.refresh()
        self._on_store_changed(frame=frame)

    # ------------------------------------------------------------------
    # Labelled-points table
    # ------------------------------------------------------------------

    def _table_layout(self) -> tuple[list[tuple[int, str]], list[str]]:
        """``(rows, keypoint columns)`` for what the store currently holds.

        A row is one ``(frame, individual)``, so every keypoint of that
        individual on that frame is visible at once.

        Before a fill only labelled rows exist, and columns cover only the
        keypoints carrying at least one label — an unlabelled 20-keypoint schema
        would otherwise push the ones being worked on off the side. Once a fill
        exists every frame has a position for every point, so every frame gets a
        row: a prediction is then visible, filterable and correctable from here
        and not only on the canvas.
        """
        if self.store.filled is not None:
            rows = [
                (frame, individual)
                for frame in range(self.store.n_frames)
                for individual in self.store.individual_names
            ]
            return rows, list(self.store.keypoint_names)
        points = self.store.labelled_points()
        rows = list(dict.fromkeys((frame, individual) for frame, individual, _kp, _x, _y in points))
        labelled = {keypoint for _frame, _individual, keypoint, _x, _y in points}
        return rows, [name for name in self.store.keypoint_names if name in labelled]

    def _layout_signature(self) -> tuple:
        """What :meth:`_table_layout` depends on, cheap enough to test per drag.

        With a fill loaded the row set is the whole video, so it must not be
        rebuilt — or even enumerated — on every mouse move of a drag.
        """
        if self.store.filled is not None:
            return (True, self.store.n_frames, tuple(self.store.individual_names), tuple(self.store.keypoint_names))
        rows, columns = self._table_layout()
        return (False, tuple(rows), tuple(columns))

    def _refresh_point_table(self, full: bool = False, frame: int | None = None) -> None:
        """Sync the table with the store.

        Values are read straight from the store by the model, so a refresh only
        has to say *what changed*: normally one frame (a placement or a drag,
        which fires on every mouse move), or everything after a fill. *frame*
        defaults to the one being edited — pass it only when an edit landed
        somewhere else, as an undo can.
        """
        signature = self._layout_signature()
        if signature != self._table_signature:
            self._table_signature = signature
            rows, columns = self._table_layout()
            self.point_model.set_layout(rows, columns)
            self.point_table.horizontalHeader().set_groups(columns, [self._keypoint_brush(name) for name in columns])
        elif full:
            self.point_model.refresh_all()
        else:
            self.point_model.refresh_frame(self._current_frame() if frame is None else frame)
        self._select_table_row_for_frame()

    def _current_frame(self) -> int:
        """The frame edits land on — the canvas mode's, else the playhead's."""
        return self._mode.frame if self._mode is not None else int(self.app_state.current_frame or 0)

    def _select_table_row_for_frame(self) -> None:
        """Highlight the playhead's first visible row, without seeking."""
        frame = int(self.app_state.current_frame or 0)
        for individual in self.store.individual_names or [None]:
            source_row = self.point_model.row_of((frame, individual))
            if source_row is None:
                continue
            index = self.point_proxy.mapFromSource(self.point_model.index(source_row, 0))
            if index.isValid():
                self.point_table.selectRow(index.row())
                self.point_table.scrollTo(index)
                return
        self.point_table.clearSelection()

    def _on_table_clicked(self, index) -> None:
        """Seek to the clicked row's frame and make what was clicked active."""
        key = self.point_model.key_at(self.point_proxy.mapToSource(index).row())
        if key is None:
            return
        frame, individual = key
        self._seek(frame)
        if self._mode is None or individual not in self.store.individual_names:
            return
        target = self.point_model.keypoint_at(index.column())
        if target is not None and self.store.has_keypoint(target[0], individual):
            self._mode.set_active(target[0], individual)
        else:
            self._mode.set_active_individual(individual)
        self._sync_tree_selection()
        self._refresh_active_label()

    def _seek(self, frame: int) -> None:
        video = self.app_state.video
        if video is None:
            notify("No video is loaded to seek.", "warning")
            return
        video.seek_to_frame(int(frame))

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def _install_key_filter(self, install: bool) -> None:
        """Catch the dialog's keys wherever they are actually pressed.

        Three targets, because focus moves between windows while labelling: the
        dialog, the **video canvas** (which belongs to the main window, so
        clicking to place a point takes focus out of the dialog) and the **main
        window** itself (key events propagate up to it from any widget inside).
        Without the third, Shift+arrows reached the main window's own
        window-stepping shortcut instead of stepping the suggested frames.

        Only the arrow keys are claimed from the main window — see
        :meth:`_owned_key`. Everything else there belongs to whatever the user
        is doing over there.

        Installed for the dialog's whole lifetime, not only while a mode is
        armed: deleting the keypoint you have selected must not require arming
        labelling first. Re-installing is safe (Qt drops the earlier
        registration of the same filter), and the canvas only exists once a
        video is loaded.
        """
        for widget in (self, self._view.canvas_widget(), self._shell):
            if widget is None:
                continue
            if install:
                widget.installEventFilter(self)
            else:
                widget.removeEventFilter(self)

    def _owned_key(self, event, main_window: bool = False) -> bool:
        """Keys this dialog consumes, given what is armed and where they landed.

        Deletion and undo always count; the target keys only while a mode runs,
        so the main window keeps its `1`-`9` behaviour labels. Events that came
        from the *main window* yield everything except the arrows: the user is
        working over there, and only the suggestion stepping is meant to reach
        across.
        """
        key = event.key()
        # Shift+arrows step the suggested frames — claimed from the main
        # window's window-stepping only while there is a list to step through.
        # Plain arrows are left alone: single-frame stepping stays theirs.
        if key in (Qt.Key_Left, Qt.Key_Right):
            return bool(self._suggestions) and self._shift_only(event)
        if main_window:
            return False
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            return not self._typing()
        if key == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            return True
        if self._mode is None:
            return False
        if key in (Qt.Key_Tab, Qt.Key_Backtab):
            return True
        return Qt.Key_1 <= key <= Qt.Key_9 and not event.modifiers()

    @staticmethod
    def _shift_only(event) -> bool:
        modifiers = event.modifiers()
        return bool(modifiers & Qt.ShiftModifier) and not modifiers & (Qt.ControlModifier | Qt.AltModifier)

    def _typing(self) -> bool:
        """Whether focus sits in a text field, where Backspace means editing.

        The spin boxes on this dialog are the ones that matter: eating their
        Backspace would delete a keypoint instead of a digit.
        """
        return isinstance(QApplication.focusWidget(), (QAbstractSpinBox, QLineEdit))

    def eventFilter(self, obj, event):
        # Type first: this filter sits on the main window too, so every event
        # there passes through here and must cost as little as possible.
        if event.type() not in (QEvent.ShortcutOverride, QEvent.KeyPress):
            return False
        # The main window binds 1-9 (behaviour labels) and Shift+arrows (window
        # stepping) as QShortcuts. Accepting the ShortcutOverride keeps those
        # from swallowing the key, so the KeyPress below still reaches us.
        if not self._owned_key(event, main_window=obj is self._shell):
            return False
        if event.type() == QEvent.ShortcutOverride:
            event.accept()
            return True
        key = event.key()
        if key in (Qt.Key_Backspace, Qt.Key_Delete):
            return self._delete_selected_point()
        if key == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self._on_undo()
            return True
        if key in (Qt.Key_Left, Qt.Key_Right):
            self._step_suggestion(1 if key == Qt.Key_Right else -1)
            return True
        if key in (Qt.Key_Tab, Qt.Key_Backtab):
            self._mode.cycle(1 if key == Qt.Key_Tab else -1)
            self._on_store_changed()
            return True
        if self._mode.select_individual_by_number(key - Qt.Key_1 + 1):
            self._on_store_changed()
            return True
        return False

    def _delete_selected_point(self) -> bool:
        """Backspace/Delete: remove the selected point from the current frame.

        With the canvas armed that is the outlined point, falling back to the
        one under the cursor. Otherwise it is whatever the Keypoints tree
        highlights — the same pair the tree, the combos and the table selection
        all agree on — so a keypoint can be deleted without arming a mode.

        Note that a *filled* point cannot be deleted, only its label: with a
        fill loaded the prediction stays on screen (dimmed, and its table row
        turns to ``Fill``), which is correct — the next fill is free to place it
        again, unconstrained by the label you removed.
        """
        if self._mode is not None:
            deleted = self._mode.delete_selected()
        else:
            frame = self._current_frame()
            individual, keypoint = self._selected_individual(), self._selected_keypoint()
            deleted = keypoint is not None and self.store.is_anchor(frame, keypoint, individual)
            if deleted:
                self.store.clear_point(frame, keypoint, individual)
                self._on_store_changed(frame=frame)
        if deleted:
            self._save_store()
        return deleted

    # ------------------------------------------------------------------
    # Fill
    # ------------------------------------------------------------------

    def _on_backend_changed(self, _index: int) -> None:
        self.app_state.labelling_backend = self.backend_combo.currentData()

    def _on_fill(self) -> None:
        if not self.store.anchor_frames():
            notify("Label at least one frame before filling.", "warning")
            return
        n_frames = self._n_frames()
        if not n_frames:
            notify("Frame count is unknown — load a video first.", "warning")
            return
        self.store.n_frames = n_frames

        key = self.backend_combo.currentData()
        label = self.backend_combo.currentText()
        busy = BusyProgressDialog(f"Filling frames with {label}…", parent=self)

        def report(stage: str):
            def progress(fraction: float) -> bool:
                busy.setLabelText(f"{stage} {fraction:.0%}")
                busy.pump_events()
                return not busy.wasCanceled()

            return progress

        # The backend is built INSIDE the dialog: CoTracker downloads ~97 MB of
        # weights on first use, and that must be visible and cancellable rather
        # than freezing the UI before any progress bar exists.
        result, error = busy.execute(self._build_and_fill, key, label, report)
        if error is not None or result is None:
            return

        filled, confidence = result
        self.store.set_fill_from_flat(filled, confidence)
        self._push_pose_override()
        # Every row's coordinates and provenance changed, and the table now
        # covers every frame rather than the labelled ones.
        self._refresh_point_table(full=True)
        self._save_store()
        notify(f"Filled {n_frames} frames from {len(self.store.anchor_frames())} labelled ones.", "info")

    def _build_and_fill(self, key: str, label: str, report):
        """Backends track flat points — the individual/keypoint split is restored after."""
        backend = build_backend(key, progress=report("Downloading CoTracker3 weights…"))
        frames = None
        if backend.requires_video:
            frames = self._open_frames()
        try:
            return backend.fill(
                self.store.flat_anchors(),
                self.store.n_frames,
                frames,
                report(f"Filling frames with {label}…"),
            )
        finally:
            if frames is not None:
                frames.close()

    def _open_frames(self, max_side: int = MAX_SIDE) -> VideoFrameSource:
        video = self._video_path()
        fps = self._fps()
        if not video:
            raise ValueError("This needs the video, but no video is loaded.")
        if not fps:
            raise ValueError("Video frame rate is unknown — cannot decode frames.")
        return VideoFrameSource(
            video,
            fps=fps,
            n_frames=self.store.n_frames,
            max_side=max_side,
            start_frame=self._view.start_frame,
        )

    # ------------------------------------------------------------------
    # Which frames to label
    # ------------------------------------------------------------------

    def _on_suggest(self) -> None:
        """Propose frames to label, then jump to the first one.

        Labelling consecutive frames is close to wasted effort — they are nearly
        identical. See :mod:`~ethograph.gui.pose_suggest` for the strategies.
        """
        method = self.suggest_method_combo.currentData()
        n_frames = self._n_frames()
        if not n_frames:
            notify("Frame count is unknown — load a video first.", "warning")
            return
        count = self._suggest_count()

        exclude = set(self.store.anchor_frames())
        if method == "uncertain" and self.store.confidence is None:
            notify("Run Fill first — this suggests the frames the fill was least sure about.", "warning")
            return

        # Only the pixel methods decode video; the others are instant.
        if method in ("uniform", "uncertain"):
            picks = suggest_frames(
                method,
                count,
                n_frames,
                exclude=exclude,
                confidence=self.store.confidence,
            )
            error = None
        else:
            busy = BusyProgressDialog("Scanning the video…", parent=self)

            def progress(fraction: float) -> bool:
                busy.setLabelText(f"Scanning the video… {fraction:.0%}")
                busy.pump_events()
                return not busy.wasCanceled()

            picks, error = busy.execute(self._run_suggest, method, count, n_frames, progress)
        if error is not None or not picks:
            if error is None:
                notify("No frames to suggest.", "warning")
            return

        self._suggestions = list(picks)
        self._suggestion_index = 0
        self._go_to_suggestion(0)

    def _run_suggest(self, method: str, count: int, n_frames: int, progress):
        # Thumbnails only — the suggestion scan never needs full-size frames.
        frames = self._open_frames(max_side=SUGGEST_MAX_SIDE)
        try:
            return suggest_frames(
                method,
                count,
                n_frames,
                frames,
                exclude=set(self.store.anchor_frames()),
                progress=progress,
            )
        finally:
            frames.close()

    def _go_to_suggestion(self, index: int) -> None:
        """Seek the playhead to a suggested frame."""
        if not self._suggestions:
            return
        self._suggestion_index = index % len(self._suggestions)
        self._seek(self._suggestions[self._suggestion_index])
        self._refresh_suggestion_label()

    def _advance_frame(self) -> None:
        """Where Loop mode goes after a click — whatever "Between clicks" says.

        Explicit rather than inferred: this used to follow the suggestion list
        whenever one existed, which meant the same click did different things
        depending on state the user could not see from the canvas.
        """
        behaviour = self.after_click_combo.currentData()
        if behaviour == AFTER_CLICK_FRAME:
            self._step_frames(1)
        elif behaviour == AFTER_CLICK_SUGGESTION:
            self._step_suggestion(1)

    def _step_frames(self, direction: int) -> None:
        """Seek one frame, clamped to the clip."""
        total = self._n_frames() or 0
        if not total:
            return
        frame = int(self.app_state.current_frame or 0) + direction
        self._seek(max(0, min(frame, total - 1)))

    def _step_suggestion(self, step: int) -> None:
        if not self._suggestions:
            notify("No suggestions yet — press Suggest frames.", "warning")
            return
        self._go_to_suggestion(self._suggestion_index + step)

    def _refresh_suggestion_label(self) -> None:
        if not self._suggestions:
            self.suggestion_label.setText("No suggested frames yet.")
            return
        position = self._suggestion_index + 1
        frame = self._suggestions[self._suggestion_index]
        done = sum(1 for f in self._suggestions if f in self.store.anchors)
        self.suggestion_label.setText(
            f"Suggestion {position}/{len(self._suggestions)} — frame {frame}   ·   {done} labelled"
        )

    def _on_clear_fill(self) -> None:
        self.store.clear_fill()
        self._push_pose_override()
        self._refresh_point_table(full=True)

    def _push_pose_override(self) -> None:
        """Render the store through the normal pose overlay.

        Confidence filtering then comes for free — low-confidence filled points
        are hidden by the existing "Filter below confidence" spinbox.
        """
        pose_mgr = self._data_widget.pose_mgr
        if pose_mgr is None:
            return
        fps = self._fps()
        if self.store.filled is None or not fps:
            pose_mgr.set_pose_override(None)
        else:
            # No y-flip here: the overlay draws in image coordinates and does
            # its own `y_world = img_height - y` (see pose_overlay). Flipping
            # first would mirror the points off the animal.
            ds = store_to_movement_ds(self.store, fps)
            pose_mgr.set_pose_override(movement_ds_to_pose_render(ds, "labelled keypoints"))
        self._data_widget.update_pose()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _on_export_movement(self) -> None:
        fps = self._fps()
        if not fps:
            notify("Video frame rate is unknown — cannot build a poses dataset.", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export poses", "keypoints.nc", "NetCDF (*.nc)")
        if not path:
            return
        store_to_movement_ds(self.store, fps, self._export_image_height()).to_netcdf(path)
        notify(f"Wrote {path}", "info")

    def _on_export_dlc(self) -> None:
        if not self.store.anchor_frames():
            notify("Nothing to export — no frames are labelled.", "warning")
            return
        scorer, ok = QInputDialog.getText(self, "DeepLabCut export", "Scorer name:", text="ethograph")
        if not ok or not scorer.strip():
            return
        scorer = scorer.strip()
        default = f"CollectedData_{scorer}.h5"
        path, _ = QFileDialog.getSaveFileName(self, "Export CollectedData", default, "HDF5 (*.h5)")
        if not path:
            return
        video = self._video_path()
        video_name = video.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0] if video else "video"
        try:
            store_to_dlc_h5(self.store, path, scorer, video_name)
        except ImportError:
            QMessageBox.warning(
                self,
                "DeepLabCut export",
                "Writing HDF5 needs pytables:\n\n    pip install tables",
            )
            return
        notify(f"Wrote {path} ({len(self.store.anchor_frames())} labelled frames)", "info")

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._detach_mode()
        self._install_key_filter(False)
        try:
            self.app_state.current_frame_changed.disconnect(self._on_frame_changed)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(event)
