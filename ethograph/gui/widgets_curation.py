"""Curation section of the Labels tab — the one place labels get reviewed.

Every label carries a ``labeling_method`` (``labels/curation.py``): *manual*,
*automated* or *curated*. This panel, sitting under the label tables, is where
a user turns automated labels into curated ones:

* **Scope** — which label classes curation acts on. Rows dragged out of the
  label tables above land in the drop area (their ids are listed); *All*
  means every class and *Reset* empties the area again.
* **Mode** — how a label gets curated:

  - *Manual (trial level)*: any label edit stamps that label manual (the
    store does this); **Ctrl+C** curates every automated label in scope of
    the current trial.
  - *Inspect is enough (trial level)*: merely opening a trial curates its
    automated labels in scope.
  - *Frame-by-frame review*: the labels in scope become a queue of
    boundaries walked one by one, each centred in a small view window.
    ``←``/``→`` nudge the video, **Enter** commits the frame on screen as the
    boundary (the label becomes manual if it moved), **Backspace** deletes the
    event outright, **B**/**N** go back / next — and *N* also curates the
    boundary it leaves when the checkbox says so.

* **Label grid view…** / **Video grid…** open the review grids
  (``dialog_label_gridview.py``, ``dialog_video_grid.py``) on the scope; a
  tile click there navigates, and in frame-by-frame mode it drops straight
  into the review at that label.

The per-trial verdict (no automated label left) colours the trial combo and
the bottom bar, and is written to the metadata table's ``curated`` column on
a timer (:data:`METADATA_SYNC_MS`) — labelling must never wait on a file
write.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.label_drawing_mixin import draw_key
from ethograph.gui.notify import notify
from ethograph.gui.shortcuts import typing_in_text_field
from ethograph.io.time_model import TimeRange
from ethograph.labels.curation import (
    CURATED_COLUMN,
    ReviewTarget,
    build_review_queue,
    curate_label,
    curate_trial,
    curated_column_differs,
    method_counts,
    queue_index_of,
    row_mask,
    targets_from_seeds,
)
from ethograph.labels.intervals import (
    HUMAN_CONFIDENCE,
    LABELING_AUTOMATED,
    LABELING_CURATED,
    LABELING_MANUAL,
    delete_interval,
    ensure_labeling_method,
)

logger = logging.getLogger(__name__)

#: How often the per-trial verdicts are pushed into the metadata table's
#: ``curated`` column. Curating is a stream of small edits; writing the
#: table after each would put a file write on the labelling path.
METADATA_SYNC_MS = 5000

#: Curation modes: key → combo text.
CURATION_MODES = {
    "manual": "Manual (trial level)",
    "inspect": "Inspect is enough (trial level)",
    "frame": "Frame-by-frame review",
}

_MODE_HINTS = {
    "manual": "Editing a label makes it manual · Ctrl+C curates every automated label in scope of this trial.",
    "inspect": "Opening a trial curates its automated labels in scope — looking is enough.",
    "frame": "Walk the labels in scope boundary by boundary; N curates what it leaves behind.",
}

_FIELD_TITLES = {"point": "POINT", "start": "START", "end": "END"}

#: Linger on a just-committed boundary this long before jumping to the next
#: seed, so the user sees the label land where they put it.
_CONFIRM_PAUSE_MS = 100

_KEYS_SCHEMATIC = (
    "<table cellspacing='2' style='color:#bbb; font-size:10px;'>"
    "<tr><td><b>←</b> / <b>→</b></td><td>one frame</td>"
    "<td>&nbsp;&nbsp;<b>Enter</b></td><td>confirm this frame</td></tr>"
    "<tr><td><b>B</b> / <b>N</b></td><td>back / next</td>"
    "<td>&nbsp;&nbsp;<b>Backspace</b></td><td>delete the event</td></tr>"
    "</table>"
)


def drag_label_ids(text: str) -> list[int]:
    """Label ids carried by a drag out of the label tables (``"1,4,8"``)."""
    ids: list[int] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            return []
        if value not in ids:
            ids.append(value)
    return ids


def _num(value) -> float | None:
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _close(a: float | None, b: float | None, atol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= atol


def _color_hex(mapping: dict) -> str:
    color = mapping.get("color")
    if color is None:
        return "#ffffff"
    return "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in color[:3]))


# ----------------------------------------------------------------------
# Scope drop area
# ----------------------------------------------------------------------


class ScopeDropArea(QFrame):
    """Where label rows dragged out of the tables land; lists their ids."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(30)
        self.setToolTip(
            "Drag label rows from the tables above here to curate only those classes.\n"
            "Empty means every class."
        )
        self._ids: list[int] = []
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        self._label = QLabel()
        self._label.setWordWrap(True)
        lay.addWidget(self._label, stretch=1)
        self._refresh()

    def ids(self) -> list[int]:
        return list(self._ids)

    def set_ids(self, ids) -> None:
        self._ids = [int(i) for i in (ids or [])]
        self._refresh()

    def add_ids(self, ids) -> bool:
        """Add *ids* (keeping order, no duplicates); True when something changed."""
        added = False
        for i in ids:
            if int(i) not in self._ids:
                self._ids.append(int(i))
                added = True
        if added:
            self._refresh()
        return added

    def _refresh(self) -> None:
        if self._ids:
            self._label.setText("Labels to curate: " + ", ".join(str(i) for i in self._ids))
            self.setStyleSheet("QFrame { border: 1px solid #ffe066; border-radius: 3px; }")
        else:
            self._label.setText("All labels — drag label rows here to narrow")
            self.setStyleSheet("QFrame { border: 1px dashed #888; border-radius: 3px; color: #aaa; }")

    def dragEnterEvent(self, event):
        if event.mimeData().hasText() and drag_label_ids(event.mimeData().text()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        ids = drag_label_ids(event.mimeData().text())
        if not ids:
            event.ignore()
            return
        # A drop here is a copy into the scope — the row stays in its table,
        # so the default MoveAction must not be reported back to the source.
        event.setDropAction(Qt.CopyAction)
        event.accept()
        if self.add_ids(ids):
            panel = self.parent()
            while panel is not None and not isinstance(panel, CurationPanel):
                panel = panel.parent()
            if panel is not None:
                panel._on_scope_edited()


class ShortcutsPopup(QDialog):
    """The frame-by-frame keys, drawn as a little schematic."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame-by-frame review keys")
        self.setModal(False)
        lay = QVBoxLayout(self)
        rows = [
            ("←  /  →", "step the video one frame back / forward"),
            ("Enter", "confirm: the frame on screen becomes the boundary and the review moves on"),
            ("Backspace  /  Delete", "this event should not exist — delete it and move on"),
            ("B", "back to the previous boundary"),
            ("N", "next boundary (curates the one you leave when the box is ticked)"),
            ("Space", "play / pause"),
            ("Ctrl+C", "curate every automated label in scope of this trial"),
        ]
        for key, what in rows:
            row = QHBoxLayout()
            key_label = QLabel(key)
            key_label.setStyleSheet(
                "QLabel { background: #333; color: #ffe066; border: 1px solid #666; border-radius: 4px;"
                " padding: 2px 8px; font-weight: bold; }"
            )
            key_label.setMinimumWidth(150)
            key_label.setAlignment(Qt.AlignCenter)
            row.addWidget(key_label)
            row.addWidget(QLabel(what), stretch=1)
            lay.addLayout(row)
        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)
        lay.addWidget(close_btn, alignment=Qt.AlignRight)


# ----------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------


class CurationPanel(QGroupBox):
    """Scope + mode + frame-by-frame review, under the label tables."""

    def __init__(self, app_state, labels_widget, parent=None):
        super().__init__("Curation", parent)
        self.app_state = app_state
        self.labels_widget = labels_widget
        self.meta = None
        self.nav = None
        self.data_widget = None
        self.plot_container = None
        self._grid_dialog = None
        self._video_dialog = None
        self._shortcuts_popup: ShortcutsPopup | None = None

        # Frame-by-frame review state (one session at a time)
        self._targets: list[ReviewTarget] = []
        self._idx = 0
        self._seed_frame: int | None = None
        self._session_active = False
        self._advance_pending = False
        self._jumping = False
        self._frame_conn = False
        self._n_confirmed = 0
        self._n_deleted = 0
        self._session_shortcuts: list[QShortcut] = []

        self._build_ui()

        self._metadata_timer = QTimer(self)
        self._metadata_timer.setInterval(METADATA_SYNC_MS)
        self._metadata_timer.timeout.connect(self.sync_metadata)
        self._metadata_timer.start()

        app_state.trial_changed.connect(self._on_trial_changed)
        app_state.current_frame_changed.connect(self._on_frame_changed)
        app_state.curation_mode_changed.connect(self._sync_mode_from_state)
        app_state.curation_label_ids_changed.connect(self._sync_scope_from_state)
        self._sync_mode_from_state()
        self._sync_scope_from_state()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(4, 4, 4, 4)

        scope_row = QHBoxLayout()
        self.scope_area = ScopeDropArea()
        scope_row.addWidget(self.scope_area, stretch=1)
        self.scope_all_btn = QPushButton("All")
        self.scope_all_btn.setFixedWidth(36)
        self.scope_all_btn.setToolTip("Curate every label class")
        self.scope_all_btn.clicked.connect(self._scope_all)
        scope_row.addWidget(self.scope_all_btn)
        self.scope_reset_btn = QPushButton("Reset")
        self.scope_reset_btn.setFixedWidth(48)
        self.scope_reset_btn.setToolTip("Empty the scope so other labels can be dragged in")
        self.scope_reset_btn.clicked.connect(self._scope_all)
        scope_row.addWidget(self.scope_reset_btn)
        lay.addLayout(scope_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        for key, text in CURATION_MODES.items():
            self.mode_combo.addItem(text, key)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_combo)
        mode_row.addWidget(self.mode_combo, stretch=1)
        lay.addLayout(mode_row)

        self.mode_hint = QLabel("")
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setStyleSheet("color: #aaa; font-size: 10px;")
        lay.addWidget(self.mode_hint)

        # ── Frame-by-frame review ───────────────────────────────────
        self.frame_group = QWidget()
        frame_lay = QVBoxLayout(self.frame_group)
        frame_lay.setContentsMargins(0, 2, 0, 0)
        frame_lay.setSpacing(3)

        self.target_label = QLabel("")
        self.target_label.setAlignment(Qt.AlignCenter)
        self.target_label.setTextFormat(Qt.PlainText)
        self.target_label.setWordWrap(True)
        frame_lay.addWidget(self.target_label)
        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        frame_lay.addWidget(self.info_label)
        self.delta_label = QLabel("")
        self.delta_label.setAlignment(Qt.AlignCenter)
        frame_lay.addWidget(self.delta_label)

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("View window:"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.02, 600.0)
        self.window_spin.setDecimals(2)
        self.window_spin.setSingleStep(0.2)
        self.window_spin.setSuffix(" s")
        self.window_spin.setToolTip(
            "Seconds of time series shown around the boundary being reviewed\n"
            "(seed centred) — independent of the navigation Before/After padding."
        )
        self.window_spin.setValue(float(self.app_state.get_with_default("refine_window_s")))
        self.window_spin.valueChanged.connect(self._on_window_changed)
        # Enter in the spinbox commits the value and hands focus back, so the
        # NEXT Enter is a boundary confirm again — one keypress, one meaning.
        self.window_spin.editingFinished.connect(self.window_spin.clearFocus)
        win_row.addWidget(self.window_spin, stretch=1)
        self.lock_checkbox = QCheckBox("Locked around label")
        self.lock_checkbox.setChecked(True)
        self.lock_checkbox.setToolTip(
            "Ticked: the view stays a small window around the boundary.\n"
            "Unticked: pan/zoom the whole trial freely — Enter still confirms\n"
            "the frame on screen, wherever you navigated."
        )
        self.lock_checkbox.toggled.connect(self._on_lock_toggled)
        win_row.addWidget(self.lock_checkbox)
        frame_lay.addLayout(win_row)

        self.next_curates_cb = QCheckBox("N (next) = seen, mark curated")
        self.next_curates_cb.setToolTip(
            "Moving on with N means you looked at the boundary and it is fine:\n"
            "an automated label becomes curated. Untick to only browse."
        )
        self.next_curates_cb.setChecked(bool(self.app_state.get_with_default("curation_next_curates")))
        self.next_curates_cb.toggled.connect(lambda v: setattr(self.app_state, "curation_next_curates", v))
        frame_lay.addWidget(self.next_curates_cb)

        keys_row = QHBoxLayout()
        keys = QLabel(_KEYS_SCHEMATIC)
        keys.setTextFormat(Qt.RichText)
        keys_row.addWidget(keys, stretch=1)
        self.shortcuts_btn = QPushButton("Shortcuts…")
        self.shortcuts_btn.setAutoDefault(False)
        self.shortcuts_btn.clicked.connect(self._show_shortcuts)
        keys_row.addWidget(self.shortcuts_btn, alignment=Qt.AlignTop)
        frame_lay.addLayout(keys_row)

        self.start_stop_btn = QPushButton("Start review")
        self.start_stop_btn.setAutoDefault(False)
        self.start_stop_btn.setDefault(False)
        self.start_stop_btn.clicked.connect(self._toggle_session)
        frame_lay.addWidget(self.start_stop_btn)
        lay.addWidget(self.frame_group)

        # ── Tools ───────────────────────────────────────────────────
        tools_row = QHBoxLayout()
        self.curate_trial_btn = QPushButton("Curate trial (Ctrl+C)")
        self.curate_trial_btn.setAutoDefault(False)
        self.curate_trial_btn.setToolTip(
            "Every automated label in scope of the current trial becomes curated.\n"
            "Manual labels stay manual."
        )
        self.curate_trial_btn.clicked.connect(self.curate_current_trial)
        tools_row.addWidget(self.curate_trial_btn)
        self.grid_btn = QPushButton("Label grid view…")
        self.grid_btn.setAutoDefault(False)
        self.grid_btn.setToolTip("A grid of video frames at the label times in scope — click a tile to go there")
        self.grid_btn.clicked.connect(self.open_grid_view)
        tools_row.addWidget(self.grid_btn)
        self.video_grid_btn = QPushButton("Video grid…")
        self.video_grid_btn.setAutoDefault(False)
        self.video_grid_btn.setToolTip(
            "Play the labels in scope side by side, one label class per group\n(decodes video — slower to build)"
        )
        self.video_grid_btn.clicked.connect(self.open_video_grid)
        tools_row.addWidget(self.video_grid_btn)
        lay.addLayout(tools_row)

        self.status_label = QLabel("")
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setStyleSheet("font-size: 10px;")
        lay.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_data_widget(self, data_widget) -> None:
        self.data_widget = data_widget
        self.nav = getattr(data_widget, "navigation_widget", None) or self.nav

    def set_plot_container(self, plot_container) -> None:
        self.plot_container = plot_container

    def set_meta(self, meta) -> None:
        self.meta = meta
        self.nav = getattr(meta, "navigation_widget", None) or self.nav
        self.data_widget = getattr(meta, "data_widget", None) or self.data_widget

    def _trials_widget(self):
        return getattr(self.meta, "trials_widget", None) or getattr(self.data_widget, "trials_widget", None)

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    def scope(self) -> set[int] | None:
        """Label classes curation acts on; ``None`` = every class."""
        return self.app_state.curation_scope()

    def scope_or_all_ids(self) -> list[int]:
        """The scope as an explicit id list (every mapped class when unset)."""
        ids = self.scope_area.ids()
        if ids:
            return ids
        mappings = getattr(self.labels_widget, "_mappings", {}) or {}
        return sorted(lid for lid in mappings if isinstance(lid, int) and lid != 0)

    def _on_scope_edited(self) -> None:
        self.app_state.curation_label_ids = self.scope_area.ids() or None
        self._refresh_status()

    def _scope_all(self) -> None:
        self.scope_area.set_ids([])
        self._on_scope_edited()

    def _sync_scope_from_state(self, *_args) -> None:
        ids = self.app_state.curation_label_ids or []
        if ids != self.scope_area.ids():
            self.scope_area.set_ids(ids)
        self._refresh_status()

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def mode(self) -> str:
        return str(self.mode_combo.currentData() or "manual")

    def _on_mode_combo(self, _index: int) -> None:
        key = self.mode()
        if self.app_state.curation_mode != key:
            self.app_state.curation_mode = key
        self._apply_mode(key)

    def _sync_mode_from_state(self, *_args) -> None:
        key = str(self.app_state.get_with_default("curation_mode") or "manual")
        idx = self.mode_combo.findData(key)
        if idx < 0:
            idx = 0
        if self.mode_combo.currentIndex() != idx:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(idx)
            self.mode_combo.blockSignals(False)
        self._apply_mode(self.mode())

    def _apply_mode(self, key: str) -> None:
        self.mode_hint.setText(_MODE_HINTS.get(key, ""))
        self.frame_group.setVisible(key == "frame")
        if key != "frame" and self._session_active:
            self._stop()
        if key == "inspect" and self.app_state.ready:
            self.curate_current_trial(quiet=True)

    # ------------------------------------------------------------------
    # Applying method changes
    # ------------------------------------------------------------------

    def _commit(self, df, n: int, *, restyle: tuple | None = None, message: str | None = None) -> int:
        """Swap *df* in and refresh what shows a label's method.

        *restyle* is ``(inst, automated)`` for a single-label transition —
        restyled in place on the plots when it is on screen, else a full
        label redraw. A whole trial changing is always a full redraw.
        """
        if not n:
            return 0
        self.app_state.replace_all_labels(df)
        self.app_state.changes_saved = False
        restyled = 0
        if restyle is not None and self.plot_container is not None:
            inst, automated = restyle
            key = draw_key(
                inst["labels"],
                self.app_state.to_display(inst["trial"], float(inst["onset_s"])),
                inst.get("individual"),
                inst.get("individual_rec"),
            )
            restyled = self.plot_container.restyle_label(key, automated)
        if not restyled and self.plot_container is not None:
            self.plot_container.schedule_labels_redraw()
        self.app_state.curation_changed.emit()
        self._refresh_status()
        if message:
            notify(message)
        return n

    def curate_current_trial(self, quiet: bool = False) -> int:
        """Ctrl+C: every automated label in scope of the current trial → curated."""
        trial = getattr(self.app_state, "trials_sel", None)
        if trial is None or not self.app_state.ready:
            return 0
        df, n = curate_trial(self.app_state._all_labels_df, trial, self.scope())
        if not n:
            if not quiet:
                notify(f"Trial {trial}: nothing left to curate in scope.")
            return 0
        return self._commit(df, n, message=None if quiet else f"Trial {trial}: curated {n} label(s).")

    def curate_labels(self, insts: list[dict]) -> int:
        """Curate the given labels (grid-view verdicts); manual ones are untouched."""
        df = self.app_state._all_labels_df
        total = 0
        for inst in insts:
            df, n = curate_label(df, inst)
            total += n
        if total:
            self._commit(df, total, message=f"Curated {total} label(s).")
        return total

    def note_labels_edited(self) -> None:
        """A label was placed, moved, deleted or undone — the verdict may have changed."""
        self.app_state.curation_changed.emit()
        self._refresh_status()

    # ------------------------------------------------------------------
    # Status + metadata
    # ------------------------------------------------------------------

    def _refresh_status(self) -> None:
        trial = getattr(self.app_state, "trials_sel", None)
        if trial is None or not self.app_state.ready:
            self.status_label.setText("")
            return
        counts = method_counts(self.app_state._all_labels_df, trial)
        automated = counts[LABELING_AUTOMATED]
        colour = "#ff7b72" if automated else "#7ee787"
        self.status_label.setText(
            f"Trial {trial}: <span style='color:{colour};'>{automated} automated</span> · "
            f"{counts[LABELING_CURATED]} curated · {counts[LABELING_MANUAL]} manual"
        )

    def sync_metadata(self) -> None:
        """Push the per-trial verdicts into the metadata ``curated`` column.

        Runs on :data:`METADATA_SYNC_MS`; a no-op unless a verdict differs
        from what the table holds, so the timer is cheap when idle.
        """
        if not self.app_state.ready or not self.app_state.trials:
            return
        trials_widget = self._trials_widget()
        if trials_widget is None:
            return
        status = self.app_state.trial_curation_status()
        mdf = getattr(self.app_state, "metadata_df", None)
        if mdf is None:
            mdf = getattr(trials_widget, "_metadata_df", None)
        if mdf is None or mdf.empty:
            return
        if not curated_column_differs(mdf, status):
            return
        trials_widget.set_column_values(CURATED_COLUMN, {t: int(v) for t, v in status.items()})

    # ------------------------------------------------------------------
    # Trial changes
    # ------------------------------------------------------------------

    def _on_trial_changed(self) -> None:
        if not self.app_state.ready:
            return
        if self.mode() == "inspect":
            # Deferred: the trial-change cascade (data load, label view) must
            # settle before the trial's labels are restamped and redrawn.
            QTimer.singleShot(0, lambda: self.curate_current_trial(quiet=True))
        self._refresh_status()
        if self._session_active and not self._jumping:
            self._follow_trial()

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------

    def open_grid_view(self) -> None:
        from ethograph.gui.dialog_label_gridview import LabelGridViewDialog

        if self.meta is None:
            return
        if self._grid_dialog is None or not self._grid_dialog.isVisible():
            self._grid_dialog = LabelGridViewDialog(self.meta, parent=self.window(), label_ids=self.scope_or_all_ids())
        self._grid_dialog.show()
        self._grid_dialog.raise_()
        self._grid_dialog.activateWindow()

    def open_video_grid(self) -> None:
        from ethograph.gui.dialog_video_grid import VideoGridDialog

        if self.meta is None:
            return
        if self._video_dialog is None or not self._video_dialog.isVisible():
            self._video_dialog = VideoGridDialog(self.meta, parent=self.window(), label_ids=self.scope_or_all_ids())
        self._video_dialog.show()
        self._video_dialog.raise_()
        self._video_dialog.activateWindow()

    # ==================================================================
    # Frame-by-frame review session
    # ==================================================================

    @property
    def session_active(self) -> bool:
        return self._session_active

    @property
    def targets(self) -> list[ReviewTarget]:
        return self._targets

    @property
    def current_index(self) -> int:
        return self._idx

    def _video_ready(self) -> bool:
        if getattr(self.app_state, "video", None) is None:
            notify("Frame-by-frame review needs a loaded video.", severity="warning")
            return False
        return True

    def _allowed_trials(self) -> set[str] | None:
        if self.nav is None:
            return None
        return self.nav._visible_trials()

    def build_queue(self) -> list[ReviewTarget]:
        """Every boundary of the labels in scope, in the trials the table shows."""
        return build_review_queue(self.app_state._all_labels_df, self.scope(), allowed_trials=self._allowed_trials())

    def _toggle_session(self) -> None:
        if self._session_active:
            self._stop()
        else:
            self.start_review()

    def start_review(self, idx: int = 0) -> bool:
        """Walk the scope's boundaries from *idx*. Returns whether a session started."""
        if not self._video_ready():
            return False
        targets = self.build_queue()
        if not targets:
            notify("No labels in scope to review.", severity="warning")
            return False
        self._begin(targets, idx)
        return True

    def start_review_at(self, inst: dict, field: str = "point") -> bool:
        """Drop into the review at *inst* (a grid tile): the scope's queue if
        the label is in it, else a one-label queue."""
        if self.mode() != "frame":
            idx = self.mode_combo.findData("frame")
            self.mode_combo.setCurrentIndex(idx)
        if not self._video_ready():
            return False
        targets = self.build_queue()
        idx = queue_index_of(targets, inst, field)
        if idx is None:
            targets = targets_from_seeds([{**inst, "field": field}])
            idx = queue_index_of(targets, inst, field) or 0
        self._begin(targets, idx)
        return True

    def start_from_seeds(self, seeds: list[dict]) -> bool:
        """Review exactly *seeds* (one boundary each) and nothing else."""
        if not self._video_ready():
            return False
        targets = targets_from_seeds(seeds)
        if not targets:
            notify("Nothing to review.", severity="warning")
            return False
        self._begin(targets, 0)
        return True

    def _begin(self, targets: list[ReviewTarget], idx: int) -> None:
        if self._session_active:
            self._teardown()
        self._targets = targets
        self._idx = min(max(idx, 0), len(targets) - 1)
        self._n_confirmed = 0
        self._n_deleted = 0
        self._session_active = True
        self._advance_pending = False
        self.start_stop_btn.setText("Stop review")
        self._install_session_shortcuts()
        self._jump_current()

    def _stop(self, done: bool = False) -> None:
        n_confirmed, n_deleted = self._n_confirmed, self._n_deleted
        self._teardown()
        self.start_stop_btn.setText("Start review")
        self.target_label.setText("")
        self.info_label.setText("")
        self.delta_label.setText("")
        if not (n_confirmed or n_deleted):
            return
        parts = []
        if n_confirmed:
            parts.append(f"confirmed {n_confirmed} boundar{'y' if n_confirmed == 1 else 'ies'}")
        if n_deleted:
            parts.append(f"deleted {n_deleted} event{'' if n_deleted == 1 else 's'}")
        notify(f"{'Done' if done else 'Stopped'} — {' and '.join(parts)}. Save with Ctrl+S.")

    def _teardown(self) -> None:
        self._session_active = False
        self._advance_pending = False
        self._seed_frame = None
        self._remove_session_shortcuts()

    # ------------------------------------------------------------------
    # Session shortcuts: Enter, Backspace/Delete, B, N
    # ------------------------------------------------------------------

    def _install_session_shortcuts(self) -> None:
        """Bind the verdict keys application-wide for the session's life.

        Disabled while a text field has focus, exactly like the shell's
        guarded shortcuts — an enabled QShortcut would swallow the key before
        the field sees it (see gui/shortcuts.py).
        """
        if self._session_shortcuts:
            return
        bindings = [
            (Qt.Key_Return, self._confirm),
            (Qt.Key_Enter, self._confirm),
            (Qt.Key_Backspace, self._delete_current),
            (Qt.Key_Delete, self._delete_current),
            (Qt.Key_B, self._back),
            (Qt.Key_N, self._next),
        ]
        for key, slot in bindings:
            shortcut = QShortcut(QKeySequence(key), self.window())
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(slot)
            self._session_shortcuts.append(shortcut)
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._sync_session_shortcuts)
        self._sync_session_shortcuts()

    def _sync_session_shortcuts(self, *_args) -> None:
        enabled = not typing_in_text_field()
        for shortcut in self._session_shortcuts:
            shortcut.setEnabled(enabled)

    def _remove_session_shortcuts(self) -> None:
        if not self._session_shortcuts:
            return
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.disconnect(self._sync_session_shortcuts)
        for shortcut in self._session_shortcuts:
            shortcut.setEnabled(False)
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._session_shortcuts = []

    def _show_shortcuts(self) -> None:
        if self._shortcuts_popup is None or not self._shortcuts_popup.isVisible():
            self._shortcuts_popup = ShortcutsPopup(self.window())
        self._shortcuts_popup.show()
        self._shortcuts_popup.raise_()

    # ------------------------------------------------------------------
    # Jumping + display
    # ------------------------------------------------------------------

    def _seed_rel(self, target: ReviewTarget) -> float:
        return target.inst["offset_s"] if target.field == "end" else target.inst["onset_s"]

    def _global_row_idx(self, inst: dict) -> int | None:
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty:
            return None
        mask = (df["trial"].astype(str) == str(inst["trial"])) & row_mask(df, inst)
        pos = np.flatnonzero(mask.to_numpy())
        return int(pos[0]) if len(pos) else None

    def _view_rel(self, seed_rel: float) -> TimeRange:
        """The seed-centred view window, slid (not shrunk) to stay in the trial."""
        size = float(self.app_state.get_with_default("refine_window_s"))
        half = size / 2.0
        t0, t1 = seed_rel - half, seed_rel + half
        tb = self.app_state.trial_bounds
        if tb is not None:
            if t0 < tb.start_s:
                t0, t1 = tb.start_s, min(tb.end_s, tb.start_s + size)
            elif t1 > tb.end_s:
                t0, t1 = max(tb.start_s, tb.end_s - size), tb.end_s
        return TimeRange(t0, t1)

    def _on_window_changed(self, value: float) -> None:
        self.app_state.refine_window_s = value
        if self._session_active and self.lock_checkbox.isChecked() and self.nav is not None:
            target = self._targets[self._idx]
            self.nav.set_view_range(target.inst["trial"], self._view_rel(self._seed_rel(target)))

    def _on_lock_toggled(self, checked: bool) -> None:
        if not self._session_active:
            return
        if checked:
            self._jump_current()
        else:
            self._free_navigation()

    def _free_navigation(self) -> None:
        """Widen the restriction to the normal navigation scope."""
        if self.nav is None:
            return
        self.nav._apply_slider_scope()
        pc = self.nav.plot_container
        if pc is not None:
            pc._apply_all_zoom_constraints()

    def _follow_trial(self) -> None:
        """Normal trial navigation pulls the session to that trial's first boundary."""
        trial = str(getattr(self.app_state, "trials_sel", None))
        if str(self._targets[self._idx].inst["trial"]) == trial:
            return
        for i, target in enumerate(self._targets):
            if str(target.inst["trial"]) == trial:
                self._advance_pending = False
                QTimer.singleShot(0, lambda i=i: self._follow_trial_jump(i))
                return

    def _follow_trial_jump(self, i: int) -> None:
        if not self._session_active:
            return
        self._idx = i
        self._jump_current()

    def _jump_current(self) -> None:
        target = self._targets[self._idx]
        inst = target.inst
        seed_rel = self._seed_rel(target)
        if self.nav is not None:
            self._jumping = True
            try:
                self.nav.jump_to_label_instance(
                    {**inst, "row_idx": self._global_row_idx(inst)},
                    seek_rel=seed_rel,
                    play=False,
                    view_rel=self._view_rel(seed_rel),
                )
                if not self.lock_checkbox.isChecked():
                    self._free_navigation()
            finally:
                self._jumping = False
        video = getattr(self.app_state, "video", None)
        seed_display = self.app_state.to_display(inst["trial"], seed_rel)
        self._seed_frame = video.time_to_frame(seed_display, round_nearest=True) if video else None
        self._update_target_display()
        self._update_delta()

    def _update_target_display(self) -> None:
        target = self._targets[self._idx]
        inst = target.inst
        mappings = getattr(self.labels_widget, "_mappings", {}) or {}
        mapping = mappings.get(inst["labels"], {})
        name = mapping.get("name", str(inst["labels"]))
        self.target_label.setText(f"{name} ({inst['labels']}) — {_FIELD_TITLES[target.field]}")
        self.target_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {_color_hex(mapping)};")
        parts = [f"{self._idx + 1} / {len(self._targets)}", f"trial {inst['trial']}"]
        individual = inst.get("individual")
        if individual is not None and not (isinstance(individual, float) and math.isnan(individual)):
            parts.append(str(individual))
        method = self._current_method()
        if method:
            parts.append(method)
        self.info_label.setText("  ·  ".join(parts))

    def _current_method(self) -> str | None:
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty or "labeling_method" not in df.columns:
            return None
        inst = self._targets[self._idx].inst
        mask = (df["trial"].astype(str) == str(inst["trial"])) & row_mask(df, inst)
        rows = df.loc[mask, "labeling_method"]
        return str(rows.iloc[0]) if len(rows) else None

    def _on_frame_changed(self, *_args) -> None:
        if self._session_active:
            self._update_delta()

    def _update_delta(self) -> None:
        if self._seed_frame is None:
            self.delta_label.setText("")
            return
        delta = int(getattr(self.app_state, "current_frame", self._seed_frame)) - self._seed_frame
        self.delta_label.setText(f"moved {delta:+d} frames")

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------

    def _current_row(self):
        """(df, row index) of the current target in ``label_intervals``, or None."""
        df = self.app_state.label_intervals
        if df is None or df.empty:
            notify("No labels in the current trial.", severity="warning")
            return None
        mask = row_mask(df, self._targets[self._idx].inst)
        pos = np.flatnonzero(mask.to_numpy())
        if not len(pos):
            notify("This label no longer exists (edited elsewhere?) — use N to skip.", severity="warning")
            return None
        return df, df.index[pos[0]]

    def _redraw_after_edit(self) -> None:
        if self.data_widget is not None:
            self.data_widget.update_main_plot(preserve_x_range=True)
        if self.labels_widget is not None:
            self.labels_widget.refresh_labels_shapes_layer()

    def _confirm(self) -> None:
        """Enter: the frame on screen is the boundary.

        A boundary that moved makes the label **manual** (and fully confident
        — a hand placed it); one confirmed where it stood is **curated**.
        """
        if not self._session_active or self._advance_pending:
            return
        target = self._targets[self._idx]
        inst = target.inst
        video = getattr(self.app_state, "video", None)
        if video is None:
            notify("No video loaded.", severity="warning")
            return
        t_display = video.frame_to_time(int(self.app_state.current_frame))
        hit = self.app_state.from_display(t_display)
        if hit is None:
            notify("The current frame lies outside any trial.", severity="warning")
            return
        trial_id, t_rel = hit
        if str(trial_id) != str(inst["trial"]):
            notify(
                f"The playhead is in trial {trial_id}, but this boundary belongs to trial {inst['trial']} — "
                "use N, or navigate back.",
                severity="warning",
            )
            return
        if target.field == "start" and t_rel >= inst["offset_s"]:
            notify("The start must stay before the end of the label.", severity="warning")
            return
        if target.field == "end" and t_rel <= inst["onset_s"]:
            notify("The end must stay after the start of the label.", severity="warning")
            return
        found = self._current_row()
        if found is None:
            return
        df, row_idx = found
        df = ensure_labeling_method(df)
        old_onset, old_offset = _num(inst["onset_s"]), _num(inst["offset_s"])

        if target.field == "end":
            df.loc[row_idx, "offset_s"] = t_rel
            inst["offset_s"] = t_rel
        else:
            df.loc[row_idx, "onset_s"] = t_rel
            if target.field == "point" and math.isfinite(inst["offset_s"]):
                df.loc[row_idx, "offset_s"] = t_rel
                inst["offset_s"] = t_rel
            inst["onset_s"] = t_rel
        moved = not (_close(old_onset, _num(inst["onset_s"])) and _close(old_offset, _num(inst["offset_s"])))
        if moved:
            df.loc[row_idx, "confidence"] = HUMAN_CONFIDENCE
            df.loc[row_idx, "labeling_method"] = LABELING_MANUAL
        elif df.loc[row_idx, "labeling_method"] == LABELING_AUTOMATED:
            df.loc[row_idx, "labeling_method"] = LABELING_CURATED

        self.app_state.record_label_edit("confirm boundary", trial=trial_id)
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(trial_id, df)
        self.app_state.changes_saved = False
        self._n_confirmed += 1
        self._redraw_after_edit()
        self.app_state.curation_changed.emit()
        self._refresh_status()

        self._advance_pending = True
        self.delta_label.setText("✓ placed")
        QTimer.singleShot(_CONFIRM_PAUSE_MS, self._advance_after_pause)

    def _delete_current(self) -> None:
        """Backspace: this event does not belong in the trial — drop the label."""
        if not self._session_active or self._advance_pending:
            return
        inst = self._targets[self._idx].inst
        trial = inst["trial"]
        current = getattr(self.app_state, "trials_sel", None)
        if current is not None and str(current) != str(trial):
            notify(
                f"The GUI is on trial {current}, but this boundary belongs to trial {trial} — navigate back first.",
                severity="warning",
            )
            return
        found = self._current_row()
        if found is None:
            return
        df, row_idx = found
        self.app_state.record_label_edit("delete event", trial=trial)
        df = delete_interval(df, row_idx)
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(trial, df)
        self.app_state.changes_saved = False
        self._n_deleted += 1
        self._redraw_after_edit()
        self.app_state.curation_changed.emit()
        self._refresh_status()
        self.delta_label.setText("✗ deleted")
        self._advance_past(inst)

    def _next(self) -> None:
        """N: leave this boundary — curating it when the checkbox says so."""
        if not self._session_active or self._advance_pending:
            return
        if self.next_curates_cb.isChecked():
            inst = self._targets[self._idx].inst
            df, n = curate_label(self.app_state._all_labels_df, inst)
            self._commit(df, n, restyle=(inst, False))
        self._advance(+1)

    def _back(self) -> None:
        if not self._session_active:
            return
        self._advance(-1)

    def _advance_past(self, inst: dict) -> None:
        """Jump to the next target that is not part of *inst* (whose row is gone)."""
        idx = self._idx + 1
        while idx < len(self._targets) and self._targets[idx].inst is inst:
            idx += 1
        if idx >= len(self._targets):
            self._stop(done=True)
            return
        self._idx = idx
        self._jump_current()

    def _advance_after_pause(self) -> None:
        if not self._advance_pending or not self._session_active:
            return  # Stop / B / N intervened during the pause
        self._advance(+1)

    def _advance(self, direction: int) -> None:
        self._advance_pending = False
        new_idx = self._idx + direction
        if new_idx >= len(self._targets):
            self._stop(done=True)
            return
        self._idx = max(0, new_idx)
        self._jump_current()
