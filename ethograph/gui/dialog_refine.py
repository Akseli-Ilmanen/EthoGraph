"""Frame-by-frame refinement of existing labels (Tools ▸ Refine labels frame-by-frame…).

EthoGraph normally places labels on the time-series plots because hunting for
the exact frame is slow. When frame accuracy matters, this dialog turns the
labels already placed into *seeds*: the user picks which label classes to
refine, and the dialog walks a queue of boundaries — one seed per point
event, a start seed then an end seed per state event — ordered by (trial,
onset) so each trial is visited once. Every seed is reached through the
ordinary label-jump navigation (:meth:`NavigationWidget.jump_to_label_instance`),
the boundary being edited is named in large coloured text, the global ←/→
shortcuts (or the dialog's ◀/▶ buttons) nudge the video one frame at a time,
and **Enter** commits the frame on screen as the new boundary time and jumps
straight to the next seed. **Backspace** is the other verdict: this event does
not belong in the trial at all, so its label is deleted and the queue moves on.

The queue normally comes from the label classes ticked here, but the frames
grid (:mod:`ethograph.gui.dialog_label_frames`) can hand one over instead —
tick the tiles that look wrong there and only those boundaries are visited
(:meth:`RefineLabelsDialog.start_from_seeds`). Every commit stamps the row's
``confidence`` back to :data:`~ethograph.labels.intervals.HUMAN_CONFIDENCE`:
a boundary placed by eye is a hand-made label, whatever produced it first.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeySequence, QShortcut, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableView,
    QVBoxLayout,
)

from ethograph.gui.file_dialogs import browse_save_file
from ethograph.gui.notify import notify
from ethograph.gui.shortcuts import typing_in_text_field
from ethograph.gui.table_filter import CategoryFilterDialog, FilterHeaderView, MultiColumnFilterProxy
from ethograph.io.time_model import TimeRange
from ethograph.labels.intervals import (
    EVENT_TYPE_POINT,
    HUMAN_CONFIDENCE,
    delete_interval,
    ensure_confidence,
)

logger = logging.getLogger(__name__)

_FIELD_TITLES = {"point": "POINT", "start": "START", "end": "END"}

#: Visit order of the boundaries of one label: START before END.
_FIELD_RANK = {"point": 0, "start": 0, "end": 1}

#: Linger on a just-committed boundary this long before jumping to the next
#: seed, so the user sees the label land where they put it.
_CONFIRM_PAUSE_MS = 100


@dataclass
class _Target:
    """One boundary to refine.

    ``inst`` is shared between the start and end targets of the same state
    event, so committing a new start updates the onset the end target (and the
    row lookup) will use.
    """

    inst: dict
    field: str  # "point" | "start" | "end"


def _num(value) -> float | None:
    """A finite float, or None (YAML-safe stand-in for NaN offsets)."""
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _close(a: float | None, b: float | None, atol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= atol


def _subject_str(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _log_identity(inst: dict) -> dict:
    """The columns identifying one label row in ``app_state.refine_log``."""
    return {
        "trial": str(inst["trial"]),
        "labels": int(inst["labels"]),
        "individual": _subject_str(inst.get("individual")),
        "individual_rec": _subject_str(inst.get("individual_rec")),
    }


def targets_from_seeds(seeds: list[dict]) -> list["_Target"]:
    """Build a seed queue from boundaries chosen elsewhere (the frames grid).

    Each seed is a label row (``trial``, ``labels``, ``onset_s``, ``offset_s``,
    the subject columns, ``event_type``) plus the ``field`` to edit. Sorted
    (trial, onset) with START before END, so a trial is visited once; the two
    boundaries of one state event share a single ``inst`` — exactly as in
    :meth:`RefineLabelsDialog._build_queue`, so committing a new start is seen
    by the end target.
    """
    ordered = sorted(
        seeds,
        key=lambda s: (str(s["trial"]), float(s["onset_s"]), _FIELD_RANK.get(s.get("field", "point"), 0)),
    )
    insts: dict[tuple, dict] = {}
    targets: list[_Target] = []
    for seed in ordered:
        key = (
            str(seed["trial"]),
            int(seed["labels"]),
            round(float(seed["onset_s"]), 6),
            _subject_str(seed.get("individual")),
            _subject_str(seed.get("individual_rec")),
        )
        inst = insts.get(key)
        if inst is None:
            offset = seed.get("offset_s")
            inst = {
                "trial": seed["trial"],
                "labels": int(seed["labels"]),
                "onset_s": float(seed["onset_s"]),
                "offset_s": float(offset) if offset is not None else float("nan"),
                "individual": seed.get("individual"),
                "individual_rec": seed.get("individual_rec"),
                "event_type": seed.get("event_type", "state"),
            }
            insts[key] = inst
        targets.append(_Target(inst, seed.get("field", "point")))
    return targets


def _fmt_bounds(onset: float | None, offset: float | None) -> str:
    if onset is None:
        return ""
    if offset is None:
        return f"{onset:.3f} s"
    return f"{onset:.3f}–{offset:.3f} s"


_EXPORT_COLS = ["trial", "labels", "name", "individual", "individual_rec", "event_type"]


def export_refine_log(log: list[dict], base_path: str | Path) -> tuple[Path, Path]:
    """Write the refined boundaries as two label-shaped TSVs.

    ``{base}_prerefined.tsv`` carries each refined row's original
    onset/offset, ``{base}_postrefined.tsv`` its latest refined values —
    same row order, so the files diff line-for-line. A deleted event has no
    refined times: its post row is blank and its ``deleted`` column is True.
    """
    stem = Path(base_path).with_suffix("")

    def frame(which: str) -> pd.DataFrame:
        rows = [
            {
                "onset_s": rec.get(f"{which}_onset_s"),
                "offset_s": rec.get(f"{which}_offset_s"),
                **{col: rec.get(col) for col in _EXPORT_COLS},
                "deleted": bool(rec.get("deleted", False)),
                "refined_time": rec.get("time"),
            }
            for rec in log
        ]
        return pd.DataFrame(rows, columns=["onset_s", "offset_s", *_EXPORT_COLS, "deleted", "refined_time"])

    pre_path = stem.parent / f"{stem.name}_prerefined.tsv"
    post_path = stem.parent / f"{stem.name}_postrefined.tsv"
    frame("orig").to_csv(pre_path, sep="\t", index=False)
    frame("new").to_csv(post_path, sep="\t", index=False)
    return pre_path, post_path


class RefineHistoryDialog(QDialog):
    """Read-only table of every refined boundary, with TSV export.

    Trial / Label / Individual / Boundary carry the shared funnel-header
    filters (``gui/table_filter.py``), so a subset can be inspected — and
    the export writes exactly the rows the filters leave visible.
    """

    _COLUMNS = ["Trial", "Label", "Individual", "Boundary", "Original", "Refined", "When"]
    _FILTER_COLS = {0, 1, 2, 3}

    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Refinement history")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)
        self.app_state = app_state

        layout = QVBoxLayout(self)
        self._model = QStandardItemModel(0, len(self._COLUMNS), self)
        self._model.setHorizontalHeaderLabels(self._COLUMNS)
        self._proxy = MultiColumnFilterProxy(self)
        self._proxy.setSourceModel(self._model)
        self.table = QTableView()
        self.table.setModel(self._proxy)
        self._header = FilterHeaderView(self._FILTER_COLS, set(), self.table)
        self._header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self._header.filter_requested.connect(self._on_filter_clicked)
        self.table.setHorizontalHeader(self._header)
        self.table.setEditTriggers(QTableView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.export_btn = QPushButton("Export TSVs…")
        self.export_btn.setAutoDefault(False)
        self.export_btn.setToolTip("Exports the rows the filters currently leave visible")
        self.export_btn.clicked.connect(self._export)
        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.export_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self.refresh()
        self.resize(680, 360)

    def refresh(self):
        log = getattr(self.app_state, "refine_log", None) or []
        self._model.setRowCount(0)
        for i, rec in enumerate(log):
            values = [
                str(rec.get("trial", "")),
                f"{rec.get('name', '')} ({rec.get('labels', '')})",
                str(rec.get("individual", "")),
                "/".join(rec.get("fields", [])),
                _fmt_bounds(rec.get("orig_onset_s"), rec.get("orig_offset_s")),
                "deleted" if rec.get("deleted") else _fmt_bounds(rec.get("new_onset_s"), rec.get("new_offset_s")),
                str(rec.get("time", "")),
            ]
            items = []
            for text in values:
                item = QStandardItem(text)
                item.setEditable(False)
                items.append(item)
            # The log index rides on the row so the export can map the
            # proxy's visible rows back to records.
            items[0].setData(i, Qt.UserRole)
            self._model.appendRow(items)
        self.export_btn.setEnabled(bool(log))

    def _on_filter_clicked(self, col: int):
        values = sorted({self._model.item(r, col).text() for r in range(self._model.rowCount())})
        dialog = CategoryFilterDialog(col, values, self._proxy.cat_filter(col), self)
        if dialog.exec_() == QDialog.Accepted:
            self._proxy.set_cat_filter(col, dialog.get_allowed())
            self._header.set_active_filters(self._proxy.active_filters())

    def _visible_log(self) -> list[dict]:
        """The records the filters currently leave visible, in table order."""
        log = getattr(self.app_state, "refine_log", None) or []
        visible = []
        for row in range(self._proxy.rowCount()):
            source = self._proxy.mapToSource(self._proxy.index(row, 0))
            visible.append(log[self._model.item(source.row(), 0).data(Qt.UserRole)])
        return visible

    def _export(self):
        log = self._visible_log()
        if not log:
            notify("The filters leave nothing to export.", severity="warning")
            return
        labels_path = self.app_state.labels_file_path()
        default = f"{labels_path.stem}_refined.tsv" if labels_path else "labels_refined.tsv"
        path = browse_save_file(
            self,
            self.app_state,
            "Export refinement TSVs (base name — _prerefined/_postrefined are appended)",
            default,
            "TSV files (*.tsv)",
            preferred_dir=labels_path,
        )
        if not path:
            return
        pre_path, post_path = export_refine_log(log, path)
        notify(f"Wrote {pre_path.name} and {post_path.name}")


def _row_mask(df: pd.DataFrame, inst: dict) -> pd.Series:
    """Locate *inst*'s row in a labels DataFrame (trial-relative times).

    Rows are matched by class + subject + onset rather than by index: every
    ``set_trial_intervals`` rebuilds the trial's rows, so indices don't
    survive an edit but the values just written do.
    """
    mask = df["labels"] == inst["labels"]
    mask &= np.isclose(df["onset_s"].astype(float), float(inst["onset_s"]), atol=1e-6)
    for col in ("individual", "individual_rec"):
        if col not in df.columns:
            continue
        val = inst.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            mask &= df[col].isna()
        else:
            mask &= df[col] == val
    return mask


class RefineLabelsDialog(QDialog):
    """Non-modal dialog driving the seed-by-seed refinement workflow.

    The queue comes from the label classes ticked here (**Start refining**) or
    from boundaries chosen elsewhere (:meth:`start_from_seeds`).
    """

    def __init__(self, meta, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Refine labels frame-by-frame")
        self.setWindowFlag(Qt.Window)
        self.setModal(False)

        self.meta = meta
        self.app_state = meta.app_state
        self.nav = meta.navigation_widget
        self.labels_widget = meta.labels_widget
        self.data_widget = meta.data_widget
        self.io_widget = meta.io_widget

        self._targets: list[_Target] = []
        self._idx = 0
        self._seed_frame: int | None = None
        self._n_refined = 0
        self._n_deleted = 0
        self._frame_conn = False
        #: Label classes the running queue covers — the ticked ones, or those
        #: the handed-over seeds happen to carry.
        self._session_label_ids: list[int] = []
        #: A queue handed over from outside cannot be rebuilt from the class
        #: list, so it is not written to ``refine_resume``.
        self._remember_resume = True
        #: True when the frames grid seeded this session — its tiles are stale
        #: once the queue is done, and the summary says so.
        self._from_grid = False
        #: True between a commit and the delayed jump to the next seed.
        self._advance_pending = False
        #: True while WE are navigating — the trial-follow handler must not
        #: react to trial changes our own jumps cause.
        self._jumping = False
        self._trial_conn = False
        #: True while a refine session runs (between _start and _stop/close) —
        #: an explicit flag, not widget visibility, which lies for an unshown
        #: dialog.
        self._session_active = False
        #: Application-wide Return/Enter → Confirm, alive only while a refine
        #: session runs: the dialog's default button needs dialog focus, and a
        #: user clicking plots between nudges loses exactly that.
        self._enter_shortcuts: list[QShortcut] = []
        #: Backspace/Delete → drop this event, same session scope.
        self._delete_shortcuts: list[QShortcut] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Setup: which labels, which individual ────────────────────
        self.setup_group = QGroupBox("Labels to refine")
        setup_lay = QVBoxLayout(self.setup_group)
        setup_lay.addWidget(QLabel("Tick the label classes whose boundaries you want to refine:"))
        self.label_list = QListWidget()
        mappings = getattr(self.labels_widget, "_mappings", {}) or {}
        for label_id, info in sorted(mappings.items(), key=lambda x: x[0]):
            if not isinstance(label_id, int) or label_id == 0:
                continue
            name = info.get("name", str(label_id))
            event_type = info.get("event_type", "state")
            item = QListWidgetItem(f"{label_id} — {name}  ({event_type})")
            item.setData(Qt.UserRole, label_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.label_list.addItem(item)
        setup_lay.addWidget(self.label_list)

        ind_row = QHBoxLayout()
        ind_row.addWidget(QLabel("Individual:"))
        self.individual_combo = QComboBox()
        self.individual_combo.addItem("All")
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is not None and "individual" in df.columns:
            for ind in sorted(df["individual"].dropna().unique()):
                self.individual_combo.addItem(str(ind))
        ind_row.addWidget(self.individual_combo, stretch=1)
        setup_lay.addLayout(ind_row)

        filter_hint = QLabel("Only trials visible in the trials table are visited.")
        filter_hint.setStyleSheet("color: grey; font-size: 10px;")
        setup_lay.addWidget(filter_hint)

        self.start_btn = QPushButton("Start refining")
        self.start_btn.setAutoDefault(False)
        self.start_btn.clicked.connect(lambda: self._start())
        setup_lay.addWidget(self.start_btn)

        self.resume_btn = QPushButton("Resume last session")
        self.resume_btn.setAutoDefault(False)
        self.resume_btn.clicked.connect(self._resume)
        setup_lay.addWidget(self.resume_btn)
        self._history_dialog: RefineHistoryDialog | None = None
        self._sync_resume_button()
        layout.addWidget(self.setup_group)

        # ── Refine: the active seed ──────────────────────────────────
        self.refine_group = QGroupBox("Refining")
        refine_lay = QVBoxLayout(self.refine_group)

        self.target_label = QLabel("")
        self.target_label.setAlignment(Qt.AlignCenter)
        self.target_label.setTextFormat(Qt.PlainText)
        refine_lay.addWidget(self.target_label)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        refine_lay.addWidget(self.info_label)

        self.delta_label = QLabel("")
        self.delta_label.setAlignment(Qt.AlignCenter)
        refine_lay.addWidget(self.delta_label)

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("View window:"))
        self.window_spin = QDoubleSpinBox()
        self.window_spin.setRange(0.02, 600.0)
        self.window_spin.setDecimals(2)
        self.window_spin.setSingleStep(0.2)
        self.window_spin.setSuffix(" s")
        self.window_spin.setToolTip(
            "Seconds of time series shown around the boundary being refined\n"
            "(seed centred) — independent of the navigation Before/After padding."
        )
        self.window_spin.setValue(float(self.app_state.get_with_default("refine_window_s")))
        self.window_spin.valueChanged.connect(self._on_window_changed)
        # Enter in the spinbox commits the value and hands focus back, so the
        # NEXT Enter is a boundary confirm again — one keypress, one meaning.
        self.window_spin.editingFinished.connect(self.window_spin.clearFocus)
        win_row.addWidget(self.window_spin, stretch=1)
        self.lock_checkbox = QCheckBox("Locked around initial label")
        self.lock_checkbox.setChecked(True)
        self.lock_checkbox.setToolTip(
            "Ticked: the view stays a small window around the seed.\n"
            "Unticked: pan/zoom the whole trial freely — for corrections that\n"
            "belong far from where the label currently sits. Enter still\n"
            "confirms the frame on screen, wherever you navigated."
        )
        self.lock_checkbox.toggled.connect(self._on_lock_toggled)
        win_row.addWidget(self.lock_checkbox)
        refine_lay.addLayout(win_row)

        step_row = QHBoxLayout()
        self.step_back_btn = QPushButton("◀  1 frame")
        self.step_back_btn.setAutoDefault(False)
        self.step_back_btn.clicked.connect(self.nav.step_frame_backward)
        self.step_fwd_btn = QPushButton("1 frame  ▶")
        self.step_fwd_btn.setAutoDefault(False)
        self.step_fwd_btn.clicked.connect(self.nav.step_frame_forward)
        step_row.addWidget(self.step_back_btn)
        step_row.addWidget(self.step_fwd_btn)
        refine_lay.addLayout(step_row)

        confirm_row = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.back_btn.setAutoDefault(False)
        self.back_btn.clicked.connect(lambda: self._advance(-1))
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setAutoDefault(False)
        self.skip_btn.clicked.connect(lambda: self._advance(+1))
        self.confirm_btn = QPushButton("Confirm (Enter)")
        # Enter → confirm comes ONLY from the session-scoped application
        # shortcut. A default/autoDefault button also catches the Return a
        # spinbox ignores after committing its edit, so typing a view window
        # and pressing Enter confirmed a boundary as a side effect.
        self.confirm_btn.setDefault(False)
        self.confirm_btn.setAutoDefault(False)
        self.confirm_btn.clicked.connect(self._confirm)
        confirm_row.addWidget(self.back_btn)
        confirm_row.addWidget(self.skip_btn)
        confirm_row.addWidget(self.confirm_btn, stretch=1)
        refine_lay.addLayout(confirm_row)

        self.delete_btn = QPushButton("Delete this event (Backspace)")
        self.delete_btn.setAutoDefault(False)
        self.delete_btn.setToolTip(
            "This event should not exist in this trial at all: delete the label\n"
            "without placing a replacement and jump to the next seed.\n"
            "Nothing reaches disk until you save (Ctrl+S)."
        )
        self.delete_btn.clicked.connect(self._delete_current)
        refine_lay.addWidget(self.delete_btn)

        hint = QLabel(
            "←/→ step one frame · Enter confirms (works anywhere) and jumps to the next seed\n"
            "Backspace deletes an event that should not exist · Space plays"
        )
        hint.setStyleSheet("color: grey; font-size: 10px;")
        hint.setAlignment(Qt.AlignCenter)
        refine_lay.addWidget(hint)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setAutoDefault(False)
        self.stop_btn.clicked.connect(lambda: self._stop())
        refine_lay.addWidget(self.stop_btn)

        self.refine_group.setVisible(False)
        layout.addWidget(self.refine_group)

        # History lives OUTSIDE both groups — which trials are already done is
        # a question asked mid-session at least as often as before one.
        self.history_btn = QPushButton("History…")
        self.history_btn.setAutoDefault(False)
        self.history_btn.setToolTip("Every boundary refined so far, with TSV export")
        self.history_btn.clicked.connect(self._open_history)
        layout.addWidget(self.history_btn)

        self.resize(420, 480)

    # ==================================================================
    # Queue building
    # ==================================================================

    def _selected_label_ids(self) -> list[int]:
        ids = []
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item.checkState() == Qt.Checked:
                ids.append(int(item.data(Qt.UserRole)))
        return ids

    def _build_queue(self, label_ids: list[int]) -> list[_Target]:
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty:
            return []
        mask = df["labels"].isin(label_ids)
        individual = self.individual_combo.currentText()
        if individual != "All" and "individual" in df.columns:
            mask &= df["individual"].astype(str) == individual
        rows = df[mask].sort_values(["trial", "onset_s"])

        insts = [
            {
                "trial": row["trial"],
                "labels": int(row["labels"]),
                "onset_s": float(row["onset_s"]),
                "offset_s": float(row["offset_s"]),
                "individual": row.get("individual"),
                "individual_rec": row.get("individual_rec"),
                "event_type": row.get("event_type", "state"),
            }
            for _, row in rows.iterrows()
        ]
        insts = self.nav._only_visible_trials(insts)

        targets: list[_Target] = []
        for inst in insts:
            is_point = inst["event_type"] == EVENT_TYPE_POINT or not math.isfinite(inst["offset_s"])
            if is_point:
                targets.append(_Target(inst, "point"))
            else:
                targets.append(_Target(inst, "start"))
                targets.append(_Target(inst, "end"))
        return targets

    # ==================================================================
    # Session control
    # ==================================================================

    def _start(self, resume: dict | None = None):
        if not self._video_ready():
            return
        label_ids = self._selected_label_ids()
        if not label_ids:
            notify("Tick at least one label class to refine.", severity="warning")
            return
        targets = self._build_queue(label_ids)
        if not targets:
            notify("No labels of the selected classes found.", severity="warning")
            return
        # _resume_index reads the queue, so it has to be in place first.
        self._targets = targets
        self._begin_session(targets, label_ids=label_ids, idx=self._resume_index(resume) if resume else 0)

    def start_from_seeds(self, seeds: list[dict], *, from_grid: bool = False) -> bool:
        """Refine exactly *seeds* — one boundary each — instead of whole classes.

        This is how the frames grid hands over the tiles the user ticked: the
        queue holds those boundaries and nothing else, so neither the class
        list here nor the trials-table filter narrows it further. Returns
        whether a session started.
        """
        if not self._video_ready():
            return False
        if self._session_active:
            # A running queue is replaced, not merged — report what it did.
            self._stop()
        targets = targets_from_seeds(seeds)
        if not targets:
            notify("Nothing selected to refine.", severity="warning")
            return False
        label_ids = sorted({target.inst["labels"] for target in targets})
        # Mirror the queue in the class list, so Stop leaves the setup panel
        # describing what was just refined.
        wanted = set(label_ids)
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            item.setCheckState(Qt.Checked if int(item.data(Qt.UserRole)) in wanted else Qt.Unchecked)
        self._begin_session(targets, label_ids=label_ids, remember=False, from_grid=from_grid)
        return True

    def _video_ready(self) -> bool:
        if getattr(self.app_state, "video", None) is None:
            notify("Frame-by-frame refinement needs a loaded video.", severity="warning")
            return False
        return True

    def _begin_session(
        self,
        targets: list[_Target],
        *,
        label_ids: list[int],
        idx: int = 0,
        remember: bool = True,
        from_grid: bool = False,
    ):
        """Start walking *targets*, whatever built them."""
        self._targets = targets
        self._idx = min(max(idx, 0), len(targets) - 1)
        self._n_refined = 0
        self._n_deleted = 0
        self._session_label_ids = list(label_ids)
        self._remember_resume = remember
        self._from_grid = from_grid
        self._session_active = True
        self.setup_group.setVisible(False)
        self.refine_group.setVisible(True)
        if not self._frame_conn:
            self.app_state.current_frame_changed.connect(self._on_frame_changed)
            self._frame_conn = True
        if not self._trial_conn:
            self.app_state.trial_changed.connect(self._on_trial_changed)
            self._trial_conn = True
        self._install_session_shortcuts()
        self._jump_current()

    def _stop(self, done: bool = False):
        n_refined, n_deleted, from_grid = self._n_refined, self._n_deleted, self._from_grid
        self._teardown_session()
        self.refine_group.setVisible(False)
        self.setup_group.setVisible(True)
        self._sync_resume_button()
        if not (n_refined or n_deleted):
            return
        parts = []
        if n_refined:
            parts.append(f"refined {n_refined} boundar{'y' if n_refined == 1 else 'ies'}")
        if n_deleted:
            parts.append(f"deleted {n_deleted} event{'' if n_deleted == 1 else 's'}")
        message = f"{'Done' if done else 'Stopped'} — {' and '.join(parts)}. Save with Ctrl+S."
        if done and from_grid:
            message += " Then close the label-frames grid and generate it again to check the updated frames."
        notify(message)

    # ------------------------------------------------------------------
    # Resume + history
    # ------------------------------------------------------------------

    def _sync_resume_button(self):
        info = getattr(self.app_state, "refine_resume", None)
        self.resume_btn.setEnabled(bool(info))
        if info:
            self.resume_btn.setToolTip(
                f"Back to trial {info.get('trial')}, label {info.get('labels')} ({info.get('field')})"
            )
        else:
            self.resume_btn.setToolTip("No previous refinement session for this dataset")

    def _resume(self):
        """Rebuild the last session's queue and jump back to where it stood."""
        info = getattr(self.app_state, "refine_resume", None)
        if not info:
            return
        wanted = set(info.get("label_ids") or [])
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            item.setCheckState(Qt.Checked if int(item.data(Qt.UserRole)) in wanted else Qt.Unchecked)
        combo_idx = self.individual_combo.findText(str(info.get("individual", "All")))
        if combo_idx >= 0:
            self.individual_combo.setCurrentIndex(combo_idx)
        self._start(resume=info)

    def _resume_index(self, info: dict) -> int:
        """The queue position matching the remembered seed, or the closest fit.

        Matched on identity + onset (the value the seed holds NOW — resume
        stores post-edit onsets, and the rebuilt queue reads the same df), so
        already-refined boundaries re-match after a restart. Falls back to the
        first seed of the remembered trial, then the queue start.
        """
        for i, target in enumerate(self._targets):
            if (
                str(target.inst["trial"]) == str(info.get("trial"))
                and target.inst["labels"] == info.get("labels")
                and target.field == info.get("field")
                and _close(_num(target.inst["onset_s"]), _num(info.get("onset_s")), atol=1e-3)
            ):
                return i
        for i, target in enumerate(self._targets):
            if str(target.inst["trial"]) == str(info.get("trial")):
                return i
        return 0

    def _open_history(self):
        if self._history_dialog is None or not self._history_dialog.isVisible():
            self._history_dialog = RefineHistoryDialog(self.app_state, parent=self)
        else:
            self._history_dialog.refresh()
        self._history_dialog.show()
        self._history_dialog.raise_()

    def _teardown_session(self):
        self._session_active = False
        self._advance_pending = False
        self._from_grid = False
        if self._frame_conn:
            self.app_state.current_frame_changed.disconnect(self._on_frame_changed)
            self._frame_conn = False
        if self._trial_conn:
            self.app_state.trial_changed.disconnect(self._on_trial_changed)
            self._trial_conn = False
        self._remove_session_shortcuts()

    def closeEvent(self, event):
        self._teardown_session()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Global Enter → Confirm, Backspace → Delete (session-scoped)
    # ------------------------------------------------------------------

    def _install_session_shortcuts(self):
        """Bind the two verdict keys application-wide for the session's life.

        Disabled while a text field has focus, exactly like the shell's
        guarded shortcuts — an enabled QShortcut would swallow the key
        before the field sees it (see gui/shortcuts.py).
        """
        if self._session_shortcuts():
            return
        self._enter_shortcuts = [self._session_shortcut(key, self._confirm) for key in (Qt.Key_Return, Qt.Key_Enter)]
        self._delete_shortcuts = [
            self._session_shortcut(key, self._delete_current) for key in (Qt.Key_Backspace, Qt.Key_Delete)
        ]
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._sync_session_shortcuts)
        self._sync_session_shortcuts()

    def _session_shortcut(self, key, slot) -> QShortcut:
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(slot)
        return shortcut

    def _session_shortcuts(self) -> list[QShortcut]:
        return [*self._enter_shortcuts, *self._delete_shortcuts]

    def _sync_session_shortcuts(self, *_args):
        enabled = not typing_in_text_field()
        for shortcut in self._session_shortcuts():
            shortcut.setEnabled(enabled)

    def _remove_session_shortcuts(self):
        shortcuts = self._session_shortcuts()
        if not shortcuts:
            return
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.disconnect(self._sync_session_shortcuts)
        for shortcut in shortcuts:
            shortcut.setEnabled(False)
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._enter_shortcuts = []
        self._delete_shortcuts = []

    # ==================================================================
    # Jumping + display
    # ==================================================================

    def _seed_rel(self, target: _Target) -> float:
        return target.inst["offset_s"] if target.field == "end" else target.inst["onset_s"]

    def _global_row_idx(self, inst: dict) -> int | None:
        """Positional row of *inst* in ``_all_labels_df`` (for build_label_window)."""
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is None or df.empty:
            return None
        mask = (df["trial"].astype(str) == str(inst["trial"])) & _row_mask(df, inst)
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

    def _on_window_changed(self, value: float):
        self.app_state.refine_window_s = value
        if self._targets and self._session_active and self.lock_checkbox.isChecked():
            target = self._targets[self._idx]
            self.nav.set_view_range(target.inst["trial"], self._view_rel(self._seed_rel(target)))

    def _on_lock_toggled(self, checked: bool):
        """Locked: snap back to the seed-centred window. Unlocked: free roam."""
        if not (self._targets and self._session_active):
            return
        if checked:
            self._jump_current()
        else:
            self._free_navigation()

    def _free_navigation(self):
        """Widen the restriction to the normal navigation scope.

        The seed view stays as the starting point, but the user can pan/zoom
        the whole trial and seek anywhere — for corrections that belong far
        from where the label currently sits. Video and marker are untouched.
        """
        self.nav._apply_slider_scope()
        pc = self.nav.plot_container
        if pc is not None:
            pc._apply_all_zoom_constraints()

    def _on_trial_changed(self):
        """Follow the user's normal trial navigation to that trial's first seed.

        Deferred a tick so the trial-change cascade (data load, panel
        refresh) settles before we place the view; a trial with no seeds is
        left alone. Guarded against the trial changes our own jumps emit.
        """
        if not self._session_active or self._jumping:
            return
        trial = str(getattr(self.app_state, "trials_sel", None))
        if str(self._targets[self._idx].inst["trial"]) == trial:
            return
        for i, target in enumerate(self._targets):
            if str(target.inst["trial"]) == trial:
                self._advance_pending = False
                QTimer.singleShot(0, lambda i=i: self._follow_trial_jump(i))
                return

    def _follow_trial_jump(self, i: int):
        if not self._session_active:
            return
        self._idx = i
        self._jump_current()

    def _jump_current(self):
        target = self._targets[self._idx]
        inst = target.inst
        seed_rel = self._seed_rel(target)
        self._jumping = True
        try:
            self.nav.jump_to_label_instance(
                {**inst, "row_idx": self._global_row_idx(inst)},
                seek_rel=seed_rel,
                play=False,
                view_rel=self._view_rel(seed_rel),
            )
            if not self.lock_checkbox.isChecked():
                # Unlocked: the seed view is only the starting point — widen
                # the restriction so the user can immediately roam the trial.
                self._free_navigation()
        finally:
            self._jumping = False
        video = getattr(self.app_state, "video", None)
        seed_display = self.app_state.to_display(inst["trial"], seed_rel)
        self._seed_frame = video.time_to_frame(seed_display, round_nearest=True) if video else None
        # Remember where the session stands — every jump, not just commits,
        # so a Stop (or crash) mid-seed resumes exactly here. A queue handed
        # over from the frames grid is not rebuildable from the class list, so
        # it leaves the remembered session alone.
        if self._remember_resume:
            self.app_state.refine_resume = {
                "label_ids": list(self._session_label_ids),
                "individual": self.individual_combo.currentText(),
                "trial": str(inst["trial"]),
                "labels": int(inst["labels"]),
                "onset_s": float(inst["onset_s"]),
                "field": target.field,
            }
        self._update_target_display()
        self._update_delta()

    def _update_target_display(self):
        target = self._targets[self._idx]
        inst = target.inst
        mappings = getattr(self.labels_widget, "_mappings", {}) or {}
        mapping = mappings.get(inst["labels"], {})
        name = mapping.get("name", str(inst["labels"]))
        color = mapping.get("color")
        hex_color = "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in color[:3])) if color is not None else "#ffffff"
        self.target_label.setText(f"{name} ({inst['labels']}) — {_FIELD_TITLES[target.field]}")
        self.target_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {hex_color};")

        parts = [f"{self._idx + 1} / {len(self._targets)}", f"trial {inst['trial']}"]
        individual = inst.get("individual")
        if individual is not None and not (isinstance(individual, float) and math.isnan(individual)):
            parts.append(str(individual))
        self.info_label.setText("  ·  ".join(parts))

    def _on_frame_changed(self, *_args):
        self._update_delta()

    def _update_delta(self):
        if self._seed_frame is None:
            self.delta_label.setText("")
            return
        delta = int(getattr(self.app_state, "current_frame", self._seed_frame)) - self._seed_frame
        self.delta_label.setText(f"moved {delta:+d} frames")

    # ==================================================================
    # Committing
    # ==================================================================

    def _confirm(self):
        if not self._targets or not self._session_active or self._advance_pending:
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
                f"The playhead is in trial {trial_id}, but this seed belongs to trial {inst['trial']} — "
                "use Skip, or navigate back.",
                severity="warning",
            )
            return

        if target.field == "start" and t_rel >= inst["offset_s"]:
            notify("The start must stay before the end of the label.", severity="warning")
            return
        if target.field == "end" and t_rel <= inst["onset_s"]:
            notify("The end must stay after the start of the label.", severity="warning")
            return

        df = self.app_state.label_intervals
        if df is None or df.empty:
            notify("No labels in the current trial.", severity="warning")
            return
        mask = _row_mask(df, inst)
        pos = np.flatnonzero(mask.to_numpy())
        if not len(pos):
            notify("This label no longer exists (edited elsewhere?) — use Skip.", severity="warning")
            return
        row_idx = df.index[pos[0]]
        old_onset = _num(inst["onset_s"])
        old_offset = _num(inst["offset_s"])

        # A boundary placed by eye is a hand-made label: the row stops carrying
        # whatever score the model that produced it had.
        df = ensure_confidence(df)
        df.loc[row_idx, "confidence"] = HUMAN_CONFIDENCE
        inst["confidence"] = HUMAN_CONFIDENCE

        if target.field == "end":
            df.loc[row_idx, "offset_s"] = t_rel
            inst["offset_s"] = t_rel
        else:
            df.loc[row_idx, "onset_s"] = t_rel
            if target.field == "point" and math.isfinite(inst["offset_s"]):
                # Some point rows store offset == onset; keep them coincident.
                df.loc[row_idx, "offset_s"] = t_rel
                inst["offset_s"] = t_rel
            inst["onset_s"] = t_rel

        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(trial_id, df)
        self.app_state.changes_saved = False
        self._n_refined += 1
        self._record_refinement(inst, target.field, old_onset, old_offset)

        if self.io_widget is not None:
            self.io_widget._human_verification_true(mode="single_trial")
        if self.data_widget is not None:
            self.data_widget.update_main_plot(preserve_x_range=True)
        if self.labels_widget is not None:
            self.labels_widget.refresh_labels_shapes_layer()

        # Linger on the committed boundary before jumping — the redraw above
        # shows the label exactly where the user put it, and an instant jump
        # took that feedback away.
        self._advance_pending = True
        self.delta_label.setText("✓ placed")
        QTimer.singleShot(_CONFIRM_PAUSE_MS, self._advance_after_pause)

    def _delete_current(self):
        """Backspace: this event does not belong in the trial — drop the label.

        The whole row goes, so a state event's other boundary is dropped from
        the queue with it; the queue then moves on exactly as a commit does.
        Nothing reaches disk until the labels are saved.
        """
        if not self._targets or not self._session_active or self._advance_pending:
            return
        target = self._targets[self._idx]
        inst = target.inst
        trial = inst["trial"]
        current = getattr(self.app_state, "trials_sel", None)
        if current is not None and str(current) != str(trial):
            notify(
                f"The GUI is on trial {current}, but this seed belongs to trial {trial} — navigate back first.",
                severity="warning",
            )
            return

        df = self.app_state.label_intervals
        if df is None or df.empty:
            notify("No labels in the current trial.", severity="warning")
            return
        mask = _row_mask(df, inst)
        pos = np.flatnonzero(mask.to_numpy())
        if not len(pos):
            notify("This label no longer exists (edited elsewhere?) — use Skip.", severity="warning")
            return

        df = delete_interval(df, df.index[pos[0]])
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(trial, df)
        self.app_state.changes_saved = False
        self._n_deleted += 1
        self._record_deletion(inst, target.field)

        if self.io_widget is not None:
            self.io_widget._human_verification_true(mode="single_trial")
        if self.data_widget is not None:
            self.data_widget.update_main_plot(preserve_x_range=True)
        if self.labels_widget is not None:
            self.labels_widget.refresh_labels_shapes_layer()

        self.delta_label.setText("✗ deleted")
        self._advance_past(inst)

    def _advance_past(self, inst: dict):
        """Jump to the next target that is not part of *inst* (whose row is gone)."""
        idx = self._idx + 1
        while idx < len(self._targets) and self._targets[idx].inst is inst:
            idx += 1
        if idx >= len(self._targets):
            self._stop(done=True)
            return
        self._idx = idx
        self._jump_current()

    def _record_deletion(self, inst: dict, field: str):
        """Append a deletion to ``app_state.refine_log``.

        It keeps the original times and has no new ones, so the history shows
        it as *deleted* and the pre/post export carries the row on one side.
        """
        log = list(getattr(self.app_state, "refine_log", None) or [])
        identity = _log_identity(inst)
        mappings = getattr(self.labels_widget, "_mappings", {}) or {}
        log.append(
            {
                **identity,
                "name": str(mappings.get(identity["labels"], {}).get("name", identity["labels"])),
                "event_type": str(inst.get("event_type", "state")),
                "orig_onset_s": _num(inst["onset_s"]),
                "orig_offset_s": _num(inst["offset_s"]),
                "new_onset_s": None,
                "new_offset_s": None,
                "fields": [field],
                "deleted": True,
                "time": datetime.now().isoformat(timespec="seconds"),
            }
        )
        self.app_state.refine_log = log
        if self._history_dialog is not None and self._history_dialog.isVisible():
            self._history_dialog.refresh()

    def _record_refinement(self, inst: dict, field: str, old_onset: float | None, old_offset: float | None):
        """Append this commit to ``app_state.refine_log`` (→ local_settings.yaml).

        Re-refining a boundary CHAINS instead of duplicating: a record whose
        latest values equal this commit's old values is the same boundary
        refined again, so its ``new_*`` move on while ``orig_*`` keep the
        very first values — exactly what the pre/post export wants.
        """
        log = list(getattr(self.app_state, "refine_log", None) or [])
        identity = _log_identity(inst)
        new_onset = _num(inst["onset_s"])
        new_offset = _num(inst["offset_s"])
        now = datetime.now().isoformat(timespec="seconds")
        for rec in log:
            if (
                not rec.get("deleted")
                and all(rec.get(k) == v for k, v in identity.items())
                and _close(rec.get("new_onset_s"), old_onset)
                and _close(rec.get("new_offset_s"), old_offset)
            ):
                rec["new_onset_s"] = new_onset
                rec["new_offset_s"] = new_offset
                rec["time"] = now
                if field not in rec["fields"]:
                    rec["fields"] = [*rec["fields"], field]
                break
        else:
            mappings = getattr(self.labels_widget, "_mappings", {}) or {}
            log.append(
                {
                    **identity,
                    "name": str(mappings.get(identity["labels"], {}).get("name", identity["labels"])),
                    "event_type": str(inst.get("event_type", "state")),
                    "orig_onset_s": old_onset,
                    "orig_offset_s": old_offset,
                    "new_onset_s": new_onset,
                    "new_offset_s": new_offset,
                    "fields": [field],
                    "time": now,
                }
            )
        self.app_state.refine_log = log
        if self._history_dialog is not None and self._history_dialog.isVisible():
            self._history_dialog.refresh()

    def _advance_after_pause(self):
        if not self._advance_pending or not self._session_active:
            return  # Stop / Back / Skip / close intervened during the pause
        self._advance(+1)

    def _advance(self, direction: int):
        self._advance_pending = False
        new_idx = self._idx + direction
        if new_idx >= len(self._targets):
            self._stop(done=True)
            return
        self._idx = max(0, new_idx)
        self._jump_current()


def open_refine_dialog(meta, parent=None) -> RefineLabelsDialog:
    """Show the session's one refine dialog, creating it if needed.

    The top bar owns the instance wherever there is one, so Tools ▸ Refine and
    the frames grid raise the same dialog — two would fight over the
    application-wide Enter/Backspace shortcuts.
    """
    top_bar = getattr(getattr(meta, "shell", None), "_top_bar", None)
    dialog = top_bar.refine_dialog() if top_bar is not None else RefineLabelsDialog(meta, parent=parent)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
