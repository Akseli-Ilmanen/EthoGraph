"""Unified navigation widget: trial / label / sequence browsing with playback."""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.gui.notify import notify
from ethograph.io.time_model import (
    RestrictionWindow,
    TimeRange,
    find_closest_trial,
    infer_slider_range,
    trial_start_range,
)
from ethograph.utils.sequences import get_label_instances, match_sequences

NAVIGATE_MODES = ["Trial", "Label", "Sequence"]
SLIDER_SCOPES = [
    "Trial start → Trial end",
    "Trial start → Trial start (i+1)",
    "Session start → Session end",
]
XLIM_MODES = ["Slider scope", "Fixed window"]

_XLIM_KEY_TO_DISPLAY = {"interval": "Slider scope", "fixed": "Fixed window"}
_XLIM_DISPLAY_TO_KEY = {v: k for k, v in _XLIM_KEY_TO_DISPLAY.items()}

_SCOPE_KEY_TO_DISPLAY = {
    "trial": "Trial start → Trial end",
    "trial_start": "Trial start → Trial start (i+1)",
    "session": "Session start → Session end",
}
_SCOPE_DISPLAY_TO_KEY = {v: k for k, v in _SCOPE_KEY_TO_DISPLAY.items()}


class _DataAlignmentDialog(QDialog):
    """Wraps TimelinePage for standalone use from the navigation widget."""

    def __init__(self, app_state, parent=None):
        from .wizard_multi_timeline import TimelinePage

        super().__init__(parent)
        self.setWindowTitle("Data Alignment")
        self.resize(1100, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._page = TimelinePage(self)
        self._page.configure_for_standalone()
        layout.addWidget(self._page, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dt = getattr(app_state, "dt", None)
        if dt is not None and getattr(dt, "trials", None):
            self._page.populate_from_trialtree(dt, app_state)


class NavigationWidget(QWidget):
    """Unified navigation: trial/label/sequence modes, filtering, playback."""

    def __init__(self, shell, app_state, parent=None):
        super().__init__(parent=parent)
        self.shell = shell
        self.app_state = app_state
        self.catalog = None
        self.plot_container = None
        self.data_widget = None
        self._mappings: dict[int, dict[str, Any]] = {}
        self._label_instances: list[dict] = []
        self._sequence_matches: list[dict] = []

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # ── Navigate (unified prev/next + mode) ──────────────────────
        navigate_group = QGroupBox("Navigate")
        navigate_layout = QVBoxLayout()
        navigate_layout.setSpacing(2)
        navigate_layout.setContentsMargins(2, 2, 2, 2)
        navigate_group.setLayout(navigate_layout)

        filter_hint = QLabel("Only navigates trials visible in the trials table")
        filter_hint.setStyleSheet("color: grey; font-size: 10px;")
        navigate_layout.addWidget(filter_hint)

        # Navigate by
        nav_mode_row = QHBoxLayout()
        nav_mode_row.addWidget(QLabel("Navigate by:"))
        self.navigate_combo = QComboBox()
        self.navigate_combo.setObjectName("navigate_mode_combo")
        self.navigate_combo.addItems(NAVIGATE_MODES)
        self.navigate_combo.currentTextChanged.connect(self._on_navigate_changed)
        nav_mode_row.addWidget(self.navigate_combo, stretch=1)
        navigate_layout.addLayout(nav_mode_row)

        # Unified prev / next / counter row
        nav_row = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.prev_button.setObjectName("prev_button")
        self.prev_button.clicked.connect(lambda: self._navigate(-1))
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("next_button")
        self.next_button.clicked.connect(lambda: self._navigate(1))
        self.nav_counter = QLabel("")
        self.nav_counter.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.prev_button)
        nav_row.addWidget(self.nav_counter, stretch=1)
        nav_row.addWidget(self.next_button)
        navigate_layout.addLayout(nav_row)

        # Stacked mode panels
        self._stack = QStackedWidget()

        # -- Trial panel --
        trial_panel = QWidget()
        trial_lay = QVBoxLayout(trial_panel)
        trial_lay.setContentsMargins(0, 0, 0, 0)
        self.trials_combo = QComboBox()
        self.trials_combo.setEditable(True)
        self.trials_combo.setObjectName("trials_combo")
        self.trials_combo.currentTextChanged.connect(self._on_trial_combo_changed)
        self.trials_combo.currentIndexChanged.connect(self._sync_trials_combo_color)
        trial_lay.addWidget(self.trials_combo)
        self._stack.addWidget(trial_panel)

        # -- Label panel --
        label_panel = QWidget()
        label_lay = QVBoxLayout(label_panel)
        label_lay.setContentsMargins(0, 0, 0, 0)
        lr1 = QHBoxLayout()
        lr1.addWidget(QLabel("Label:"))
        self.label_combo = QComboBox()
        self.label_combo.currentIndexChanged.connect(self._on_label_selected)
        lr1.addWidget(self.label_combo, stretch=1)
        label_lay.addLayout(lr1)
        lr2 = QHBoxLayout()
        lr2.addWidget(QLabel("Individual:"))
        self.individual_combo = QComboBox()
        self.individual_combo.addItem("All")
        self.individual_combo.currentIndexChanged.connect(self._on_label_filter_changed)
        lr2.addWidget(self.individual_combo, stretch=1)
        label_lay.addLayout(lr2)
        self._stack.addWidget(label_panel)

        # -- Sequence panel --
        seq_panel = QWidget()
        seq_lay = QVBoxLayout(seq_panel)
        seq_lay.setContentsMargins(0, 0, 0, 0)
        sr = QHBoxLayout()
        sr.addWidget(QLabel("Pattern:"))
        self.sequence_input = QLineEdit()
        self.sequence_input.setPlaceholderText("e.g. 1-2-3-5")
        self.sequence_input.returnPressed.connect(self._on_sequence_search)
        sr.addWidget(self.sequence_input, stretch=1)
        self.sequence_search_btn = QPushButton("Search")
        self.sequence_search_btn.clicked.connect(self._on_sequence_search)
        sr.addWidget(self.sequence_search_btn)
        seq_lay.addLayout(sr)
        self._stack.addWidget(seq_panel)

        navigate_layout.addWidget(self._stack)

        # Slider scope
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Slider scope:"))
        self.scope_combo = QComboBox()
        self.scope_combo.setObjectName("slider_scope_combo")
        self.scope_combo.addItems(SLIDER_SCOPES)
        self.scope_combo.currentTextChanged.connect(self._on_scope_changed)
        scope_row.addWidget(self.scope_combo, stretch=1)
        navigate_layout.addLayout(scope_row)

        # X-limits mode: slider-scope-based or fixed window
        xlim_row = QHBoxLayout()
        xlim_row.addWidget(QLabel("X-limits:"))
        self.xlim_combo = QComboBox()
        self.xlim_combo.setObjectName("xlim_mode_combo")
        self.xlim_combo.addItems(XLIM_MODES)
        self.xlim_combo.setToolTip(
            "Slider scope: x-limits follow the slider scope's interval\n"
            "(trial period / label / sequence) plus Before/After padding.\n"
            "Fixed window: x-limits span a fixed-size window from t=0."
        )
        self.xlim_combo.currentTextChanged.connect(self._on_xlim_mode_changed)
        xlim_row.addWidget(self.xlim_combo, stretch=1)
        navigate_layout.addLayout(xlim_row)

        # Before / After padding (interval mode only)
        self.interval_pad_widget = QWidget()
        ba_row = QHBoxLayout(self.interval_pad_widget)
        ba_row.setContentsMargins(0, 0, 0, 0)
        ba_row.addWidget(QLabel("Before:"))
        self.before_spin = QDoubleSpinBox()
        self.before_spin.setRange(0.0, 600.0)
        self.before_spin.setSingleStep(0.5)
        self.before_spin.setSuffix(" s")
        self.before_spin.setValue(app_state.get_with_default("before_s_trial"))
        self.before_spin.valueChanged.connect(self._on_before_after_changed)
        ba_row.addWidget(self.before_spin)
        ba_row.addWidget(QLabel("After:"))
        self.after_spin = QDoubleSpinBox()
        self.after_spin.setRange(0.0, 600.0)
        self.after_spin.setSingleStep(0.5)
        self.after_spin.setSuffix(" s")
        self.after_spin.setValue(app_state.get_with_default("after_s_trial"))
        self.after_spin.valueChanged.connect(self._on_before_after_changed)
        ba_row.addWidget(self.after_spin)
        navigate_layout.addWidget(self.interval_pad_widget)

        # Window size (fixed mode only)
        self.fixed_window_widget = QWidget()
        fw_row = QHBoxLayout(self.fixed_window_widget)
        fw_row.setContentsMargins(0, 0, 0, 0)
        fw_row.addWidget(QLabel("Window size:"))
        self.fixed_window_spin = QDoubleSpinBox()
        self.fixed_window_spin.setObjectName("fixed_window_spin")
        self.fixed_window_spin.setRange(0.1, 36000.0)
        self.fixed_window_spin.setSingleStep(1.0)
        self.fixed_window_spin.setDecimals(1)
        self.fixed_window_spin.setSuffix(" s")
        self.fixed_window_spin.setValue(app_state.get_with_default("fixed_window_s"))
        self.fixed_window_spin.valueChanged.connect(self._on_fixed_window_changed)
        fw_row.addWidget(self.fixed_window_spin, stretch=1)
        navigate_layout.addWidget(self.fixed_window_widget)

        # Auto-play checkbox
        self.autoplay_checkbox = QCheckBox("Auto-play on navigate")
        self.autoplay_checkbox.setToolTip("Start playback from onset when navigating to next item")
        navigate_layout.addWidget(self.autoplay_checkbox)

        # Jump to time
        jump_row = QHBoxLayout()
        jump_row.addWidget(QLabel("Jump to:"))
        self.jump_time_spin = QDoubleSpinBox()
        self.jump_time_spin.setRange(0.0, 1e8)
        self.jump_time_spin.setDecimals(3)
        self.jump_time_spin.setSuffix(" s")
        jump_row.addWidget(self.jump_time_spin, stretch=1)
        jump_btn = QPushButton("Go")
        jump_btn.setFixedWidth(40)
        jump_btn.clicked.connect(self._on_jump_to_time)
        jump_row.addWidget(jump_btn)
        navigate_layout.addLayout(jump_row)

        # Jump step (arrow-key time step)
        step_row = QHBoxLayout()
        self.time_jump_label = QLabel("Jump step:")
        step_row.addWidget(self.time_jump_label)
        self.time_jump_spin = QDoubleSpinBox()
        self.time_jump_spin.setRange(0.001, 1000.0)
        self.time_jump_spin.setSingleStep(0.1)
        self.time_jump_spin.setDecimals(3)
        self.time_jump_spin.setSuffix(" s")
        self.time_jump_spin.setToolTip("Step size for keyboard time jumps (Shift+←/→)")
        self.time_jump_spin.setValue(app_state.get_with_default("time_jump_s"))
        self.time_jump_spin.valueChanged.connect(lambda v: setattr(app_state, "time_jump_s", v))
        step_row.addWidget(self.time_jump_spin, stretch=1)
        navigate_layout.addLayout(step_row)

        # Playback controls (mode / FPS / center / hide-label / rotate) now
        # live in the bottom playback bar; screen recording is in Tools menu.

        # ── Assemble ─────────────────────────────────────────────────
        main_layout.addWidget(navigate_group)
        self.setLayout(main_layout)

        # Restore saved modes
        saved_nav = app_state.get_with_default("navigate_mode")
        nav_items = [m.lower() for m in NAVIGATE_MODES]
        nav_idx = nav_items.index(saved_nav) if saved_nav in nav_items else 0
        self.navigate_combo.setCurrentIndex(nav_idx)

        saved_scope = app_state.get_with_default("slider_scope")
        scope_display = _SCOPE_KEY_TO_DISPLAY.get(saved_scope, SLIDER_SCOPES[0])
        scope_idx = SLIDER_SCOPES.index(scope_display) if scope_display in SLIDER_SCOPES else 0
        self.scope_combo.setCurrentIndex(scope_idx)

        self._sync_xlim_combo_from_state()

    # ==================================================================
    # Public API (used by widgets_data, shortcuts, widgets_meta, etc.)
    # ==================================================================

    def set_plot_container(self, pc):
        self.plot_container = pc

    def set_data_widget(self, dw):
        self.data_widget = dw

    def set_mappings(self, mappings: dict[int, dict[str, Any]]):
        self._mappings = mappings
        self._populate_label_combo()

    def refresh_after_load(self):
        self._populate_label_combo()
        self._populate_individual_combo()
        # xlim_mode is a global user preference and may have been loaded/
        # changed since this widget was built — re-sync the combo from
        # app_state.
        self._sync_xlim_combo_from_state()

    def on_labels_changed(self):
        """Refresh label/sequence instances after labels are modified.

        Preserves the current instance index so the user stays at their
        position rather than jumping back to the first instance.
        """
        mode = self.app_state.navigate_mode
        if mode == "label":
            old_idx = self.app_state.label_instance_idx
            self._refresh_label_instances_keep_position(old_idx)
        elif mode == "sequence" and self._sequence_matches:
            old_idx = self.app_state.sequence_match_idx
            self._on_sequence_search()
            self.app_state.sequence_match_idx = min(old_idx, max(0, len(self._sequence_matches) - 1))
            self._update_counter()

    def _refresh_label_instances_keep_position(self, old_idx: int):
        """Refresh label instances and keep index close to old_idx."""
        label_id = self.label_combo.currentData()
        if label_id is None:
            self._label_instances = []
            self._update_counter()
            return
        individual = self.individual_combo.currentText()
        ind_filter = None if individual == "All" else individual
        df = getattr(self.app_state, "_all_labels_df", None)
        self._label_instances = get_label_instances(df, label_id, ind_filter)
        self.app_state.label_instance_idx = min(old_idx, max(0, len(self._label_instances) - 1))
        self._update_counter()

    def navigate_to_trial(self, trial_id):
        self.trials_combo.setCurrentText(str(trial_id))

    def next_trial(self):
        self._navigate(1)

    def prev_trial(self):
        self._navigate(-1)

    # Keyboard shortcut helpers (unchanged API)
    def step_frame_forward(self):
        self._step_frame(+1)

    def step_frame_backward(self):
        self._step_frame(-1)

    def step_window_forward(self):
        self._step_window(+1)

    def step_window_backward(self):
        self._step_window(-1)

    # ==================================================================
    # Unified navigate (prev / next works in all modes)
    # ==================================================================

    def _navigate(self, direction: int):
        mode = self.navigate_combo.currentText().lower()
        if mode == "trial":
            self._navigate_trial(direction)
        elif mode == "label":
            self._navigate_label(direction)
        elif mode == "sequence":
            self._navigate_sequence(direction)

    # ==================================================================
    # Navigate by / Slider scope switching
    # ==================================================================

    def _on_navigate_changed(self, mode_text: str):
        mode = mode_text.lower()
        self.app_state.navigate_mode = mode
        self._stack.setCurrentIndex(NAVIGATE_MODES.index(mode_text))

        # Swap spinbox values to the per-category stored values
        self._sync_spinboxes_to_mode(mode)

        if mode == "label":
            self._refresh_label_instances()
            self._apply_label_restriction()
        elif mode == "sequence":
            self._on_sequence_search()
        else:
            self._apply_slider_scope()

        self._update_counter()

    def _sync_spinboxes_to_mode(self, mode: str):
        """Load the per-category before/after values into the spinboxes."""
        self.before_spin.blockSignals(True)
        self.after_spin.blockSignals(True)
        self.before_spin.setValue(self.app_state.get_with_default(f"before_s_{mode}"))
        self.after_spin.setValue(self.app_state.get_with_default(f"after_s_{mode}"))
        self.before_spin.blockSignals(False)
        self.after_spin.blockSignals(False)

    def _on_scope_changed(self, scope_text: str):
        scope_key = _SCOPE_DISPLAY_TO_KEY.get(scope_text, "trial")
        self.app_state.slider_scope = scope_key

        if scope_key == "session":
            self._apply_slider_scope()
            self._update_viewport_for_scope()
        elif scope_key in ("trial", "trial_start"):
            self._snap_to_closest_trial()
        else:
            self._apply_slider_scope()

    def _on_xlim_mode_changed(self, mode_text: str):
        mode = _XLIM_DISPLAY_TO_KEY.get(mode_text, "interval")
        self.app_state.xlim_mode = mode
        self._sync_xlim_widgets(mode)
        self._apply_slider_scope()
        self._update_viewport_for_scope()

    def _sync_xlim_widgets(self, mode: str):
        self.interval_pad_widget.setVisible(mode == "interval")
        self.fixed_window_widget.setVisible(mode == "fixed")

    def _sync_xlim_combo_from_state(self):
        mode = self.app_state.get_with_default("xlim_mode")
        display = _XLIM_KEY_TO_DISPLAY.get(mode, "Slider scope")
        self.xlim_combo.blockSignals(True)
        self.xlim_combo.setCurrentIndex(XLIM_MODES.index(display))
        self.xlim_combo.blockSignals(False)
        self._sync_xlim_widgets(mode)

    def _on_fixed_window_changed(self, value: float):
        self.app_state.fixed_window_s = value
        if self.app_state.get_with_default("xlim_mode") == "fixed":
            self._apply_slider_scope()
            self._update_viewport_for_scope()

    def _fixed_pan_extent(self) -> TimeRange | None:
        """Full extent the fixed window can slide over (scope-dependent)."""
        if self.app_state.slider_scope == "session":
            sc = getattr(self.app_state, "source_collection", None)
            return sc.session_range if sc else None
        alignment = getattr(self.app_state, "trial_alignment", None)
        return alignment.trial_range if alignment else None

    def _apply_fixed_window(self, anchor: float | None = None):
        """Build a fixed-size restrict_window anchored at *anchor* (or t=0).

        ``core_range`` is the visible window of ``fixed_window_s`` seconds;
        ``time_range`` spans the whole scope extent so the window can be
        dragged / slid across a long recording (plots pan within it, buffers
        load from it).
        """
        size = self.app_state.get_with_default("fixed_window_s")
        trial_id = getattr(self.app_state, "trials_sel", None)
        extent = self._fixed_pan_extent()
        base = anchor if anchor is not None else (extent.start_s if extent else 0.0)
        if extent is not None:
            base = max(extent.start_s, min(base, extent.end_s - size))
        window = TimeRange(base, base + size)
        self.app_state.restrict_window = RestrictionWindow(
            mode="fixed",
            time_range=extent.union(window) if extent else window,
            core_range=window,
            trial_id=trial_id,
        )

    def _update_viewport_for_scope(self):
        """Set the plot x-range to match the current restrict_window."""
        if self.plot_container is None:
            return
        rw = getattr(self.app_state, "restrict_window", None)
        if rw is None:
            return
        # Re-apply zoom constraints first: in fixed mode the span is locked
        # (minXRange == maxXRange), so stale limits would block the new range.
        self.plot_container._apply_all_zoom_constraints()
        master = getattr(self.plot_container, "_xlink_master", None) or getattr(
            self.plot_container, "_feature_plot", None
        )
        if master is not None:
            tr = rw.core_range if rw.mode == "fixed" else rw.time_range
            master.vb.setXRange(tr.start_s, tr.end_s, padding=0)

    def _snap_to_closest_trial(self):
        """Switch to the trial closest to the current time marker, then update viewport."""
        sio = getattr(self.app_state, "nwb_alignment", None)
        trials = getattr(self.app_state, "trials", None)
        if not sio or not trials:
            self._apply_slider_scope()
            return

        # Get current time from time marker / slider
        current_time = 0.0

        # Convert local time to session-absolute for lookup
        sc = getattr(self.app_state, "source_collection", None)
        new_trial = None
        if sc and sc.n_trials > 0:
            curr_trial = getattr(self.app_state, "trials_sel", None)
            if curr_trial in trials:
                session_time = sc.to_session(curr_trial, current_time)
                hit = sc.to_trial(session_time)
                if hit is not None:
                    new_trial = hit[0]

        if new_trial is not None and new_trial in trials:
            if new_trial != self.app_state.trials_sel:
                self.app_state.trials_sel = new_trial
                self.trials_combo.blockSignals(True)
                self.trials_combo.setCurrentText(str(new_trial))
                self.trials_combo.blockSignals(False)
                self.app_state.trial_changed.emit()
                self._update_counter()
                return

        self._apply_slider_scope()
        self._update_viewport_for_scope()

    def _apply_slider_scope(self):
        """Build restrict_window from the current slider scope + before/after."""
        if self.app_state.get_with_default("xlim_mode") == "fixed":
            self._apply_fixed_window()
            return

        alignment = getattr(self.app_state, "trial_alignment", None)
        trial_id = getattr(self.app_state, "trials_sel", None)
        scope = self.app_state.slider_scope
        before = self.before_spin.value()
        after = self.after_spin.value()

        if scope == "trial" and alignment and alignment.trial_range:
            core = alignment.trial_range
            time_range = TimeRange(core.start_s - before, core.end_s + after)
            self.app_state.restrict_window = RestrictionWindow(
                mode="trial",
                time_range=time_range,
                core_range=core,
                trial_id=trial_id,
            )
        elif scope == "trial_start" and alignment and alignment.trial_range:
            # End at the next trial's start; fall back to the alignment range
            # (video duration / session end) for the last trial.
            sio = getattr(self.app_state, "nwb_alignment", None)
            core = (trial_start_range(sio, trial_id) if sio else None) or alignment.trial_range
            time_range = TimeRange(core.start_s - before, core.end_s + after)
            self.app_state.restrict_window = RestrictionWindow(
                mode="trial_start",
                time_range=time_range,
                core_range=core,
                trial_id=trial_id,
            )
        elif scope == "session":
            sc = getattr(self.app_state, "source_collection", None)
            session = sc.session_range if sc else None
            if session:
                self.app_state.restrict_window = RestrictionWindow(
                    mode="session",
                    time_range=session,
                    core_range=session,
                    trial_id=trial_id,
                )
        elif alignment and alignment.trial_range:
            # Fallback: use trial range
            core = alignment.trial_range
            time_range = TimeRange(core.start_s - before, core.end_s + after)
            self.app_state.restrict_window = RestrictionWindow(
                mode="trial",
                time_range=time_range,
                core_range=core,
                trial_id=trial_id,
            )

    def auto_infer_scope(self):
        """Auto-detect slider scope from alignment timing and update the combo."""
        sio = getattr(self.app_state, "nwb_alignment", None)
        trial_id = getattr(self.app_state, "trials_sel", None)
        sc = getattr(self.app_state, "source_collection", None)
        if sio is None or trial_id is None:
            return
        scope, _ = infer_slider_range(sio, trial_id, sc)
        display = _SCOPE_KEY_TO_DISPLAY.get(scope, SLIDER_SCOPES[0])
        self.scope_combo.blockSignals(True)
        self.scope_combo.setCurrentText(display)
        self.scope_combo.blockSignals(False)
        self.app_state.slider_scope = scope

    # ==================================================================
    # Trial mode
    # ==================================================================

    def _on_trial_combo_changed(self):
        if not self.app_state.ready:
            return
        trials_sel = self.trials_combo.currentText()
        if not trials_sel or trials_sel.strip() == "":
            return
        trials = getattr(self.app_state, "trials", None)
        if not trials:
            return
        if trials_sel not in trials and str(trials_sel) not in [str(t) for t in trials]:
            notify(f"Unknown trial: {trials_sel!r}", severity="warning")
            return
        try:
            self.app_state.set_key_sel("trials", trials_sel)
        except KeyError:
            notify(f"Unknown trial: {trials_sel!r}", severity="warning")
            return
        self.app_state.trial_changed.emit()
        self._update_counter()
        tb = self.app_state.trial_bounds
        if tb:
            self._center_and_maybe_play(tb.start_s, tb.end_s)

    def _navigate_trial(self, direction: int):
        if not self.app_state.trials:
            return
        try:
            curr_idx = self.app_state.trials.index(self.app_state.trials_sel)
        except ValueError:
            curr_idx = 0
        new_idx = curr_idx + direction

        if 0 <= new_idx < len(self.app_state.trials):
            new_trial = self.app_state.trials[new_idx]
            self.app_state.trials_sel = new_trial
            self.trials_combo.blockSignals(True)
            self.trials_combo.setCurrentText(str(new_trial))
            self.trials_combo.blockSignals(False)
            self.app_state.trial_changed.emit()
            self._update_counter()
            tb = self.app_state.trial_bounds
            if tb:
                self._center_and_maybe_play(tb.start_s, tb.end_s)

    def _apply_trial_restriction(self):
        self._apply_slider_scope()

    # ==================================================================
    # Label mode
    # ==================================================================

    def _populate_label_combo(self):
        self.label_combo.blockSignals(True)
        self.label_combo.clear()
        for label_id, info in sorted(self._mappings.items(), key=lambda x: x[0]):
            if label_id == 0:
                continue
            name = info.get("name", str(label_id))
            self.label_combo.addItem(f"{label_id} ({name})", label_id)
        self.label_combo.blockSignals(False)

    def _populate_individual_combo(self):
        self.individual_combo.blockSignals(True)
        self.individual_combo.clear()
        self.individual_combo.addItem("All")
        df = getattr(self.app_state, "_all_labels_df", None)
        if df is not None and "individual" in df.columns:
            for ind in sorted(df["individual"].unique()):
                self.individual_combo.addItem(str(ind))
        self.individual_combo.blockSignals(False)

    def _on_label_selected(self):
        if not self.app_state.ready or self.app_state.navigate_mode != "label":
            return
        self._refresh_label_instances()
        self._apply_label_restriction()

    def _on_label_filter_changed(self):
        if not self.app_state.ready or self.app_state.navigate_mode != "label":
            return
        self._refresh_label_instances()
        self._apply_label_restriction()

    def _refresh_label_instances(self):
        label_id = self.label_combo.currentData()
        if label_id is None:
            self._label_instances = []
            self._update_counter()
            return
        individual = self.individual_combo.currentText()
        ind_filter = None if individual == "All" else individual
        df = getattr(self.app_state, "_all_labels_df", None)
        self._label_instances = get_label_instances(df, label_id, ind_filter)
        self.app_state.label_instance_idx = 0
        self._update_counter()

    def _navigate_label(self, direction: int):
        if not self._label_instances:
            return
        idx = self.app_state.label_instance_idx + direction
        idx = max(0, min(idx, len(self._label_instances) - 1))
        self.app_state.label_instance_idx = idx
        self._update_counter()
        self._apply_label_restriction()

    def _apply_label_restriction(self):
        if not self._label_instances:
            return
        idx = self.app_state.label_instance_idx
        inst = self._label_instances[idx]
        trial_id = inst["trial"]

        if getattr(self.app_state, "trials_sel", None) != trial_id:
            self.app_state.trials_sel = trial_id
            self.trials_combo.blockSignals(True)
            self.trials_combo.setCurrentText(str(trial_id))
            self.trials_combo.blockSignals(False)
            self.app_state.trial_changed.emit()

        self._update_counter()
        onset_s = float(inst["onset_s"])
        offset_s = float(inst["offset_s"])
        self._center_and_maybe_play(onset_s, offset_s)

    # ==================================================================
    # Sequence mode
    # ==================================================================

    def _on_sequence_search(self):
        pattern = self.sequence_input.text().strip()
        if not pattern:
            self._sequence_matches = []
            self._update_counter()
            return
        self.app_state.sequence_pattern = pattern
        df = getattr(self.app_state, "_all_labels_df", None)
        self._sequence_matches = match_sequences(df, pattern)
        self.app_state.sequence_match_idx = 0
        self._update_counter()
        if self._sequence_matches:
            self._apply_sequence_restriction()

    def _navigate_sequence(self, direction: int):
        if not self._sequence_matches:
            return
        idx = self.app_state.sequence_match_idx + direction
        idx = max(0, min(idx, len(self._sequence_matches) - 1))
        self.app_state.sequence_match_idx = idx
        self._update_counter()
        self._apply_sequence_restriction()

    def _apply_sequence_restriction(self):
        if not self._sequence_matches:
            return
        idx = self.app_state.sequence_match_idx
        match = self._sequence_matches[idx]
        trial_id = match["trial"]

        if getattr(self.app_state, "trials_sel", None) != trial_id:
            self.app_state.trials_sel = trial_id
            self.trials_combo.blockSignals(True)
            self.trials_combo.setCurrentText(str(trial_id))
            self.trials_combo.blockSignals(False)
            self.app_state.trial_changed.emit()

        self._update_counter()
        onset_s = float(match["onset_s"])
        offset_s = float(match["offset_s"])
        self._center_and_maybe_play(onset_s, offset_s)

    # ==================================================================
    # Counter display
    # ==================================================================

    def _update_counter(self):
        mode = self.navigate_combo.currentText().lower()
        if mode == "trial":
            trials = getattr(self.app_state, "trials", [])
            sel = getattr(self.app_state, "trials_sel", None)
            if trials and sel in trials:
                idx = trials.index(sel)
                self.nav_counter.setText(f"{idx + 1} / {len(trials)}")
            else:
                self.nav_counter.setText("")
        elif mode == "label":
            total = len(self._label_instances)
            idx = self.app_state.label_instance_idx
            if total == 0:
                self.nav_counter.setText("0 / 0")
            else:
                inst = self._label_instances[idx]
                self.nav_counter.setText(f"{idx + 1} / {total}  (trial={inst['trial']})")
        elif mode == "sequence":
            total = len(self._sequence_matches)
            idx = self.app_state.sequence_match_idx
            if total == 0:
                self.nav_counter.setText("0 / 0")
            else:
                m = self._sequence_matches[idx]
                self.nav_counter.setText(f"{idx + 1} / {total}  (trial={m['trial']})")

    # ==================================================================
    # Center view + auto-play
    # ==================================================================

    def _center_and_maybe_play(self, onset_s: float, offset_s: float):
        """Center view on *onset_s..offset_s* + context, seek to onset, optionally play."""
        if self.plot_container is None:
            return

        master = getattr(self.plot_container, "_xlink_master", None) or getattr(
            self.plot_container, "_feature_plot", None
        )

        if self.app_state.get_with_default("xlim_mode") == "fixed":
            # Fixed window: anchor at the navigated interval's onset for
            # label/sequence navigation, else at the scope origin (t=0).
            anchor = onset_s if self.app_state.navigate_mode in ("label", "sequence") else None
            self._apply_fixed_window(anchor)
            rw = self.app_state.restrict_window
            if master is not None and rw is not None:
                master.vb.setXRange(rw.core_range.start_s, rw.core_range.end_s, padding=0)
        else:
            extra_t0 = self.before_spin.value()
            extra_t1 = self.after_spin.value()
            if master is not None:
                master.vb.setXRange(onset_s - extra_t0, offset_s + extra_t1, padding=0)

        self.plot_container.update_time_marker_by_time(onset_s)

        video = getattr(self.app_state, "video", None)
        if video is not None and not self.autoplay_checkbox.isChecked():
            video.seek_to_frame(video.time_to_frame(onset_s))

        if self.autoplay_checkbox.isChecked():
            self._play_interval(onset_s, offset_s)

    def _play_interval(self, onset_s: float, offset_s: float):
        """Play a specific time interval via video or audio-only fallback."""
        if self.app_state.video:
            start_frame = self.app_state.video.time_to_frame(onset_s)
            end_frame = self.app_state.video.time_to_frame(offset_s)
            self.app_state.video.play_segment(start_frame, end_frame)
        elif self.plot_container and hasattr(self.plot_container, "audio_player"):
            self.plot_container.audio_player.play_segment(onset_s, offset_s)

    # ==================================================================
    # Before / After padding
    # ==================================================================

    def _on_before_after_changed(self):
        mode = self.app_state.navigate_mode
        setattr(self.app_state, f"before_s_{mode}", self.before_spin.value())
        setattr(self.app_state, f"after_s_{mode}", self.after_spin.value())
        self._apply_slider_scope()

    # ==================================================================
    # Jump to time
    # ==================================================================

    def _on_jump_to_time(self):
        trials = getattr(self.app_state, "trials", None)
        sc = getattr(self.app_state, "source_collection", None)
        global_t = self.jump_time_spin.value()

        hit = sc.to_trial(global_t) if sc is not None and sc.n_trials > 0 else None
        if hit is None:
            # No trial bookmarks — fall back to the alignment table lookup.
            sio = getattr(self.app_state, "nwb_alignment", None)
            if sio is None or not trials:
                notify("Cannot jump: no trial timing info", severity="warning")
                return
            try:
                hit = find_closest_trial(sio, trials, global_t)
            except ValueError:
                notify("Cannot jump: no trial timing info", severity="warning")
                return
        trial_id, rel_t = hit

        if trial_id != getattr(self.app_state, "trials_sel", None):
            self.app_state.trials_sel = trial_id
            self.trials_combo.blockSignals(True)
            self.trials_combo.setCurrentText(str(trial_id))
            self.trials_combo.blockSignals(False)
            self.app_state.trial_changed.emit()
            self._update_counter()

        self._seek_to_time(self.app_state.to_display(trial_id, max(0.0, rel_t)))

    def _seek_to_time(self, time_s: float):
        """Move the time marker (and video) to *time_s*, scrolling it into view."""
        if self.plot_container is None:
            return
        self.plot_container.update_time_marker_by_time(time_s)
        visible = TimeRange(*self.plot_container.get_current_xlim())
        if not visible.contains(time_s):
            master = getattr(self.plot_container, "_xlink_master", None) or getattr(
                self.plot_container, "_feature_plot", None
            )
            if master is not None:
                half = self.app_state.view_span / 2.0
                master.vb.setXRange(time_s - half, time_s + half, padding=0)
        video = getattr(self.app_state, "video", None)
        if video is not None:
            video.seek_to_frame(video.time_to_frame(time_s))

    def setup_trial_conditions(self, catalog):
        """No-op — trial condition filtering moved to TrialsWidget."""
        self.catalog = catalog

    # ==================================================================
    # Playback
    # ==================================================================

    def _step_frame(self, direction: int):
        if not self.app_state.ready:
            return
        if self.app_state.video:
            video = self.app_state.video
            new_frame = self.app_state.current_frame + direction
            new_frame = max(0, min(new_frame, self.app_state.num_frames - 1))
            video.seek_to_frame(new_frame)

    def _current_time(self) -> float | None:
        """Where the playhead is — from the video, else the middle of the view."""
        video = self.app_state.video
        if video is not None:
            return video.frame_to_time(self.app_state.current_frame)
        if self.plot_container is None:
            return None
        t0, t1 = self.plot_container.get_current_xlim()
        return (t0 + t1) / 2.0

    def _step_window(self, direction: int):
        """Jump one view span, clamped to the navigable extent (Shift+←/→).

        Goes through :meth:`_seek_to_time`, which moves the marker, scrolls the
        window and seeks the video — the previous implementation called a
        ``_step_time_no_video`` helper and a ``plot_container.time_slider`` that
        both went away with the bottom bar, so this raised AttributeError.
        """
        if not self.app_state.ready:
            return
        span = self.app_state.view_span
        current = self._current_time()
        if span <= 0 or current is None:
            return
        target = current + direction * span
        bounds = self.app_state.padded_bounds
        if bounds is not None:
            target = max(bounds.start_s, min(target, bounds.end_s))
        self._seek_to_time(target)

    def _sync_trials_combo_color(self):
        idx = self.trials_combo.currentIndex()
        le = self.trials_combo.lineEdit()
        if le is None:
            return
        bg = self.trials_combo.itemData(idx, Qt.BackgroundRole)
        fg = self.trials_combo.itemData(idx, Qt.ForegroundRole)
        if bg and fg:
            le.setStyleSheet(f"background-color: {bg.name()}; color: {fg.name()};")
        else:
            le.setStyleSheet("")
