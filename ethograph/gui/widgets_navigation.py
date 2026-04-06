"""Unified navigation widget: trial / label / sequence browsing with playback."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from napari import Viewer
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ethograph.io.time_model import build_trial_window, find_closest_trial
from ethograph.utils.sequences import get_label_instances, match_sequences

from .app_constants import AUDIO_SPEED_MAX, AUDIO_SPEED_MIN, AUDIO_SPEED_STEP
from .dialog_screen_recorder import RecordButton

logger = logging.getLogger(__name__)

RESTRICT_MODES = ["Trial", "Video", "Label", "Sequence"]


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

    def __init__(self, viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.viewer = viewer
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
        navigate_group.setLayout(navigate_layout)

        filter_hint = QLabel("Only navigates trials visible in the Trials tab")
        filter_hint.setStyleSheet("color: grey; font-size: 10px;")
        navigate_layout.addWidget(filter_hint)

        # Mode selector row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Restrict:"))
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("restrict_mode_combo")
        self.mode_combo.addItems(RESTRICT_MODES)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, stretch=1)
        navigate_layout.addLayout(mode_row)

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

        # -- Video panel --
        video_panel = QWidget()
        video_lay = QVBoxLayout(video_panel)
        video_lay.setContentsMargins(0, 0, 0, 0)
        self.video_combo = QComboBox()
        self.video_combo.setObjectName("video_combo")
        self.video_combo.currentIndexChanged.connect(self._on_video_combo_changed)
        video_lay.addWidget(self.video_combo)
        self._video_info_label = QLabel("")
        self._video_info_label.setStyleSheet("color: grey; font-size: 10px;")
        video_lay.addWidget(self._video_info_label)
        self._stack.addWidget(video_panel)

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

        # Extra context
        ctx_row = QHBoxLayout()
        ctx_row.addWidget(QLabel("Context:"))
        self.extra_t0_spin = QDoubleSpinBox()
        self.extra_t0_spin.setRange(0.0, 60.0)
        self.extra_t0_spin.setSingleStep(0.1)
        self.extra_t0_spin.setSuffix(" s")
        self.extra_t0_spin.setValue(app_state.get_with_default("restrict_extra_t0"))
        self.extra_t0_spin.valueChanged.connect(self._on_extra_context_changed)
        ctx_row.addWidget(self.extra_t0_spin)
        ctx_row.addWidget(QLabel("+"))
        self.extra_t1_spin = QDoubleSpinBox()
        self.extra_t1_spin.setRange(0.0, 60.0)
        self.extra_t1_spin.setSingleStep(0.1)
        self.extra_t1_spin.setSuffix(" s")
        self.extra_t1_spin.setValue(app_state.get_with_default("restrict_extra_t1"))
        self.extra_t1_spin.valueChanged.connect(self._on_extra_context_changed)
        ctx_row.addWidget(self.extra_t1_spin)
        navigate_layout.addLayout(ctx_row)

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

        # ── Playback ─────────────────────────────────────────────────
        playback_group = QGroupBox("Playback")
        playback_layout = QGridLayout()
        playback_group.setLayout(playback_layout)

        self.fps_label = QLabel("Playback FPS:")
        self.fps_playback_edit = QLineEdit()
        self.fps_playback_edit.setObjectName("fps_playback_edit")
        self.fps_playback_edit.setText(str(app_state.get_with_default("fps_playback")))
        self.fps_playback_edit.editingFinished.connect(self._on_fps_changed)
        self.fps_playback_edit.setToolTip(
            "Playback FPS for video.\n"
            "Audio playback speed is coupled to this setting.\n"
            "Set to recording FPS for normal audio playback."
        )

        self.audio_speed_label = QLabel("Audio speed:")
        self.audio_speed_spin = QDoubleSpinBox()
        self.audio_speed_spin.setObjectName("audio_speed_spin")
        self.audio_speed_spin.setRange(AUDIO_SPEED_MIN, AUDIO_SPEED_MAX)
        self.audio_speed_spin.setSingleStep(AUDIO_SPEED_STEP)
        self.audio_speed_spin.setDecimals(2)
        self.audio_speed_spin.setSuffix("\u00d7")
        self.audio_speed_spin.setValue(app_state.get_with_default("audio_playback_speed"))
        self.audio_speed_spin.valueChanged.connect(self._on_audio_speed_changed)

        self.coupling_button = QPushButton("\U0001f517")
        self.coupling_button.setCheckable(True)
        self.coupling_button.setChecked(app_state.get_with_default("av_speed_coupled"))
        self.coupling_button.setFixedWidth(30)
        self.coupling_button.toggled.connect(self._on_coupling_toggled)

        self.skip_frames_checkbox = QCheckBox("Skip Frames")
        self.skip_frames_checkbox.setChecked(app_state.get_with_default("skip_frames"))
        self.skip_frames_checkbox.setToolTip(
            "Skip frames to match playback FPS.\n"
            "Good for low-resolution video and fast streaming/seeking.\n"
            "For high-resolution video, disabling this\n"
            "sometimes gives smoother playback."
        )
        self.skip_frames_checkbox.toggled.connect(lambda v: setattr(app_state, "skip_frames", v))

        self.center_playback_checkbox = QCheckBox("Center playback")
        self.center_playback_checkbox.setChecked(app_state.get_with_default("center_playback"))
        self.center_playback_checkbox.toggled.connect(lambda v: setattr(app_state, "center_playback", v))

        self.time_jump_label = QLabel("Jump step (ms):")
        self.time_jump_spin = QDoubleSpinBox()
        self.time_jump_spin.setRange(1.0, 5000.0)
        self.time_jump_spin.setSingleStep(10.0)
        self.time_jump_spin.setDecimals(0)
        self.time_jump_spin.setSuffix(" ms")
        self.time_jump_spin.setValue(app_state.get_with_default("time_jump_ms"))
        self.time_jump_spin.valueChanged.connect(lambda v: setattr(app_state, "time_jump_ms", v))

        playback_layout.addWidget(self.fps_label, 0, 0)
        playback_layout.addWidget(self.fps_playback_edit, 0, 1)
        playback_layout.addWidget(self.skip_frames_checkbox, 0, 2)
        playback_layout.addWidget(self.audio_speed_label, 1, 0)
        playback_layout.addWidget(self.audio_speed_spin, 1, 1)
        playback_layout.addWidget(self.coupling_button, 1, 2)
        playback_layout.addWidget(self.time_jump_label, 2, 0)
        playback_layout.addWidget(self.time_jump_spin, 2, 1)
        playback_layout.addWidget(self.center_playback_checkbox, 2, 2)

        self.play_pause_btn = QPushButton("\u25B6")
        self.play_pause_btn.setToolTip("Play / Pause  (Space)")
        self.play_pause_btn.setFixedWidth(36)
        self.play_pause_btn.clicked.connect(self._on_play_pause_clicked)

        self.record_button = RecordButton(viewer, parent=self)
        self.hide_label_text_cb = QCheckBox("Hide label text")
        self.hide_label_text_cb.setToolTip(
            "Hide the label name overlay shown on the video canvas during playback"
        )
        self.hide_label_text_cb.toggled.connect(self._on_hide_label_text_toggled)

        play_record_row = QHBoxLayout()
        play_record_row.addWidget(self.play_pause_btn)
        play_record_row.addWidget(self.record_button)
        playback_layout.addLayout(play_record_row, 3, 0, 1, 2)
        playback_layout.addWidget(self.hide_label_text_cb, 3, 2)

        # ── Assemble ─────────────────────────────────────────────────
        main_layout.addWidget(navigate_group)
        main_layout.addWidget(playback_group)
        self.setLayout(main_layout)

        # Restore saved mode
        saved = app_state.get_with_default("restrict_mode")
        idx = RESTRICT_MODES.index(saved.capitalize()) if saved.capitalize() in RESTRICT_MODES else 0
        self.mode_combo.setCurrentIndex(idx)

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
        mode = self.mode_combo.currentText().lower()
        if mode in ("trial", "video"):
            self._navigate_trial(direction)
        elif mode == "label":
            self._navigate_label(direction)
        elif mode == "sequence":
            self._navigate_sequence(direction)

    # ==================================================================
    # Mode switching
    # ==================================================================

    def _on_mode_changed(self, mode_text: str):
        mode = mode_text.lower()
        self.app_state.restrict_mode = mode
        self._stack.setCurrentIndex(RESTRICT_MODES.index(mode_text))

        self.trials_combo.setEnabled(mode in ("trial", "video"))

        if mode == "video":
            self._populate_video_combo()
            self._apply_video_restriction()
        elif mode == "label":
            self._refresh_label_instances()
            self._apply_label_restriction()
        elif mode == "sequence":
            self._on_sequence_search()
        else:
            self._apply_trial_restriction()

        self._update_counter()

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
        try:
            self.app_state.set_key_sel("trials", trials_sel)
        except KeyError:
            self.app_state.trials_sel = trials[0]
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
            # Sync video combo if it exists
            if hasattr(self, 'video_combo') and self.video_combo.count() > 0:
                self.video_combo.blockSignals(True)
                self.video_combo.setCurrentIndex(new_idx)
                self.video_combo.blockSignals(False)
            self.app_state.trial_changed.emit()
            self._update_counter()
            tb = self.app_state.trial_bounds
            if tb:
                self._center_and_maybe_play(tb.start_s, tb.end_s)

    def _apply_trial_restriction(self):
        alignment = getattr(self.app_state, "trial_alignment", None)
        if alignment is None:
            return
        trial_id = getattr(self.app_state, "trials_sel", None)
        rw = build_trial_window(
            alignment, trial_id,
            extra_t0=self.extra_t0_spin.value(),
            extra_t1=self.extra_t1_spin.value(),
        )
        self.app_state.restrict_window = rw

    # ==================================================================
    # Video mode
    # ==================================================================

    def _populate_video_combo(self):
        """Fill video_combo with filenames from the primary camera."""
        import os

        self.video_combo.blockSignals(True)
        self.video_combo.clear()
        sio = getattr(self.app_state, "nwb_alignment", None)
        if sio is None:
            self.video_combo.blockSignals(False)
            return
        camera = getattr(self.app_state, "primary_camera", None)
        trials = getattr(self.app_state, "trials", [])
        for tid in trials:
            path = sio.resolve_media_path(tid, "video", device=camera)
            name = os.path.basename(path) if path else str(tid)
            self.video_combo.addItem(name, tid)
        current = getattr(self.app_state, "trials_sel", None)
        if current in trials:
            self.video_combo.setCurrentIndex(trials.index(current))
        self.video_combo.blockSignals(False)

    def _on_video_combo_changed(self, index: int):
        if not self.app_state.ready or index < 0:
            return
        tid = self.video_combo.itemData(index)
        if tid is None:
            return
        current = getattr(self.app_state, "trials_sel", None)
        if tid != current:
            self.app_state.trials_sel = tid
            self.trials_combo.blockSignals(True)
            self.trials_combo.setCurrentText(str(tid))
            self.trials_combo.blockSignals(False)
            self.app_state.trial_changed.emit()
        self._apply_video_restriction()
        self._update_counter()
        tb = self.app_state.trial_bounds
        if tb:
            self._center_and_maybe_play(tb.start_s, tb.end_s)

    def _apply_video_restriction(self):
        """Restrict time range to current trial's primary camera video duration."""
        from ethograph.io.time_model import RestrictionWindow, TimeRange

        alignment = getattr(self.app_state, "trial_alignment", None)
        video = getattr(self.app_state, "video", None)
        if alignment is None or video is None:
            self._apply_trial_restriction()
            return

        vid_start = alignment.video_offset
        vid_end = vid_start + video.total_duration
        extra_t0 = self.extra_t0_spin.value()
        extra_t1 = self.extra_t1_spin.value()

        core = TimeRange(vid_start, vid_end)
        time_range = TimeRange(vid_start - extra_t0, vid_end + extra_t1)
        trial_id = getattr(self.app_state, "trials_sel", None)

        self.app_state.restrict_window = RestrictionWindow(
            mode="video", time_range=time_range, core_range=core,
            trial_id=trial_id,
        )
        self._video_info_label.setText(
            f"Video: {vid_start:.2f}s \u2013 {vid_end:.2f}s  ({video.total_duration:.2f}s)"
        )

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
        if not self.app_state.ready or self.app_state.restrict_mode != "label":
            return
        self._refresh_label_instances()
        self._apply_label_restriction()

    def _on_label_filter_changed(self):
        if not self.app_state.ready or self.app_state.restrict_mode != "label":
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
        mode = self.mode_combo.currentText().lower()
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

        extra_t0 = self.extra_t0_spin.value()
        extra_t1 = self.extra_t1_spin.value()

        master = getattr(self.plot_container, "_xlink_master", None) or getattr(self.plot_container, "_feature_plot", None)
        if master is not None:
            master.vb.setXRange(onset_s - extra_t0, offset_s + extra_t1, padding=0)

        self.plot_container.update_time_marker_by_time(onset_s)
        if hasattr(self.plot_container, "time_slider"):
            self.plot_container.time_slider.set_slider_time(onset_s)

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
    # Extra context
    # ==================================================================

    def _on_extra_context_changed(self):
        self.app_state.restrict_extra_t0 = self.extra_t0_spin.value()
        self.app_state.restrict_extra_t1 = self.extra_t1_spin.value()
        mode = self.app_state.restrict_mode
        if mode == "video":
            self._apply_video_restriction()
        elif mode == "trial":
            self._apply_trial_restriction()
        elif mode == "label":
            self._apply_label_restriction()
        elif mode == "sequence":
            self._apply_sequence_restriction()

    # ==================================================================
    # Jump to time
    # ==================================================================

    def _on_jump_to_time(self):
        sio = getattr(self.app_state, "nwb_alignment", None)
        trials = getattr(self.app_state, "trials", None)
        if sio is None or not trials:
            return
        global_t = self.jump_time_spin.value()
        try:
            trial_id, _rel_t = find_closest_trial(sio, trials, global_t)
        except ValueError:
            logger.warning("Cannot jump: no trial timing info")
            return
        self.app_state.trials_sel = trial_id
        self.trials_combo.blockSignals(True)
        self.trials_combo.setCurrentText(str(trial_id))
        self.trials_combo.blockSignals(False)
        self.app_state.trial_changed.emit()

    def setup_trial_conditions(self, catalog):
        """No-op — trial condition filtering moved to TrialsWidget."""
        self.catalog = catalog

    # ==================================================================
    # Playback
    # ==================================================================

    def _on_play_pause_clicked(self):
        if hasattr(self, "_data_widget") and self._data_widget is not None:
            self._data_widget.toggle_pause_resume()
        self._sync_play_icon()

    def _sync_play_icon(self):
        video = getattr(self.app_state, "video", None)
        playing = video.is_playing if video else False
        self.play_pause_btn.setText("\u23F8" if playing else "\u25B6")

    def connect_video_sync(self, sync):
        """Connect playback_stopped signal to reset the play button icon."""
        if getattr(self, "_connected_sync", None) is sync:
            return
        if getattr(self, "_connected_sync", None) is not None:
            try:
                self._connected_sync.playback_stopped.disconnect(self._sync_play_icon)
            except (RuntimeError, TypeError):
                pass
        sync.playback_stopped.connect(self._sync_play_icon)
        self._connected_sync = sync

    def _on_fps_changed(self):
        fps_playback = float(self.fps_playback_edit.text())
        self.app_state.fps_playback = fps_playback
        qt_dims = self.viewer.window.qt_viewer.dims
        if qt_dims.slider_widgets:
            slider_widget = qt_dims.slider_widgets[0]
            slider_widget._update_play_settings(fps=fps_playback, loop_mode="once", frame_range=None)
        if self.app_state.av_speed_coupled and self.app_state.video:
            recording_fps = self.app_state.video_fps
            audio_speed = fps_playback / recording_fps
            self.app_state.audio_playback_speed = audio_speed
            self.audio_speed_spin.blockSignals(True)
            self.audio_speed_spin.setValue(audio_speed)
            self.audio_speed_spin.blockSignals(False)

    def _on_audio_speed_changed(self, value: float):
        self.app_state.audio_playback_speed = value
        if self.app_state.av_speed_coupled and self.app_state.video:
            recording_fps = self.app_state.video_fps
            fps_playback = value * recording_fps
            self.app_state.fps_playback = fps_playback
            self.fps_playback_edit.blockSignals(True)
            self.fps_playback_edit.setText(str(fps_playback))
            self.fps_playback_edit.blockSignals(False)

    def _on_coupling_toggled(self, checked: bool):
        self.app_state.av_speed_coupled = checked
        self.coupling_button.setText("\U0001f517" if checked else "\U0001f513")

    def _on_hide_label_text_toggled(self, checked: bool):
        labels_widget = getattr(self, '_labels_widget', None)
        if labels_widget is None:
            return
        labels_widget._label_overlay_hidden = checked
        overlay = getattr(labels_widget, '_label_overlay', None)
        if overlay is None:
            return
        if checked:
            overlay.hide()
        else:
            labels_widget._label_overlay_last_text = ""
            if hasattr(labels_widget, '_update_labels_text'):
                labels_widget._update_labels_text()

    def _step_frame(self, direction: int):
        if not self.app_state.ready:
            return
        if self.app_state.video:
            video = self.app_state.video
            new_frame = self.app_state.current_frame + direction
            new_frame = max(0, min(new_frame, self.app_state.num_frames - 1))
            video.seek_to_frame(new_frame)
        else:
            self._step_time_no_video(direction)

    def _step_window(self, direction: int):
        if not self.app_state.ready:
            return
        self._step_time_no_video(direction)
        video = self.app_state.video    
        if video and self.plot_container:                                                                                    
            new_time = self.plot_container.time_slider.current_time                                                                                         
            frame = video.time_to_frame(new_time)
            video.blockSignals(True)
            video.seek_to_frame(frame)
            video.blockSignals(False)  



    def _step_time_no_video(self, direction: int):
        if not self.plot_container:
            return
        slider = self.plot_container.time_slider
        jump_s = self.app_state.time_jump_ms / 1000.0
        new_time = slider.current_time + direction * jump_s
        new_time = max(slider._t_min, min(new_time, slider._t_max))
        slider.set_slider_time(new_time)
        self.plot_container.update_time_marker_by_time(new_time)
        center = getattr(self.app_state, "center_playback", False)
        xlim = self.plot_container.get_current_xlim()
        if center or new_time < xlim[0] or new_time > xlim[1]:
            window_size = self.app_state.get_with_default("window_size")
            half = window_size / 2.0
            master = self.plot_container._xlink_master or self.plot_container._feature_plot
            master.vb.setXRange(new_time - half, new_time + half, padding=0)

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
