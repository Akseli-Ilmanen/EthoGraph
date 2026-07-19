"""Bottom playback control bar: play/pause, time slider, FPS, trial navigation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QWidget,
)

if TYPE_CHECKING:
    from ethograph.gui.app_state import ObservableAppState

logger = logging.getLogger(__name__)

_TIMEBAR_RESOLUTION = 1000


class _InteractiveSlider(QSlider):
    """QSlider that tracks mouse interaction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_dragging = False

    def mousePressEvent(self, event: QMouseEvent):
        self.is_dragging = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.is_dragging = False
        super().mouseReleaseEvent(event)


class BottomPlaybackBar(QWidget):
    """Bottom control bar with play/pause, time slider, FPS, and trial navigation."""

    def __init__(self, app_state: ObservableAppState, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.setStyleSheet("QWidget { background-color: #2a2a2a; color: white; padding: 4px; }")
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 4, 8, 4)

        # Add-panel button — opens the SourcePopup (wired in main_window).
        self.add_panel_btn = QPushButton("➕ Add panel")
        self.add_panel_btn.setFixedWidth(90)
        self.add_panel_btn.setToolTip(
            "Add a panel: drag a source onto the plot area, or press Enter  (Ctrl+N)"
        )
        layout.addWidget(self.add_panel_btn)

        # Play/pause button
        self.play_pause_btn = QPushButton("▶")
        self.play_pause_btn.setFixedWidth(36)
        self.play_pause_btn.setToolTip("Play / Pause  (Space)")
        self.play_pause_btn.clicked.connect(self._on_play_pause_clicked)
        layout.addWidget(self.play_pause_btn)

        # Time slider
        self.time_slider = _InteractiveSlider(Qt.Horizontal)
        self.time_slider.setRange(0, _TIMEBAR_RESOLUTION)
        self.time_slider.setMinimumHeight(20)
        self.time_slider.setTracking(True)
        self.time_slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.time_slider, stretch=1)

        # FPS display/edit
        fps_label = QLabel("FPS:")
        fps_label.setFixedWidth(40)
        layout.addWidget(fps_label)

        self.fps_display = QLineEdit()
        self.fps_display.setFixedWidth(50)
        self.fps_display.setText(str(app_state.get_with_default("fps_playback")))
        self.fps_display.editingFinished.connect(self._on_fps_changed)
        layout.addWidget(self.fps_display)

        # Trial navigation cluster: ◀ Trial <id> (i/n) ▶
        nav_style = """
            QPushButton {
                background-color: transparent;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                color: #dddddd;
                font-size: 13px;
                padding: 0px;
            }
            QPushButton:hover { background-color: #3d3d3d; border-color: #5a5a5a; }
            QPushButton:pressed { background-color: #505050; }
            QPushButton:disabled { color: #555555; border-color: #383838; }
        """

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedSize(28, 24)
        self.prev_btn.setStyleSheet(nav_style)
        self.prev_btn.setToolTip("Previous trial")
        self.prev_btn.setFocusPolicy(Qt.NoFocus)
        self.prev_btn.clicked.connect(self._on_prev_trial)
        layout.addWidget(self.prev_btn)

        self.trial_label = QLabel("Trial - / -")
        self.trial_label.setMinimumWidth(100)
        self.trial_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.trial_label)

        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedSize(28, 24)
        self.next_btn.setStyleSheet(nav_style)
        self.next_btn.setToolTip("Next trial")
        self.next_btn.setFocusPolicy(Qt.NoFocus)
        self.next_btn.clicked.connect(self._on_next_trial)
        layout.addWidget(self.next_btn)

        # Wire app_state signals. trials_sel is a dynamic *_sel attribute with
        # no auto-generated signal — trial changes are announced via trial_changed.
        app_state.current_frame_changed.connect(self._update_slider_from_frame)
        app_state.fps_playback_changed.connect(self._update_fps_display)
        app_state.trial_changed.connect(self._update_trial_label)
        if hasattr(app_state, "ready_changed"):
            app_state.ready_changed.connect(self._update_trial_label)

        self._update_trial_label()
        self._sync_play_icon()

    def _on_play_pause_clicked(self):
        """Toggle playback."""
        if hasattr(self, "_data_widget") and self._data_widget is not None:
            self._data_widget.toggle_pause_resume()
        self._sync_play_icon()

    def _sync_play_icon(self):
        """Update button icon based on playback state."""
        video = getattr(self.app_state, "video", None)
        is_playing = video.is_playing if video else False
        self.play_pause_btn.setText("⏸" if is_playing else "▶")

    def _on_slider_value_changed(self):
        """Handle slider value changes — only seek if user is dragging."""
        if not self.time_slider.is_dragging:
            return
        if self.app_state.video is None or self.app_state.num_frames <= 0:
            return
        slider_pos = self.time_slider.value()
        if self.app_state.num_frames <= 1:
            frame = 0
        else:
            frame = int(slider_pos / _TIMEBAR_RESOLUTION * (self.app_state.num_frames - 1))
        self.app_state.video.seek_to_frame(frame)

    def _update_slider_from_frame(self):
        """Update slider position based on current frame."""
        if self.app_state.num_frames <= 0:
            return
        frame = self.app_state.current_frame
        if self.app_state.num_frames <= 1:
            slider_pos = 0
        else:
            slider_pos = int(frame / (self.app_state.num_frames - 1) * _TIMEBAR_RESOLUTION)
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(slider_pos)
        self.time_slider.blockSignals(False)

    def _update_fps_display(self):
        """Update FPS display."""
        fps = self.app_state.fps_playback
        self.fps_display.blockSignals(True)
        self.fps_display.setText(f"{fps:.1f}")
        self.fps_display.blockSignals(False)

    def _on_fps_changed(self):
        """Handle FPS field edit."""
        text = self.fps_display.text().strip()
        if not text:
            return
        try:
            fps = float(text)
            if fps > 0:
                self.app_state.fps_playback = fps
            else:
                self.fps_display.setText(f"{self.app_state.fps_playback:.1f}")
        except ValueError:
            self.fps_display.setText(f"{self.app_state.fps_playback:.1f}")

    def _update_trial_label(self):
        """Update trial label with the actual trial ID plus positional counter."""
        trials = getattr(self.app_state, "trials", None)
        trials_sel = getattr(self.app_state, "trials_sel", None)
        if trials and trials_sel is not None:
            try:
                idx = trials.index(trials_sel)
                self.trial_label.setText(f"Trial {trials_sel} ({idx + 1}/{len(trials)})")
                self.prev_btn.setEnabled(idx > 0)
                self.next_btn.setEnabled(idx < len(trials) - 1)
            except (ValueError, IndexError):
                self.trial_label.setText(f"Trial {trials_sel}")
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
        else:
            self.trial_label.setText("Trial - / -")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

    def _on_prev_trial(self):
        """Navigate to previous trial."""
        trials = getattr(self.app_state, "trials", None)
        if not trials:
            return
        try:
            curr_idx = trials.index(self.app_state.trials_sel)
        except (ValueError, IndexError):
            return
        if curr_idx > 0:
            new_trial = trials[curr_idx - 1]
            self.app_state.trials_sel = new_trial
            self.app_state.trial_changed.emit()

    def _on_next_trial(self):
        """Navigate to next trial."""
        trials = getattr(self.app_state, "trials", None)
        if not trials:
            return
        try:
            curr_idx = trials.index(self.app_state.trials_sel)
        except (ValueError, IndexError):
            return
        if curr_idx < len(trials) - 1:
            new_trial = trials[curr_idx + 1]
            self.app_state.trials_sel = new_trial
            self.app_state.trial_changed.emit()

    def set_data_widget(self, data_widget):
        """Reference to DataWidget for toggling playback."""
        self._data_widget = data_widget

    def connect_video_sync(self, sync):
        """Connect to video sync to monitor playback state."""
        if getattr(self, "_connected_sync", None) is sync:
            return
        if getattr(self, "_connected_sync", None) is not None:
            try:
                self._connected_sync.playback_stopped.disconnect(self._sync_play_icon)
            except (RuntimeError, TypeError):
                pass
        sync.playback_stopped.connect(self._sync_play_icon)
        self._connected_sync = sync
