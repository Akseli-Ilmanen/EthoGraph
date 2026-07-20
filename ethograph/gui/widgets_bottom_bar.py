"""Bottom playback control bar: play/pause, time slider, FPS, trial navigation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QColor, QIcon, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap
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


def _playback_icon(kind: str, color: str = "#e6e6e6") -> QIcon:
    """Render a crisp antialiased play/pause icon (font glyphs look bad on Windows)."""
    s = 64  # oversampled; QIcon scales down smoothly
    pm = QPixmap(s, s)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    # Stroke with a round-join pen to give the shapes softly rounded corners.
    painter.setPen(QPen(c, s * 0.09, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(c)
    if kind == "play":
        path = QPainterPath()
        path.moveTo(s * 0.34, s * 0.24)
        path.lineTo(s * 0.78, s * 0.50)
        path.lineTo(s * 0.34, s * 0.76)
        path.closeSubpath()
        painter.drawPath(path)
    else:
        bar_w = s * 0.13
        painter.drawRoundedRect(QRectF(s * 0.30, s * 0.26, bar_w, s * 0.48), 2, 2)
        painter.drawRoundedRect(QRectF(s * 0.57, s * 0.26, bar_w, s * 0.48), 2, 2)
    painter.end()
    return QIcon(pm)


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
        self.add_panel_btn.setToolTip("Add a panel: drag a source onto the plot area, or press Enter  (Ctrl+N)")
        layout.addWidget(self.add_panel_btn)

        # Play/pause button
        self._play_icon = _playback_icon("play")
        self._pause_icon = _playback_icon("pause")
        self.play_pause_btn = QPushButton()
        self.play_pause_btn.setIcon(self._play_icon)
        self.play_pause_btn.setIconSize(QSize(16, 16))
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
        self.fps_display.setToolTip(
            "Playback FPS for video.\n"
            "Audio playback speed is coupled to this setting.\n"
            "Set to recording FPS for normal audio playback."
        )
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
        # The slider position itself follows time_marker_updated (see
        # set_data_widget), not current_frame — one time-based mapping both ways.
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
        if video is not None:
            is_playing = video.is_playing
        else:
            audio_player = self._audio_player()
            is_playing = audio_player.playing if audio_player else False
        self.play_pause_btn.setIcon(self._pause_icon if is_playing else self._play_icon)

    def _audio_player(self):
        """No-video playback controller (marker-driven), if available."""
        data_widget = getattr(self, "_data_widget", None)
        plot_container = getattr(data_widget, "plot_container", None)
        return getattr(plot_container, "audio_player", None)

    def _on_slider_value_changed(self):
        """Handle slider value changes — only seek if user is dragging.

        The slider is time-based: its position maps onto ``padded_bounds``
        and seeks the time marker directly, video or not. A loaded video is
        seeked to the matching frame as a side effect.
        """
        if not self.time_slider.is_dragging:
            return
        tr = self.app_state.padded_bounds
        if tr is None or tr.duration <= 0:
            return
        time_s = tr.start_s + self.time_slider.value() / _TIMEBAR_RESOLUTION * tr.duration
        fixed = self.app_state.get_with_default("xlim_mode") == "fixed"
        video = self.app_state.video
        if fixed:
            # Slide BEFORE seeking: the frame_changed handler then finds the
            # marker inside the new window and skips its center-scroll,
            # avoiding a second conflicting setXRange per slider tick.
            self._slide_fixed_window(time_s, move_marker=video is None)
        if video is not None and self.app_state.num_frames > 0:
            video.seek_to_frame(video.time_to_frame(time_s))
        elif not fixed:
            self._seek_marker(time_s)

    def _seek_marker(self, time_s: float):
        """Move the time marker (no video): update it and keep it in view."""
        data_widget = getattr(self, "_data_widget", None)
        plot_container = getattr(data_widget, "plot_container", None)
        if plot_container is not None:
            plot_container._on_slider_time(time_s)

    def _slide_fixed_window(self, t0: float, move_marker: bool = False):
        """Fixed x-limits mode: the slider moves the window's start (t0)."""
        data_widget = getattr(self, "_data_widget", None)
        plot_container = getattr(data_widget, "plot_container", None)
        if plot_container is None:
            return
        span = self.app_state.view_span
        tr = self.app_state.padded_bounds
        if tr is not None:
            t0 = max(tr.start_s, min(t0, tr.end_s - span))
        master = getattr(plot_container, "_xlink_master", None) or getattr(plot_container, "_feature_plot", None)
        if master is not None:
            master.vb.setXRange(t0, t0 + span, padding=0)
        if move_marker:
            plot_container.update_time_marker_by_time(t0)

    def _update_slider_from_time(self, time_s: float):
        """Follow the time marker: every marker move (playback, seek, click)
        emits ``time_marker_updated``; map that time onto the slider."""
        if self.time_slider.is_dragging:
            return
        tr = self.app_state.padded_bounds
        if tr is None or tr.duration <= 0:
            return
        frac = (time_s - tr.start_s) / tr.duration
        self._set_slider_silently(int(max(0.0, min(1.0, frac)) * _TIMEBAR_RESOLUTION))

    def _set_slider_silently(self, slider_pos: int):
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
        plot_container = getattr(data_widget, "plot_container", None)
        if plot_container is not None:
            plot_container.time_marker_updated.connect(self._update_slider_from_time)
            plot_container.audio_player.on_state_changed = self._sync_play_icon

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
