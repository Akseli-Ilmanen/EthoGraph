"""Enhanced navigation widget with proper sync mode handling."""

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
    QVBoxLayout,
    QWidget,
)

from .app_constants import AUDIO_SPEED_MAX, AUDIO_SPEED_MIN, AUDIO_SPEED_STEP



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
    """Widget for trial navigation and sync toggle between video and lineplot."""


    def __init__(self, viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.viewer = viewer
        self.app_state = app_state
        self.type_vars_dict = {}

        # === Filter trials group ===
        filter_group = QGroupBox("Filter trials")
        filter_layout = QGridLayout()
        filter_group.setLayout(filter_layout)

        self.confidence_skip_combo = QComboBox()
        self.confidence_skip_combo.setObjectName("confidence_skip_combo")
        self.confidence_skip_combo.addItems(["Show all", "Low confidence only", "High confidence only"])
        self.confidence_skip_combo.currentTextChanged.connect(self._on_confidence_filter_changed)
        confidence_label = QLabel("By confidence:")
        confidence_label.setObjectName("confidence_label")

        self.trial_conditions_combo = QComboBox()
        self.trial_conditions_combo.setObjectName("trial_conditions_combo")
        self.trial_conditions_combo.addItem("None")
        self.trial_conditions_combo.currentTextChanged.connect(self._on_trial_conditions_changed)
        self.trial_conditions_label = QLabel("By condition:")
        self.trial_conditions_label.setObjectName("trial_conditions_label")

        self.trial_conditions_value_combo = QComboBox()
        self.trial_conditions_value_combo.setObjectName("trial_condition_value_combo")
        self.trial_conditions_value_combo.addItem("None")
        self.trial_conditions_value_combo.currentTextChanged.connect(self._on_trial_condition_values_changed)
        self.filter_label = QLabel("Value:")
        self.filter_label.setObjectName("filter_label")

        filter_layout.addWidget(confidence_label, 0, 0)
        filter_layout.addWidget(self.confidence_skip_combo, 0, 1, 1, 3)
        filter_layout.addWidget(self.trial_conditions_label, 1, 0)
        filter_layout.addWidget(self.trial_conditions_combo, 1, 1)
        filter_layout.addWidget(self.filter_label, 1, 2)
        filter_layout.addWidget(self.trial_conditions_value_combo, 1, 3)

        # === Navigate trials group ===
        navigate_group = QGroupBox("Navigate trials")
        navigate_layout = QGridLayout()
        navigate_group.setLayout(navigate_layout)

        self.trials_combo = QComboBox()
        self.trials_combo.setEditable(True)
        self.trials_combo.setObjectName("trials_combo")
        self.trials_combo.currentTextChanged.connect(self._on_trial_changed)
        self.trials_combo.currentIndexChanged.connect(self._sync_trials_combo_color)

        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("next_button")
        self.next_button.clicked.connect(lambda: self._update_trial(1))

        self.prev_button = QPushButton("Previous")
        self.prev_button.setObjectName("prev_button")
        self.prev_button.clicked.connect(lambda: self._update_trial(-1))

        navigate_layout.addWidget(self.prev_button, 0, 0)
        navigate_layout.addWidget(self.next_button, 0, 1)
        navigate_layout.addWidget(self.trials_combo, 0, 2, 1, 2)

        # === Video/Audio playback group ===
        playback_group = QGroupBox("Video/Audio playback")
        playback_layout = QGridLayout()
        playback_group.setLayout(playback_layout)

        self.fps_playback_edit = QLineEdit()
        self.fps_playback_edit.setObjectName("fps_playback_edit")
        fps_playback = app_state.get_with_default("fps_playback")
        self.fps_playback_edit.setText(str(fps_playback))
        self.fps_playback_edit.editingFinished.connect(self._on_fps_changed)
        self.fps_label = QLabel("Playback FPS:")
        self.fps_label.setObjectName("fps_label")
        self.fps_playback_edit.setToolTip(
            "Playback FPS for video.\n"
            "Note: Video decoding typically caps at ~30-50 fps\n"
            "depending on resolution, codec, and hardware.\n"
            "Audio playback speed is coupled to this setting.\n"
            "Set to recording FPS for normal audio playback."
        )

        self.audio_speed_label = QLabel("Audio speed:")
        self.audio_speed_label.setObjectName("audio_speed_label")
        self.audio_speed_spin = QDoubleSpinBox()
        self.audio_speed_spin.setObjectName("audio_speed_spin")
        self.audio_speed_spin.setRange(AUDIO_SPEED_MIN, AUDIO_SPEED_MAX)
        self.audio_speed_spin.setSingleStep(AUDIO_SPEED_STEP)
        self.audio_speed_spin.setDecimals(2)
        self.audio_speed_spin.setSuffix("\u00d7")
        self.audio_speed_spin.setValue(app_state.get_with_default("audio_playback_speed"))
        self.audio_speed_spin.setToolTip(
            "Audio playback speed multiplier.\n"
            "1.0\u00d7 = normal speed, 2.0\u00d7 = double speed.\n"
            "Speed is achieved by adjusting sample rate\n"
            "(pitch changes proportionally)."
        )
        self.audio_speed_spin.valueChanged.connect(self._on_audio_speed_changed)

        self.coupling_button = QPushButton("\U0001f517")
        self.coupling_button.setObjectName("coupling_button")
        self.coupling_button.setCheckable(True)
        self.coupling_button.setChecked(app_state.get_with_default("av_speed_coupled"))
        self.coupling_button.setToolTip(
            "Link video FPS and audio speed.\n"
            "When linked, changing one adjusts the other\n"
            "to maintain synchronized playback."
        )
        self.coupling_button.setFixedWidth(30)
        self.coupling_button.toggled.connect(self._on_coupling_toggled)

        self.skip_frames_checkbox = QCheckBox("Skip Frames")
        self.skip_frames_checkbox.setObjectName("skip_frames_checkbox")
        self.skip_frames_checkbox.setChecked(app_state.get_with_default("skip_frames"))
        self.skip_frames_checkbox.setToolTip(
            "Skip frames during playback to maintain speed.\n"
            "When enabled, frames are dropped so playback\n"
            "tries to keeps up with the requested FPS instead of\n"
            "slowing down when rendering can't keep pace.\n"
            "Note: For play segment (Press 'v'), audio-video\n"
            "sync may not be accurate during skip frames."
        )
        self.skip_frames_checkbox.toggled.connect(self._on_skip_frames_changed)

        self.center_playback_checkbox = QCheckBox("Center playback")
        self.center_playback_checkbox.setObjectName("center_playback_checkbox")
        self.center_playback_checkbox.setChecked(app_state.get_with_default("center_playback"))
        self.center_playback_checkbox.setToolTip(
            "When enabled, the view continuously scrolls to keep\n"
            "the current playback position centered."
        )
        self.center_playback_checkbox.toggled.connect(self._on_center_playback_changed)

        self.time_jump_label = QLabel("Jump step (ms):")
        self.time_jump_label.setObjectName("time_jump_label")
        self.time_jump_spin = QDoubleSpinBox()
        self.time_jump_spin.setObjectName("time_jump_spin")
        self.time_jump_spin.setRange(1.0, 5000.0)
        self.time_jump_spin.setSingleStep(10.0)
        self.time_jump_spin.setDecimals(0)
        self.time_jump_spin.setSuffix(" ms")
        self.time_jump_spin.setValue(app_state.get_with_default("time_jump_ms"))
        self.time_jump_spin.setToolTip(
            "Time jump for Left/Right arrow keys.\n"
            "In no-video mode, arrows jump by this amount.\n"
            "In video mode, arrows step one frame."
        )
        self.time_jump_spin.valueChanged.connect(self._on_time_jump_changed)

        self.plot_container = None

        playback_layout.addWidget(self.fps_label, 0, 0)
        playback_layout.addWidget(self.fps_playback_edit, 0, 1)
        playback_layout.addWidget(self.skip_frames_checkbox, 0, 2)
        playback_layout.addWidget(self.audio_speed_label, 1, 0)
        playback_layout.addWidget(self.audio_speed_spin, 1, 1)
        playback_layout.addWidget(self.coupling_button, 1, 2)
        playback_layout.addWidget(self.time_jump_label, 2, 0)
        playback_layout.addWidget(self.time_jump_spin, 2, 1)
        playback_layout.addWidget(self.center_playback_checkbox, 2, 2)

        # === Main layout ===
        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.addWidget(filter_group)
        main_layout.addWidget(navigate_group)
        main_layout.addWidget(playback_group)
        self.setLayout(main_layout)

    def _on_trial_changed(self):
        """Handle trial selection change."""
        if not self.app_state.ready:
            return

        trials_sel = self.trials_combo.currentText()
        if not trials_sel or trials_sel.strip() == "":
            return

        try:
            try:
                self.app_state.set_key_sel("trials", trials_sel)
            except KeyError:
                self.app_state.trials_sel = self.app_state.trials[0]

            self.app_state.trial_changed.emit()

            # Reset time to 0 on trial change
            if hasattr(self.app_state, "current_time"):
                self._update_slider_display()
        except ValueError:
            return

    def navigate_to_trial(self, trial_id: str):
        """Navigate to a specific trial by ID (same as selecting it in the combo)."""
        self.trials_combo.setCurrentText(str(trial_id))

    def next_trial(self):
        """Go to the next trial."""
        self._update_trial(1)

    def prev_trial(self):
        """Go to the previous trial."""
        self._update_trial(-1)

    def _update_trial(self, direction: int):
        """Navigate to next/previous trial."""
        if not hasattr(self.app_state, "trials") or not self.app_state.trials:
            return


        curr_idx = self.app_state.trials.index(self.app_state.trials_sel)
        new_idx = curr_idx + direction
        
        while 0 <= new_idx < len(self.app_state.trials):
            new_trial = self.app_state.trials[new_idx]
            
         
            confidence_mode = self.confidence_skip_combo.currentText()
            confidence_levels = getattr(self.app_state, "pred_confidence_levels", {})

            if not confidence_levels or confidence_mode == "Show all":
                break

            trial_confidence = confidence_levels.get(new_trial)
            if trial_confidence is None:
                break

            should_skip = (
                (confidence_mode == "Low confidence only" and trial_confidence == "high") or
                (confidence_mode == "High confidence only" and trial_confidence == "low")
            )
            
            if not should_skip:
                # Found a matching trial
                break
            
            # Skip this trial and continue
            new_idx += direction
    
            


        if 0 <= new_idx < len(self.app_state.trials):
            new_trial = self.app_state.trials[new_idx]
            self.app_state.trials_sel = new_trial
            
            

            # Update combo box without triggering signal
            self.trials_combo.blockSignals(True)
            self.trials_combo.setCurrentText(str(new_trial))
            self.trials_combo.blockSignals(False)

            self.app_state.trial_changed.emit()
            


    def _on_fps_changed(self):
        """Handle playback FPS change from UI."""
        fps_playback = float(self.fps_playback_edit.text())
        self.app_state.fps_playback = fps_playback

        # Update the playback settings in the viewer if using napari mode
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
        """Handle audio speed spinbox change."""
        self.app_state.audio_playback_speed = value
        if self.app_state.av_speed_coupled and self.app_state.video:
            recording_fps = self.app_state.video_fps
            fps_playback = value * recording_fps
            self.app_state.fps_playback = fps_playback
            self.fps_playback_edit.blockSignals(True)
            self.fps_playback_edit.setText(str(fps_playback))
            self.fps_playback_edit.blockSignals(False)

    def _on_coupling_toggled(self, checked: bool):
        """Handle coupling toggle between video FPS and audio speed."""
        self.app_state.av_speed_coupled = checked
        self.coupling_button.setText("\U0001f517" if checked else "\U0001f513")

 


    def _on_skip_frames_changed(self, checked: bool):
        self.app_state.skip_frames = checked

    def _on_center_playback_changed(self, checked: bool):
        self.app_state.center_playback = checked

    def setup_trial_conditions(self, type_vars_dict: dict):
        """Populate trial conditions combo with available conditions."""
        self.type_vars_dict = type_vars_dict

        if "trial_conditions" not in type_vars_dict:
            self.trial_conditions_combo.hide()
            self.trial_conditions_value_combo.hide()
            self.trial_conditions_label.hide()
            self.filter_label.hide()
            return

        self.trial_conditions_combo.blockSignals(True)
        self.trial_conditions_combo.clear()
        self.trial_conditions_combo.addItem("None")
        for condition in type_vars_dict["trial_conditions"]:
            self.trial_conditions_combo.addItem(str(condition))
        self.trial_conditions_combo.blockSignals(False)

    def _on_trial_conditions_changed(self):
        """Handle trial condition key selection change."""
        if not self.app_state.ready:
            return

        selected = self.trial_conditions_combo.currentText()
        self.app_state.set_key_sel("trial_conditions", selected)
        self._update_trial_condition_values()

    def _update_trial_condition_values(self):
        """Update the trial condition value dropdown based on selected key."""
        filter_condition = self.app_state.trial_conditions_sel

        if filter_condition == "None":
            self.trial_conditions_value_combo.blockSignals(True)
            self.trial_conditions_value_combo.clear()
            self.trial_conditions_value_combo.addItem("None")
            self.trial_conditions_value_combo.blockSignals(False)
            return

        self.trial_conditions_value_combo.blockSignals(True)
        self.trial_conditions_value_combo.clear()

        if filter_condition in self.type_vars_dict.get("trial_conditions", []):
            filter_values = [ds.attrs[filter_condition] for _, ds in self.app_state.dt.trial_items() if filter_condition in ds.attrs]
            unique_values = np.unique(filter_values)
            sorted_values = sorted(unique_values, key=str)
            self.trial_conditions_value_combo.addItems(["None"] + [str(v) for v in sorted_values])

        self.trial_conditions_value_combo.blockSignals(False)

    def _on_confidence_filter_changed(self):
        """Update available trials based on confidence filtering."""
        if not self.app_state.ready:
            return
        self._apply_all_filters()

    def _on_trial_condition_values_changed(self):
        """Update the available trials based on condition filtering."""
        if not self.app_state.ready:
            return
        self._apply_all_filters()

    def _apply_all_filters(self):
        """Apply both confidence and condition filters to trials list."""
        if not hasattr(self.app_state, "dt") or not self.app_state.dt:
            return
        original_trials = self.app_state.dt.trials
        filtered_trials = set(original_trials)

        # Apply condition filter
        filter_condition = getattr(self.app_state, "trial_conditions_sel", "None")
        filter_value = self.trial_conditions_value_combo.currentText()
        if filter_condition and filter_condition != "None" and filter_value != "None":
            filt_dt = self.app_state.dt.filter_by_attr(filter_condition, filter_value)
            filtered_trials &= set(filt_dt.trials)

        # Apply confidence filter
        confidence_mode = self.confidence_skip_combo.currentText()
        confidence_levels = getattr(self.app_state, "pred_confidence_levels", {})
        if confidence_mode != "Show all" and confidence_levels:
            target_confidence = "low" if confidence_mode == "Low confidence only" else "high"
            filtered_trials = {t for t in filtered_trials if confidence_levels.get(t) == target_confidence}

        # Sort trials numerically if possible
        try:
            self.app_state.trials = sorted(filtered_trials, key=int)
        except (ValueError, TypeError):
            self.app_state.trials = sorted(filtered_trials)

        self.trials_combo.blockSignals(True)
        self.trials_combo.clear()

        if hasattr(self, 'data_widget') and self.data_widget:
            self.data_widget.update_trials_combo()

        if self.app_state.trials_sel not in self.app_state.trials:
            if self.app_state.trials:
                self.app_state.trials_sel = self.app_state.trials[0]
                self.trials_combo.setCurrentText(str(self.app_state.trials_sel))

        self.trials_combo.blockSignals(False)

        if hasattr(self, 'data_widget') and self.data_widget:
            self.data_widget.update_main_plot()

    def set_data_widget(self, data_widget):
        """Set reference to data widget for callbacks."""
        self.data_widget = data_widget

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

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

    def _on_time_jump_changed(self, value: float):
        self.app_state.time_jump_ms = value

    def step_frame_forward(self):
        self._step_frame(+1)

    def step_frame_backward(self):
        self._step_frame(-1)

    def step_window_forward(self):
        self._step_window(+1)

    def step_window_backward(self):
        self._step_window(-1)

    def _step_frame(self, direction: int):
        if not self.app_state.ready:
            return
        if self.app_state.video:
            self._step_frame_video(direction)
        else:
            self._step_time_no_video(direction)

    def _step_window(self, direction: int):
        if not self.app_state.ready:
            return
        self._step_time_no_video(direction)

    def _step_frame_video(self, direction: int):
        video = getattr(self.app_state, 'video', None)
        if not video:
            return
        new_frame = self.app_state.current_frame + direction
        new_frame = max(0, min(new_frame, self.app_state.num_frames - 1))
        video.seek_to_frame(new_frame)

    def _step_time_no_video(self, direction: int):
        if not self.plot_container:
            return
        slider = self.plot_container.time_slider
        jump_s = self.app_state.time_jump_ms / 1000.0
        new_time = slider.current_time + direction * jump_s
        new_time = max(slider._t_min, min(new_time, slider._t_max))
        slider.set_slider_time(new_time)
        self.plot_container.update_time_marker_by_time(new_time)
        center = getattr(self.app_state, 'center_playback', False)
        xlim = self.plot_container.get_current_xlim()
        if center or new_time < xlim[0] or new_time > xlim[1]:
            window_size = self.app_state.get_with_default("window_size")
            half = window_size / 2.0
            master = self.plot_container._xlink_master or self.plot_container._feature_plot
            master.vb.setXRange(new_time - half, new_time + half, padding=0)

