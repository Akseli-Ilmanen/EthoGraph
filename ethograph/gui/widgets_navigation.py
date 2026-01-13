"""Enhanced navigation widget with proper sync mode handling."""

from napari import Viewer
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget


class NavigationWidget(QWidget):
    """Widget for trial navigation and sync toggle between video and lineplot."""


    def __init__(self, viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.viewer = viewer
        self.app_state = app_state
        
        self.confidence_skip_combo = QComboBox()
        self.confidence_skip_combo.setObjectName("confidence_skip_combo")
        self.confidence_skip_combo.addItems(["Show all", "Show low confidence only", "Show high confidence only"])
        confidence_label = QLabel("Trial filter:")
        confidence_label.setObjectName("confidence_label")

        # Trial selection combo
        self.trials_combo = QComboBox()
        self.trials_combo.setObjectName("trials_combo")
        self.trials_combo.currentTextChanged.connect(self._on_trial_changed)
        trial_label = QLabel("Trial:")
        trial_label.setObjectName("trial_label")

        # Navigation buttons
        self.prev_button = QPushButton("Previous Trial")
        self.prev_button.setObjectName("prev_button")
        self.prev_button.clicked.connect(lambda: self._update_trial(-1))

        self.next_button = QPushButton("Next Trial")
        self.next_button.setObjectName("next_button")
        self.next_button.clicked.connect(lambda: self._update_trial(1))

        # Playback FPS control
        self.fps_playback_edit = QLineEdit()
        self.fps_playback_edit.setObjectName("fps_playback_edit")
        fps_playback = app_state.get_with_default("fps_playback")
        self.fps_playback_edit.setText(str(fps_playback))
        self.fps_playback_edit.editingFinished.connect(self._on_fps_changed)
        fps_label = QLabel("Playback FPS:")
        fps_label.setObjectName("fps_label")
        self.fps_playback_edit.setToolTip(
            "Playback FPS for video.\n"
            "Note: Video decoding typically caps at ~30-50 fps\n"
            "depending on resolution, codec, and hardware."
            "Audio playback speed is coupled to this setting."
            "Set to recording FPS for normal audio playback."
        )


        row0 = QHBoxLayout()
        row0.addWidget(confidence_label)
        row0.addWidget(self.confidence_skip_combo)
        
        row1 = QHBoxLayout()
        row1.addWidget(trial_label)
        row1.addWidget(self.trials_combo)

        row2 = QHBoxLayout()
        row2.addWidget(self.prev_button)
        row2.addWidget(self.next_button)

        row3 = QHBoxLayout()
        row3.addWidget(fps_label)
        row3.addWidget(self.fps_playback_edit)


        main_layout = QVBoxLayout()
        main_layout.addLayout(row0)
        main_layout.addLayout(row1)
        main_layout.addLayout(row2)
        main_layout.addLayout(row3)
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
            
         
            trial_attrs = self.app_state.label_dt.trial(new_trial).attrs
            
            if "model_confidence" not in trial_attrs:
                break
            
            trial_confidence = trial_attrs["model_confidence"]
            confidence_mode = self.confidence_skip_combo.currentText()
            
            should_skip = (
                (confidence_mode == "Show low confidence only" and trial_confidence == "high") or
                (confidence_mode == "Show high confidence only" and trial_confidence == "low")
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

