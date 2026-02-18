"""Enhanced navigation widget with proper sync mode handling."""

import warnings
import webbrowser

import numpy as np
from napari import Viewer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)



class NavigationWidget(QWidget):
    """Widget for trial navigation and sync toggle between video and lineplot."""


    def __init__(self, viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.viewer = viewer
        self.app_state = app_state
        self.type_vars_dict = {}

        # === Help buttons ===
        help_layout = QHBoxLayout()
        self.docs_button = QPushButton("📚 Documentation")
        self.docs_button.clicked.connect(lambda: webbrowser.open("https://ethograph.readthedocs.io/en/latest/"))
        help_layout.addWidget(self.docs_button)

        self.github_button = QPushButton("🔗 GitHub Issues")
        self.github_button.clicked.connect(lambda: webbrowser.open("https://github.com/akseli-ilmanen/ethograph/issues"))
        help_layout.addWidget(self.github_button)

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

        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("next_button")
        self.next_button.clicked.connect(lambda: self._update_trial(1))

        self.prev_button = QPushButton("Previous")
        self.prev_button.setObjectName("prev_button")
        self.prev_button.clicked.connect(lambda: self._update_trial(-1))


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
            "depending on resolution, codec, and hardware.\n"
            "Audio playback speed is coupled to this setting.\n"
            "Set to recording FPS for normal audio playback."
        )

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

        self.filter_warnings_checkbox = QCheckBox("Filter Warnings")
        self.filter_warnings_checkbox.setObjectName("filter_warnings_checkbox")
        self.filter_warnings_checkbox.setChecked(app_state.get_with_default("filter_warnings"))
        self.filter_warnings_checkbox.setToolTip(
            "Suppress repetitive warnings (e.g. video seek warnings).\n"
            "When enabled, each warning is shown only once."
        )
        self.filter_warnings_checkbox.toggled.connect(self._on_filter_warnings_changed)
        self._apply_warning_filters(app_state.get_with_default("filter_warnings"))

        navigate_layout.addWidget(self.prev_button, 0, 0)
        navigate_layout.addWidget(self.next_button, 0, 1)
        navigate_layout.addWidget(self.trials_combo, 0, 2, 1, 2)
        navigate_layout.addWidget(fps_label, 1, 0)
        navigate_layout.addWidget(self.fps_playback_edit, 1, 1)
        navigate_layout.addWidget(self.skip_frames_checkbox, 1, 2)
        navigate_layout.addWidget(self.filter_warnings_checkbox, 1, 3)

        # === Main layout ===
        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.addLayout(help_layout)
        main_layout.addWidget(filter_group)
        main_layout.addWidget(navigate_group)
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

    def _on_skip_frames_changed(self, checked: bool):
        self.app_state.skip_frames = checked

    def _on_filter_warnings_changed(self, checked: bool):
        self.app_state.filter_warnings = checked
        self._apply_warning_filters(checked)

    def _apply_warning_filters(self, enabled: bool):
        if enabled:
            warnings.filterwarnings("ignore")
        else:
            warnings.resetwarnings()

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
            filter_values = [node.ds.attrs[filter_condition] for node in self.app_state.dt.children.values()]
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
        if confidence_mode != "Show all" and hasattr(self.app_state, "label_dt") and self.app_state.label_dt:
            confidence_filtered = []
            target_confidence = "low" if confidence_mode == "Low confidence only" else "high"
            for trial in filtered_trials:
                trial_attrs = self.app_state.label_dt.trial(trial).attrs
                if trial_attrs.get("model_confidence") == target_confidence:
                    confidence_filtered.append(trial)
            filtered_trials = set(confidence_filtered)

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

