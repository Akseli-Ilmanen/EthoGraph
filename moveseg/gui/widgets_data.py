"""Widget for selecting start/stop times and playing a segment in napari."""

import os
from pathlib import Path
from typing import Dict, Optional

import napari
import numpy as np
import xarray as xr
from movement.filtering import rolling_filter, savgol_filter
from movement.napari.loader_widgets import DataLoader
from napari.utils.notifications import show_error
from napari.viewer import Viewer
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from ethograph.gui.parameters import create_function_selector
from ethograph.utils.data_utils import sel_valid
from .data_loader import load_dataset
from .plots_space import SpacePlot
from .plots_spectrogram import SharedAudioCache
from .video_sync import NapariVideoSync

# PyAV streamer with fixes 
from napari_pyav._reader import FastVideoReader


class DataWidget(DataLoader, QWidget):
    """Widget to control which data is loaded, displayed and stored for next time."""

    def __init__(
        self,
        napari_viewer: Viewer,
        app_state,
        meta_widget,
        io_widget,
        parent=None,
    ):
        DataLoader.__init__(self, napari_viewer)  # Pass required args for DataLoader
        QWidget.__init__(self, parent=parent)
        self.parent = parent
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())
        self.app_state = app_state
        self.meta_widget = meta_widget
        self.io_widget = io_widget  
        self.video = None  
        self.plot_container = None
        self.labels_widget = None
        self.plots_widget = None  
        self.audio_player = None 
        self.video_path = None
        self.audio_path = None
        self.space_plot = None  

        # Dictionary to store all combo boxes
        self.combos = {}
        # Dictionary to store all controls for enabling/disabling
        self.controls = []

        # Tracking stuff
        self.fps = None
        self.source_software = None
        self.file_path = None
        self.file_name = None

        self.app_state.audio_video_sync = None
        # E.g. {keypoints = ["beakTip, StickTip"], trials=[1, 2, 3, 4], ...}
        self.type_vars_dict = {}  # Gets filled by load_dataset
        
        

    def set_references(self, plot_container, labels_widget, plots_widget, navigation_widget):
        """Set references to other widgets after creation."""
        self.plot_container = plot_container
        self.labels_widget = labels_widget
        self.plots_widget = plots_widget
        self.navigation_widget = navigation_widget
        

    


    def on_load_clicked(self):
        """Load the file and show line plot in napari dock."""
        
        if not self.app_state.nc_file_path:
            show_error("Please select a path ending with .nc")
            return
        
        
        self.setVisible(False)

        # Load ds
        nc_file_path = self.io_widget.get_nc_file_path()
        
        
        self.app_state.dt, label_dt, self.type_vars_dict = load_dataset(nc_file_path)
        trials = self.app_state.dt.trials
        
        
        if self.app_state.import_labels_nc_data:
            self.app_state.label_dt = label_dt
            
        else:
            self.app_state.label_dt = self.app_state.dt.get_label_dt(empty=True)
            
        self.app_state.label_ds = self.app_state.label_dt.sel(trials=trials[0])


        
        self.app_state.ds = self.app_state.dt.sel(trials=trials[0])
        


        self.update_trials_combo()
        self.app_state.trials = sorted(trials)
        
        
        

        self._create_trial_controls()

        
        self._restore_or_set_defaults()
        self._set_controls_enabled(True)
        self.app_state.ready = True

        self.navigation_widget._trial_change_consequences()


        load_btn = self.io_widget.load_button
        load_btn.setEnabled(False)
        load_btn.setText("Restart app to load new data")

        self.setVisible(True)

        self._force_layout_update()

            
    def update_trials_combo(self) -> None:
        """Update trials combo box with verification status color coding in dropdown."""
        if not self.app_state.ready:
            return
        
        combo = self.navigation_widget.trials_combo
        combo.blockSignals(True)
        combo.clear()
        
      
        trial_status = self._collect_trial_status()
        
    
        for trial in self.app_state.trials:
            combo.addItem(str(trial))
            
      
            index = combo.count() - 1
            is_verified = trial_status.get(trial)
            
      
            bg_color = QColor(144, 238, 144) if is_verified else QColor(255, 182, 193)  # 
            combo.setItemData(index, bg_color, Qt.BackgroundRole)
            

            text_color = QColor(0, 100, 0) if is_verified else QColor(139, 0, 0)  
            combo.setItemData(index, text_color, Qt.ForegroundRole)
        
        
        combo.setCurrentText(str(self.app_state.trials_sel))
        combo.blockSignals(False)


    def _collect_trial_status(self) -> Dict[int, int]:
        """Extract verification status for each trial from data tree."""
        trial_status = {}
        
        for trial in self.app_state.trials:
            is_verified = self.app_state.label_dt.sel(trials=trial).attrs.get('human_verified', 0)
            trial_status[trial] = bool(is_verified)

        return trial_status


    def zero_all_datasets(self, dt: xr.DataTree) -> xr.DataTree:
        """Set all values in all datasets to zero."""
        for node in dt.subtree:
            if node.ds is not None:
                for var in node.ds.data_vars:
                    node.ds[var].values[:] = 0
        return dt

    def _create_trial_controls(self):
        """Create all trial-related controls based on info configuration."""

        # Create device combos in IOWidget
        self.io_widget.create_device_controls(self.type_vars_dict)

        
        if 'position' in self.app_state.ds.data_vars:
            self.space_plot_combo = QComboBox()
            self.space_plot_combo.setObjectName("space_plot_combo")
            
            self.space_plot_combo.addItems(["Layer controls", "space_2D", "space_3D"])        
            self.space_plot_combo.currentTextChanged.connect(self._on_space_plot_changed)
            self.controls.append(self.space_plot_combo)
        
            functions = {
                "savgol_filter": {
                    "func": savgol_filter,
                    "docs": "https://movement.neuroinformatics.dev/api/movement.filtering.savgol_filter.html"
                },
                "rolling_filter": {
                    "func": rolling_filter,
                    "docs": "https://movement.neuroinformatics.dev/api/movement.filtering.rolling_filter.html"
                },
            }
            func_combo, settings_btn, docs_btn = create_function_selector(functions, parent=self.viewer.window.qt_viewer
                                                                        , on_execute=self._smooth_tracks)
            
            
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(self.space_plot_combo)
            row_layout.addWidget(func_combo.native)
            row_layout.addWidget(settings_btn.native)
            row_layout.addWidget(docs_btn.native)
            self.layout().addRow(row_widget)
        
        

        # 5. Add gap (empty row) for separation
        gap_label = QLabel("")
        gap_label.setFixedHeight(10)  # Create visual gap
        self.layout().addRow(gap_label)

        # 6. Now add remaining controls
        remaining_type_vars = ["individuals", "keypoints", "features", "colors", "trial_conditions"]
        
        
        
        for type_var in remaining_type_vars:
            if type_var in self.type_vars_dict.keys():
 
                if type_var == "features" and self.app_state.audio_folder:
                    features_list = list(self.type_vars_dict[type_var])
 
       
                    features_with_spec = features_list + ["Spectrogram"]
                    self._create_combo_widget(type_var, features_with_spec)
                else:
                    self._create_combo_widget(type_var, self.type_vars_dict[type_var])



        self.show_confidence_checkbox = QCheckBox("Show confidence")
        self.show_confidence_checkbox.setChecked(False)
        self.show_confidence_checkbox.stateChanged.connect(self._on_show_confidence_changed)
        self.layout().addRow(self.show_confidence_checkbox)
        




        # Initially disable trial controls until data is loaded
        self._set_controls_enabled(False)
        
    

    def _on_show_confidence_changed(self):
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

        
        
    def _smooth_tracks(self, func, **kwargs):
        
        ds_orig = self.app_state.ds

        ds_kwargs = self.app_state.get_ds_kwargs()
        _, valid_kwargs = sel_valid(ds_orig.position, ds_kwargs)
        
        
        ds_temp = ds_orig.copy()
        ds_temp.update(
            {
                "position": func(
                    ds_orig.position, **kwargs
                )
            }
        )
        self.app_state.ds = ds_temp
        self.update_main_plot()
        self.update_space_plot()
        self.app_state.ds = ds_orig 
          

    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable all trial-related controls."""
        for control in self.controls:
            control.setEnabled(enabled)
        # Also enable/disable IOWidget device controls
        self.io_widget.set_controls_enabled(enabled)
        self.app_state.ready = enabled

    def _force_layout_update(self):
        """Force Qt to recalculate layout by toggling collapsible widget."""
        if self.meta_widget and hasattr(self.meta_widget, 'collapsible_widgets'):
            io_collapsible = self.meta_widget.collapsible_widgets[1]
            io_collapsible.collapse()
            QApplication.processEvents()
            io_collapsible.expand()

    def _create_combo_widget(self, key, vars):
        """Create a combo box widget for a given info key."""

        combo = QComboBox()
        combo.setObjectName(f"{key}_combo")
        combo.currentTextChanged.connect(self._on_combo_changed)
        if key in ["colors", "trial_conditions"]:
            colour_variables = ["None"] + [str(var) for var in vars]
            combo.addItems(colour_variables)
        else:
            combo.addItems([str(var) for var in vars])
        
        self.layout().addRow(f"{key.capitalize()}:", combo)

        self.combos[key] = combo
        self.controls.append(combo)

        if key == "features":
            self.feature_dims_input = QLineEdit()
            self.feature_dims_input.setObjectName("feature_dims_input")
            dim = self._get_dim()
            if dim:
                values = [str(v) for v in self.app_state.ds[dim].values.tolist()]
                completer = QCompleter(["All"] + values)
                completer.setMaxVisibleItems(10) 
                self.feature_dims_input.setCompleter(completer)
            else:
                completer = QCompleter(["N/A"])
                self.feature_dims_input.setCompleter(completer)

            self.layout().addRow("Feature Dims:", self.feature_dims_input)

            
    
            self.feature_dims_input.textChanged.connect(self._dim_changed)
            self.controls.append(self.feature_dims_input)
            self.combos["feature_dims"] = self.feature_dims_input
            
            

        if key == "trial_conditions":
            self.trial_conditions_value_combo = QComboBox()
            self.trial_conditions_value_combo.setObjectName("trial_condition_value_combo")
            self.trial_conditions_value_combo.addItem("None")
            self.trial_conditions_value_combo.currentTextChanged.connect(self._on_trial_condition_values_changed)
            self.layout().addRow("Filter by condition:", self.trial_conditions_value_combo)
            self.controls.append(self.trial_conditions_value_combo)
        return combo

    def _get_dim(self):
        """Return the non-time dimension for a given feature in ds."""
        if not hasattr(self.app_state, 'features_sel') or self.app_state.features_sel == "Spectrogram":
            return None
        
        ds = self.app_state.ds
        arr = ds[self.app_state.features_sel]

        excluded = ["time"] + list(self.type_vars_dict.keys())
        if all(dim in excluded for dim in arr.dims):
            return None
        
        dim = next(dim for dim in arr.dims if dim not in excluded)
        return dim

    @staticmethod
    def is_number(s: str) -> bool:
        """Check if string can be converted to a number."""
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    def _dim_changed(self):
        dim = self._get_dim()
        if dim:
            input = self.feature_dims_input.text().strip()

            if not input or input == "All":
                return
            
            if self.is_number(input):
                input = float(input) if '.' in input else int(input)

            self.app_state.set_key_sel(dim, input)
            xmin, xmax = self.plot_container.get_current_xlim()
            self.update_main_plot(t0=xmin, t1=xmax)


    def _on_combo_changed(self):
        if not self.app_state.ready:
            return

        # Figure out which key this belongs to
        combo = self.sender()
        key = None

        # Check IOWidget combos first
        for io_key, io_value in self.io_widget.combos.items():
            if io_value is combo:
                key = io_key
                break

        # Check DataWidget combos if not found in IOWidget
        if key is None:
            for data_key, data_value in self.combos.items():
                if data_value is combo:
                    key = data_key
                    break

        if key:
            selected_value = combo.currentText()
            self.app_state.set_key_sel(key, selected_value)

            if key == "features":
                if selected_value == "Spectrogram":
                    self.plot_container.switch_to_spectrogram()
                else:
                    self.plot_container.switch_to_lineplot()

                    
                    
                current_plot = self.plot_container.get_current_plot()
                xmin, xmax = current_plot.get_current_xlim()
                
                  
                        
                dim = self._get_dim()
                if dim:
                    values = [str(v) for v in self.app_state.ds[dim].values.tolist()]
                    completer = QCompleter(["All"] + values)
                    completer.setMaxVisibleItems(10)  
                    self.feature_dims_input.setCompleter(completer)
                    self.feature_dims_input.clear()
    
                    if len(values) > 5:
                        self.feature_dims_input.setText(values[0])
                        input = self.feature_dims_input.text().strip()
                        if self.is_number(input):
                            input = float(input) if '.' in input else int(input)
                        self.app_state.set_key_sel(dim, input)

                        self.app_state.set_key_sel(dim, input)

                else:
                    completer = QCompleter(["N/A"])
                    self.feature_dims_input.setCompleter(completer)
 
                    self.feature_dims_input.setText("N/A")
 
            if key in ["cameras", "mics"]:
                self.update_video_audio()

            if key in ["features", "colors", "individuals", "keypoints", "feature_dims", "mics"]:
                current_plot = self.plot_container.get_current_plot()
                xmin, xmax = current_plot.get_current_xlim()
                self.update_main_plot(t0=xmin, t1=xmax)

            if key in ["individuals", "keypoints", "colors"]:
                self.update_space_plot()


            if key == "tracking":
                self.update_tracking()
        
            if key == "trial_conditions":
                self._update_trial_condition_values()





    def _restore_or_set_defaults(self):
        """Restore saved selections from app_state or set defaults from available options."""

        for key, vars in self.type_vars_dict.items():
            # Check IOWidget combos first, then DataWidget combos
            combo = self.io_widget.combos.get(key) or self.combos.get(key)

            if combo is not None:
                if key == "trial_conditions":
                    # Always default to None
                    combo.setCurrentText("None")
                    self.app_state.set_key_sel(key, "None")
                elif self.app_state.key_sel_exists(key) and self.app_state.get_key_sel(key) in [str(var) for var in vars]:
                    combo.setCurrentText(str(self.app_state.get_key_sel(key)))
                else:
                    # Default to first value
                    combo.setCurrentText(str(vars[0]))
                    self.app_state.set_key_sel(key, str(vars[0]))


        if self.app_state.key_sel_exists("trials"):
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.get_key_sel("trials")))
            self.app_state.trials_sel = self.app_state.get_key_sel("trials")
        else:
            # Default to first value
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials[0]))
            self.app_state.trials_sel = self.app_state.trials[0]
            
        # Restore space plot type
        space_plot_type = getattr(self.app_state, 'space_plot_type', 'None')
        if hasattr(self, 'space_plot_combo'):
            self.space_plot_combo.setCurrentText(space_plot_type)

                
   


    def _update_trial_condition_values(self):
        """Update the trial condition value dropdown based on selected key."""
        filter_condition = self.app_state.trial_conditions_sel

        if filter_condition == "None":
            self.trial_conditions_value_combo.setCurrentText("None")
            return

        self.trial_conditions_value_combo.blockSignals(True)
        self.trial_conditions_value_combo.clear()

        if filter_condition in self.type_vars_dict.get("trial_conditions", []):
    
            filter_values = [node.ds.attrs[filter_condition] for node in self.app_state.dt.children.values()]
            unique_values = np.unique(filter_values)

            self.trial_conditions_value_combo.addItems(["None"] + [str(int(val)) for val in np.sort(unique_values)])

        self.trial_conditions_value_combo.blockSignals(False)

    def _on_trial_condition_values_changed(self):
        """Update the available trials based on condition filtering."""
        filter_condition = self.app_state.trial_conditions_sel
        filter_value = self.trial_conditions_value_combo.currentText()

        original_trials = self.app_state.dt.trials

        if filter_condition != "None" and filter_value != "None":
            filt_dt = self.app_state.dt.filter_by_attr(filter_condition, filter_value)
            
            
            self.app_state.trials = list(set(original_trials) & set(filt_dt.trials))
        else:
            # Reset to all trials
            self.app_state.trials = original_trials

        # Update trials dropdown
        self.navigation_widget.trials_combo.clear()
        self.update_trials_combo()

        # Update current trial if needed
        if self.app_state.trials_sel not in self.app_state.trials:
            if self.app_state.trials:
                self.app_state.trials_sel = self.app_state.trials[0]
                self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials_sel))

        self.navigation_widget.trials_combo.blockSignals(False)
        self.update_main_plot()



    def update_main_plot(self, **kwargs):
        """Update the line plot or spectrogram with current trial/keypoint/variable selection."""
        if not self.app_state.ready:
            return

        ds_kwargs = self.app_state.get_ds_kwargs()
        current_plot = self.plot_container.get_current_plot()

        try:
            # Update the current plot
            current_plot.update_plot(**kwargs)
            self.update_motif_plot(ds_kwargs)

            # Handle confidence plotting based on checkbox state
            if self.show_confidence_checkbox.isChecked():
                try:
                    label_confidence, _ = sel_valid(self.app_state.label_ds.labels_confidence, ds_kwargs)
                    self.plot_container.show_confidence_plot(label_confidence)
                except (KeyError, AttributeError):
                    pass
            else:
                self.plot_container.hide_confidence_plot()
                    
        except (KeyError, AttributeError, ValueError) as e:
            show_error(f"Error updating plot: {e}")


    def update_motif_plot(self, ds_kwargs):
        """Update motif plot with labels and predictions."""
        label_ds = self.app_state.label_ds
        time_data = label_ds.time.values
        labels, _ = sel_valid(label_ds.labels, ds_kwargs)

        predictions = None
        if (
            self.labels_widget.pred_show_predictions.isChecked()
            and hasattr(self.app_state, 'pred_ds')
            and self.app_state.pred_ds is not None
        ):
            pred_ds = self.app_state.pred_ds
            predictions, _ = sel_valid(pred_ds.labels, ds_kwargs)
            self.labels_widget.plot_all_motifs(time_data, labels, predictions)
        else:
            self.labels_widget.plot_all_motifs(time_data, labels)








    def update_video_audio(self):
        """Update video and audio using appropriate sync manager based on sync_state."""
        
  

        if not self.app_state.ready or not self.app_state.video_folder:
            return 

        current_frame = getattr(self.app_state, 'current_frame', 0)
        

        if self.video:
            try:
                self.video.frame_changed.disconnect(self._on_sync_frame_changed)
                self.video.cleanup()              
                for layer in list(self.viewer.layers):
                    if layer.name in ["video", "Video Stream"]:
                        self.viewer.layers.remove(layer)

                
                self.video = None
            except (RuntimeError, TypeError):
                print("Error during clean up")
                pass


        video_path = None
        if self.app_state.video_folder and hasattr(self.app_state, 'cameras_sel'):                    
            video_file = self.app_state.ds.attrs[self.app_state.cameras_sel]
            video_path = os.path.join(self.app_state.video_folder, video_file)
            self.app_state.video_path = os.path.normpath(video_path)


        # Set up audio path if available
        audio_path = None
        if self.app_state.audio_folder and hasattr(self.app_state, 'mics_sel'):
            try:
                audio_file = (
                    self.app_state.ds.attrs[self.app_state.mics_sel]
                )
                audio_path = os.path.join(self.app_state.audio_folder, audio_file)
                self.app_state.audio_path = os.path.normpath(audio_path)
            except (KeyError, AttributeError):
                self.app_state.audio_path = None


        
        video_data = FastVideoReader(self.app_state.video_path, read_format='rgb24')
        video_layer = self.viewer.add_image(video_data, name="video", rgb=True)
        video_index = self.viewer.layers.index(video_layer)
        self.viewer.layers.move(video_index, 0)  # Move to bottom layer
        
        try:
            self.video = NapariVideoSync(
                viewer=self.viewer,
                app_state=self.app_state,
                video_source=self.app_state.video_path,
                audio_source=self.app_state.audio_path
            )
        except Exception as e:
            print(f"Error initializing NapariVideoSync: {e}")
            return
        
        self.video.seek_to_frame(current_frame)
        
        # Connect sync manager frame changes to app state and lineplot
        self.video.frame_changed.connect(self._on_sync_frame_changed)


    def update_motif_label(self):
        """Update motif label display."""
        self.labels_widget.refresh_motif_shapes_layer()


    def toggle_pause_resume(self):
        """Toggle play/pause state of the video/audio stream."""
        if not self.video:
            return
        self.video.toggle_pause_resume()
        
        
    def _on_sync_frame_changed(self, frame_number: int):
        """Handle frame changes from sync manager."""
        self.app_state.current_frame = frame_number
        self.plot_container.update_time_marker_and_window(frame_number)
        

        current_time = frame_number / self.app_state.ds.fps
        xlim = self.plot_container.get_current_xlim()
        if current_time < xlim[0] or current_time > xlim[1]:
            self.plot_container.set_x_range(mode='center', center_on_frame=frame_number)
                
                
                
    def update_tracking(self):
        if not self.app_state.tracking_folder or not hasattr(self.app_state, 'tracking_sel'):
            return
 
        # Remove all previous layers with name "video"
        for layer in list(self.viewer.layers):
            if self.file_name and layer.name in [
                f"tracks: {self.file_name}",
                f"points: {self.file_name}",
                f"boxes: {self.file_name}",
                f"skeleton: {self.file_name}"
            ]:
                self.viewer.layers.remove(layer)

        self.fps = self.app_state.ds.fps
        self.source_software = self.app_state.ds.source_software

        tracking_file = (
            self.app_state.ds.attrs[self.app_state.tracking_sel]
        )

        self.file_path = os.path.join(self.app_state.tracking_folder, tracking_file)
        self.file_name = Path(self.file_path).name

        self._format_data_for_layers()
        self._set_common_color_property()
        self._set_text_property()
        self._add_points_layer()


        # connections = [("nose", "ear_left"), ("nose", "ear_right"), ("ear_left", "neck"), ("ear_right", "neck"),
        #                ("hip_left", "neck"), ("hip_right", "neck"), ("hip_left", "tail_base"), ("hip_right", "tail_base")]
        # self._add_skeleton_layer(connections)
        
        # COMMENTED OUT 
        # self._add_tracks_layer() 
        if self.data_bboxes is not None:
            self._add_boxes_layer()
        self._set_initial_state()

            


    def closeEvent(self, event):
        """Clean up video stream and data cache."""

        SharedAudioCache.clear_cache()

        self.video.stop()

        super().closeEvent(event)


    def _on_space_plot_changed(self):
        """Handle space plot combo change."""
        if not self.app_state.ready:
            return
            
        plot_type = self.space_plot_combo.currentText()
        self.app_state.space_plot_type = plot_type
        self.update_space_plot()

    def update_space_plot(self):
        """Update space plot based on current selection."""
        if not self.app_state.ready:
            return

        plot_type = self.app_state.get_with_default('space_plot_type')

        if plot_type == "Layer controls":
            if self.space_plot:
                self.space_plot.hide()
        else:
            # Create space plot if it doesn't exist
            if not self.space_plot:
                self.space_plot = SpacePlot(self.viewer, self.app_state)
                if self.labels_widget:
                    self.labels_widget.highlight_spaceplot.connect(self._highlight_positions_in_space_plot)

            # Get current selections
            individual = self.combos.get('individuals', None)
            individual_text = individual.currentText() if individual else None
            keypoints = self.combos.get('keypoints', None)
            keypoints_text = keypoints.currentText() if keypoints else None
            color_variable = self.combos.get('colors', None)
            color_variable = color_variable.currentText() if color_variable else None

            if plot_type == "space_3D":
                view_3d = True
            elif plot_type == "space_2D":
                view_3d = False

            self.space_plot.update_plot(individual_text, keypoints_text, color_variable, view_3d)
            self.space_plot.show()
            
            

    def _highlight_positions_in_space_plot(self, start_frame: int, end_frame: int):
        """Highlight positions in space plot based on current frame."""
        if self.space_plot and self.space_plot.dock_widget.isVisible():
            self.space_plot.highlight_positions(start_frame, end_frame)

    def reset_widget_state(self):
        """Reset the data widget to its default state."""
        # Clear all combo boxes in this widget
        for combo in self.combos.values():
            combo.clear()
            combo.addItems(["None"])
            combo.setCurrentText("None")
        
        # Reset checkboxes
        if hasattr(self, 'plot_spec_checkbox'):
            self.plot_spec_checkbox.setChecked(False)
        if hasattr(self, 'clear_audio_checkbox'):
            self.clear_audio_checkbox.setChecked(False)
        
        # Reset space plot combo if it exists
        if hasattr(self, 'space_plot_combo'):
            self.space_plot_combo.clear()
            self.space_plot_combo.addItems(["Layer controls"])
            self.space_plot_combo.setCurrentText("Layer controls")
        
        # Reset trial conditions combo if it exists
        if hasattr(self, 'trial_conditions_value_combo'):
            self.trial_conditions_value_combo.clear()
            self.trial_conditions_value_combo.addItems(["None"])
            self.trial_conditions_value_combo.setCurrentText("None")
            
        # Clear navigation widget trials combo
        if self.navigation_widget and hasattr(self.navigation_widget, 'trials_combo'):
            self.navigation_widget.trials_combo.clear()
        
        # Reset various state variables
        self.type_vars_dict = {}
        self.video_path = None
        self.audio_path = None
        self.fps = None
        self.source_software = None
        self.file_path = None
        self.file_name = None
        
        # Hide space plot if it exists
        if self.space_plot:
            self.space_plot.hide()
            
        print("DataWidget state reset to default")
