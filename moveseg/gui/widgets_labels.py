"""Widget for labeling segments in movement data."""
from distinctipy import name
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import numpy as np
import pyqtgraph as pg
from napari.viewer import Viewer
from napari.utils.notifications import show_info
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from napari.utils.notifications import show_warning
from moveseg.features.changepoints import correct_changepoints_one_trial, snap_to_nearest_changepoint
from moveseg.utils.labels import load_motif_mapping, remove_small_blocks
from moveseg.utils.data_utils import sel_valid
from moveseg.utils.io import TrialTree    
import json
import time
import xarray as xr
from qtpy.QtCore import Qt, Signal


class LabelsWidget(QWidget):
    """Widget for labeling movement motifs in time series data."""
    
    highlight_spaceplot = Signal(int, int)

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.app_state = app_state



        self.plot_container = None  # Will be set after creation
        self.data_widget = None

        # Make widget focusable for keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Remove Qt event filter and key event logic
        # Instead, rely on napari's @viewer.bind_key for global shortcuts
        # Shortcut bindings are now handled outside the widget

        # Labeling state
        self.motif_mappings: dict[int, dict[str, Any]] = {}
        self.ready_for_label_click = False
        self.ready_for_play_click = False
        self.first_click = None
        self.second_click = None
        self.selected_motif_id = 0

        # Current motif selection for editing
        self.current_motif_pos: list[int] | None = None  # [start, end] idx of selected motif
        self.current_motif_id: int | None = None  # ID of currently selected motif
        self.current_motif_is_prediction: bool = False  # Whether selected motif is from predictions

        # Edit mode state  
        self.old_motif_pos: list[int] | None = None  # Original position when editing
        self.old_motif_id: int | None = None  # Original ID when editing
        
        # Frame tracking for motif display
        self.previous_frame: int | None = None


        # UI components
        self.motifs_table = None

        self._setup_ui()


        # Use absolute path to mapping.txt in the project root

        
        repo_root = Path(__file__).resolve().parents[2]
        mapping_path = repo_root / "configs" / "mapping.txt"
        self.motif_mappings = load_motif_mapping(mapping_path) # HARD CODED FOR NOW
        self._populate_motifs_table()

        self._sync_disabled = False
        self._in_labeling_operation = False  # Add flag to prevent re-entrance
        self._operation_lock = threading.Lock()  # Thread-safe flag
        


    def _mark_changes_unsaved(self):
        """Mark that changes have been made and are not saved."""
        self.app_state.changes_saved = False

    def set_data_widget(self, data_widget):
        """Set reference to data widget."""
        self.data_widget = data_widget

    def set_meta_widget(self, meta_widget):
        """Set reference to meta widget."""
        self.meta_widget = meta_widget

    def set_plot_container(self, plot_container):
        """Set the plot container reference and connect click handler."""
        self.plot_container = plot_container


        self.plot_container.line_plot.plot_clicked.connect(self._on_plot_clicked)
        if hasattr(self.plot_container, 'spectrogram_plot'):
            self.plot_container.spectrogram_plot.plot_clicked.connect(self._on_plot_clicked)

    def plot_all_motifs(self, time_data, labels, predictions=None):
        """Plot all motifs for current trial and keypoint based on current labels state.

        This implements state-based plotting similar to the MATLAB plot_motifs() function.
        It clears all existing motif rectangles and redraws them based on the current labels.

        Args:
            time_data: Time array for x-axis
            labels: Primary label data to plot as main rectangles
            predictions: Optional prediction data to plot as small rectangles at top
        """
        if labels is None or self.plot_container is None:
            return
        

        current_plot = self.plot_container.get_current_plot()
        if hasattr(current_plot, "label_items"):
            for item in current_plot.label_items:
                try:
                    current_plot.plot_item.removeItem(item)
                except:
                    pass
            current_plot.label_items.clear()

        try:
          
            self._plot_motif_segments(time_data, labels, is_main=True)

    
            if (predictions is not None and
                self.pred_show_predictions.isChecked() and
                hasattr(self.app_state, 'pred_ds') and
                self.app_state.pred_ds is not None):
                self._plot_motif_segments(time_data, predictions, is_main=False)

        except (KeyError, IndexError, AttributeError) as e:
            print(f"Error plotting motifs: {e}")

    def _plot_motif_segments(self, time_data, data, is_main=True):
        """Plot motif segments for a given data array.

        Args:
            time_data: Time array for x-axis
            data: Label/prediction data array
            is_main: If True, plot full-height rectangles; if False, plot small rectangles at top
        """
        current_motif_id = 0
        segment_start = None

        for i, label in enumerate(data):
            if label != 0:  # Start of a motif or continuing one
                if label != current_motif_id:  # New motif starts
                    # End previous motif if it exists
                    if current_motif_id != 0 and segment_start is not None:
                        self._draw_motif_rectangle(
                            time_data[segment_start],
                            time_data[i - 1],
                            current_motif_id,
                            is_main=is_main
                        )

                    # Start new motif
                    current_motif_id = label
                    segment_start = i

            else:  # End of current motif
                if current_motif_id != 0 and segment_start is not None:
                    self._draw_motif_rectangle(
                        time_data[segment_start],
                        time_data[i - 1],
                        current_motif_id,
                        is_main=is_main
                    )
                    current_motif_id = 0
                    segment_start = None

        # Handle case where motif continues to the end
        if current_motif_id != 0 and segment_start is not None:
            self._draw_motif_rectangle(
                time_data[segment_start],
                time_data[-1],
                current_motif_id,
                is_main=is_main
            )

    def _draw_motif_rectangle(self, start_time, end_time, motif_id, is_main=True):
        """Draw motif rectangle using PyQtGraph.

        Args:
            start_time: Start time of the motif
            end_time: End time of the motif
            motif_id: ID of the motif for color mapping
            is_main: If True, draw full-height rectangle; if False, draw small rectangle at top
        """
        if motif_id not in self.motif_mappings:
            return

        color = self.motif_mappings[motif_id]["color"]
        color_rgb = tuple(int(c * 255) for c in color)

        current_plot = self.plot_container.get_current_plot()
        is_spectrogram = self.plot_container.is_spectrogram()

        if is_main:
            if is_spectrogram:
                # Spectrogram: transparent fill with thick colored edges
                y_range = current_plot.plot_item.getViewBox().viewRange()[1]
                y_min, y_max = y_range[0], y_range[1]

                # Very transparent fill rectangle
                rect = pg.LinearRegionItem(
                    values=(start_time, end_time),
                    orientation="vertical",
                    brush=(*color_rgb, 40),  # Nearly transparent fill
                    movable=False,
                )
                rect.setZValue(-10)
                current_plot.plot_item.addItem(rect)
                current_plot.label_items.append(rect)

                # Thick colored edge lines (left, right, top, bottom)
                edge_pen = pg.mkPen(color=(*color_rgb, 255), width=3)

                left_edge = pg.PlotDataItem(
                    [start_time, start_time], [y_min, y_max], pen=edge_pen
                )
                right_edge = pg.PlotDataItem(
                    [end_time, end_time], [y_min, y_max], pen=edge_pen
                )
                top_edge = pg.PlotDataItem(
                    [start_time, end_time], [y_max, y_max], pen=edge_pen
                )
                bottom_edge = pg.PlotDataItem(
                    [start_time, end_time], [y_min, y_min], pen=edge_pen
                )

                for edge in [left_edge, right_edge, top_edge, bottom_edge]:
                    edge.setZValue(-5)
                    current_plot.plot_item.addItem(edge)
                    current_plot.label_items.append(edge)
            else:
                # LinePlot: standard semi-transparent rectangle
                rect = pg.LinearRegionItem(
                    values=(start_time, end_time),
                    orientation="vertical",
                    brush=(*color_rgb, 180),
                    movable=False,
                )
                rect.setZValue(-10)
                current_plot.plot_item.addItem(rect)
                current_plot.label_items.append(rect)
        else:
            # Small rectangle at top for secondary data (predictions)
            y_range = current_plot.plot_item.getViewBox().viewRange()[1]
            y_top = y_range[1]
            y_height = (y_range[1] - y_range[0]) * 0.10

            x_coords = [start_time, end_time, end_time, start_time, start_time]
            y_coords = [y_top, y_top, y_top - y_height, y_top - y_height, y_top]

            rect = pg.PlotDataItem(
                x_coords, y_coords,
                fillLevel=y_top - y_height,
                brush=(*color_rgb, 200),
                pen=None
            )
            rect.setZValue(10)
            current_plot.plot_item.addItem(rect)
            current_plot.label_items.append(rect)

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(2)  # Minimal spacing
        self.setLayout(layout)

        # Create toggle buttons for collapsible sections
        self._create_toggle_buttons()
        layout.addWidget(self.toggle_widget)

        # Create motifs table
        self._create_motifs_table_and_edit_buttons()

        # Create control buttons
        self._create_control_buttons()

        # Add collapsible sections
        layout.addWidget(self.motifs_table)
        layout.addWidget(self.controls_widget)

        # Set initial state: table visible, controls hidden
        self.table_toggle.setText("📋 Motifs Table ✓")
        self.controls_toggle.setText("🎛️ Controls")
        self.controls_widget.hide()

    def _create_toggle_buttons(self):
        """Create toggle buttons for collapsible sections."""
        self.toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        self.toggle_widget.setLayout(toggle_layout)

        # Table toggle button
        self.table_toggle = QPushButton("📋 Motifs Table")
        self.table_toggle.setCheckable(True)
        self.table_toggle.setChecked(True)
        self.table_toggle.clicked.connect(self._toggle_table)
        toggle_layout.addWidget(self.table_toggle)

        # Controls toggle button
        self.controls_toggle = QPushButton("🎛️ Controls")
        self.controls_toggle.setCheckable(True)
        self.controls_toggle.setChecked(False)  # Start with controls collapsed
        self.controls_toggle.clicked.connect(self._toggle_controls)
        toggle_layout.addWidget(self.controls_toggle)

    def _toggle_table(self):
        """Toggle motifs table visibility (mutually exclusive with controls)."""
        if self.table_toggle.isChecked():
            # Show table, hide controls
            self.motifs_table.show()
            self.controls_widget.hide()
            self.table_toggle.setText("📋 Motifs Table ✓")
            self.controls_toggle.setText("🎛️ Controls")
            self.controls_toggle.setChecked(False)
        else:
            # If trying to uncheck table, force controls to be checked instead
            self.controls_widget.show()
            self.motifs_table.hide()
            self.controls_toggle.setText("🎛️ Controls ✓")
            self.table_toggle.setText("📋 Motifs Table")
            self.controls_toggle.setChecked(True)

    def _toggle_controls(self):
        """Toggle controls visibility (mutually exclusive with table)."""
        if self.controls_toggle.isChecked():
            # Show controls, hide table
            self.controls_widget.show()
            self.motifs_table.hide()
            self.controls_toggle.setText("🎛️ Controls ✓")
            self.table_toggle.setText("📋 Motifs Table")
            self.table_toggle.setChecked(False)
        else:
            # If trying to uncheck controls, force table to be checked instead
            self.motifs_table.show()
            self.controls_widget.hide()
            self.table_toggle.setText("📋 Motifs Table ✓")
            self.controls_toggle.setText("🎛️ Controls")
            self.table_toggle.setChecked(True)

    def _create_motifs_table_and_edit_buttons(self):
        """Create the motifs table showing available motif types."""
        self.motifs_table = QTableWidget()
        self.motifs_table.setColumnCount(3)
        self.motifs_table.setHorizontalHeaderLabels(["ID", "Name", "Color"])

        # Hide row numbers (left column)
        self.motifs_table.verticalHeader().setVisible(False)

        # Set column widths
        header = self.motifs_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)  # ID column - fixed width
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Name column - stretches
        header.setSectionResizeMode(2, QHeaderView.Fixed)  # Color column - fixed width

        # Set specific widths for ID and Color columns
        self.motifs_table.setColumnWidth(0, 20)  # ID column narrow
        self.motifs_table.setColumnWidth(2, 20)  # Color column narrow

        self.motifs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.motifs_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.motifs_table.setMaximumHeight(600)  # Limit table height to 600px

        # Connect row selection to activate_motif
        self.motifs_table.itemSelectionChanged.connect(self._on_table_selection_changed)

        # First control row
        first_row = QWidget()
        first_layout = QHBoxLayout()
        first_layout.setSpacing(0)
        first_row.setLayout(first_layout)

        self.changepoint_correction_checkbox = QCheckBox("Label changepoint correction")
        self.changepoint_correction_checkbox.setChecked(True)
        first_layout.addWidget(self.changepoint_correction_checkbox)

        self.delete_button = QPushButton("Delete")
        self.delete_button.setToolTip("Shortcut: (Ctrl + D)")
        self.delete_button.clicked.connect(self._delete_motif)
        first_layout.addWidget(self.delete_button)

        self.edit_button = QPushButton("Edit")
        self.edit_button.setToolTip("Shortcut: (Ctrl + E)")
        self.edit_button.clicked.connect(self._edit_motif)
        first_layout.addWidget(self.edit_button)

        self.play_button = QPushButton("Play")
        self.play_button.setToolTip("Click on segment then press v to play")
        self.play_button.clicked.connect(self._play_segment)
        first_layout.addWidget(self.play_button)

        self.layout().addWidget(first_row)



    def _create_control_buttons(self):
        """Create control buttons for labeling operations."""
        
        self.controls_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(0)
        self.controls_widget.setLayout(layout)


        # Predictions file row
        pred_row = QWidget()
        pred_layout = QHBoxLayout()
        pred_layout.setSpacing(0)
        pred_row.setLayout(pred_layout)

        self.pred_file_path_edit = QLineEdit()
        self.pred_file_path_edit.setPlaceholderText("Predictions file (.nc)")
        self.pred_file_path_edit.setReadOnly(True)
        pred_layout.addWidget(self.pred_file_path_edit)

        self.import_predictions_btn = QPushButton("Get predictions")
        self.import_predictions_btn.setToolTip("Import predictions.nc file from labels\\... folder")
        self.import_predictions_btn.clicked.connect(self._import_predictions_from_file)
        pred_layout.addWidget(self.import_predictions_btn)

        self.pred_show_predictions = QCheckBox("Show predictions")
        self.pred_show_predictions.setEnabled(False)
        self.pred_show_predictions.setChecked(False)
        self.pred_show_predictions.stateChanged.connect(self._on_pred_show_predictions_changed)
        pred_layout.addWidget(self.pred_show_predictions)
        layout.addWidget(pred_row)


        # Control grid (3 rows x 4 columns)
        control_grid = QWidget()
        grid_layout = QGridLayout()
        grid_layout.setSpacing(4)
        grid_layout.setVerticalSpacing(4)
        control_grid.setLayout(grid_layout)

        # Header labels (row 0)
        apply_label = QLabel("Apply")
        apply_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(apply_label, 0, 0)

        to_label = QLabel("to")
        to_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(to_label, 0, 1)



        empty_label = QLabel("to")
        grid_layout.addWidget(empty_label, 0, 2)


        status_label = QLabel("Status")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid_layout.addWidget(status_label, 0, 3)

        # Changepoint correction row (row 1)
        cp_label = QLabel("Changepoint correction")
        grid_layout.addWidget(cp_label, 1, 0)

        self.cp_correction_trial_btn = QPushButton("Single Trial")
        self.cp_correction_trial_btn.clicked.connect(lambda: self._cp_correction("single_trial"))
        grid_layout.addWidget(self.cp_correction_trial_btn, 1, 1)


        self.cp_correction_all_trials_btn = QPushButton("All Trials")
        self.cp_correction_all_trials_btn.clicked.connect(lambda: self._cp_correction("all_trials"))
        grid_layout.addWidget(self.cp_correction_all_trials_btn, 1, 2)

        self.cp_status_btn = QPushButton()
        self.cp_status_btn.setEnabled(False)
        self.cp_status_btn.setText("Not corrected")
        grid_layout.addWidget(self.cp_status_btn, 1, 3)



        # Human verification row (row 2)
        hv_label = QLabel("Human verification")
        grid_layout.addWidget(hv_label, 2, 0)

        self.human_verify_trial_btn = QPushButton("Single Trial")
        self.human_verify_trial_btn.clicked.connect(lambda: self._human_verification_true("single_trial"))
        grid_layout.addWidget(self.human_verify_trial_btn, 2, 1)


        self.human_verify_all_trials_btn = QPushButton("All Trials")
        self.human_verify_all_trials_btn.clicked.connect(lambda: self._human_verification_true("all_trials"))
        grid_layout.addWidget(self.human_verify_all_trials_btn, 2, 2)


        self.human_verified_status = QPushButton()
        self.human_verified_status.setEnabled(False)
        self.human_verified_status.setText("Not verified")
        grid_layout.addWidget(self.human_verified_status, 2, 3)


        layout.addWidget(control_grid)

        # Save buttons row
        save_row = QWidget()
        save_layout = QHBoxLayout()
        save_layout.setSpacing(5)
        save_row.setLayout(save_layout)

        self.save_labels_button = QPushButton("Save labels file")
        self.save_labels_button.setToolTip("Shortcut: (Ctrl + S). Save file in labels\\... folder")
        self.save_labels_button.clicked.connect(lambda: self.app_state.save_labels())
        save_layout.addWidget(self.save_labels_button)

        self.save_button = QPushButton("Merge labels and save data.nc")
        self.save_button.setToolTip("Takes current labels and saves to original data.nc file")
        self.save_button.clicked.connect(lambda: self.app_state.save_file())
        save_layout.addWidget(self.save_button)

        save_layout.addStretch()
        layout.addWidget(save_row)

       
    def _human_verification_true(self, mode=None):
        """Mark current trial as human verified."""
        if self.app_state.label_dt is None or self.app_state.trials_sel is None:
            return
        if mode == "single_trial":
            self.app_state.label_dt.sel(trials=self.app_state.trials_sel).attrs['human_verified'] = np.int8(1)
        elif mode == "all_trials":
            for trial in self.app_state.label_dt.trials:
                self.app_state.label_dt.sel(trials=trial).attrs['human_verified'] = np.int8(1)
        

        self._update_human_verified_status()

        
    def _update_human_verified_status(self):
        if self.app_state.label_dt is None or self.app_state.trials_sel is None:
            self.human_verified_status.setText("Not verified")
            self.human_verified_status.setStyleSheet("background-color: red; color: white;")
            return

        attrs = self.app_state.label_dt.sel(trials=self.app_state.trials_sel).attrs
        if attrs.get('human_verified', None) == True: # == np.int8(1):
            self.human_verified_status.setText("Human verified")
            self.human_verified_status.setStyleSheet("background-color: green; color: white;")
        else:
            self.human_verified_status.setText("Not verified")
            self.human_verified_status.setStyleSheet("background-color: red; color: white;")

        self.meta_widget.update_labels_widget_title()            
        self.data_widget.update_trials_combo()  
        
    def _update_cp_status(self):
        if self.app_state.label_dt is None or self.app_state.trials_sel is None:
            self.cp_status_btn.setText("Not corrected")
            self.cp_status_btn.setStyleSheet("background-color: red; color: white;")
            return

        attrs = self.app_state.label_dt.attrs
        if attrs.get('changepoint_corrected', 0):
            self.cp_status_btn.setText("CP corrected (global)")
            self.cp_status_btn.setStyleSheet("background-color: green; color: white;")
        else:
            self.cp_status_btn.setText("Not corrected")
            self.cp_status_btn.setStyleSheet("background-color: red; color: white;")

    


    def _import_predictions_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select prediction .nc file", "", "NetCDF files (*.nc);;All Files (*)")
        if file_path:
            if 'predictions' not in os.path.basename(file_path):
                show_warning("Filename must include 'predictions' .")
                return
            self.app_state.pred_dt = TrialTree.load(file_path)
            self.app_state.pred_ds = self.app_state.pred_dt.sel(trials=self.app_state.trials_sel)
            self.pred_show_predictions.setEnabled(True)
            self.pred_show_predictions.setChecked(True)
            self.pred_file_path_edit.setText(file_path)

        xmin, xmax = self.plot_container.get_current_xlim()
        self.data_widget.update_main_plot(t0=xmin, t1=xmax)
        self.refresh_motif_shapes_layer()
    
    
    
    def _on_pred_show_predictions_changed(self):
        xmin, xmax = self.plot_container.get_current_xlim()
        self.data_widget.update_main_plot(t0=xmin, t1=xmax)
            
            
    def _cp_correction(self, mode):
        
        # HARD CODED for now
        params_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs', "Freddy_train_20251021_164220.json")

        with open(params_path, "r") as f:
            all_params = json.load(f)
            
        ds_kwargs = self.app_state.get_ds_kwargs()
 
 
 
        if mode == "single_trial":
            curr_labels, filt_kwargs = sel_valid(self.app_state.label_ds.labels, ds_kwargs)
            self.app_state.label_dt.sel(trials=self.app_state.trials_sel).labels.loc[filt_kwargs] = correct_changepoints_one_trial(curr_labels, self.app_state.ds, all_params, speed_correction=False)

    
        if mode == "all_trials":
            if self.app_state.label_dt.attrs.get("changepoint_corrected", 0) == 1:
                show_warning("Changepoint correction has already been applied to all trials. Don't re-apply.")
                return
                

            for trial in self.app_state.label_dt.trials:
                curr_labels, filt_kwargs = sel_valid(self.app_state.label_dt.sel(trials=trial).labels, ds_kwargs)
                ds = self.app_state.dt.sel(trials=trial)
                self.app_state.label_dt.sel(trials=trial).labels.loc[filt_kwargs] = correct_changepoints_one_trial(curr_labels, ds, all_params, speed_correction=False)
                self.app_state.label_dt.attrs["changepoint_corrected"] = np.int8(1)    
            self._update_cp_status()
        
        xmin, xmax = self.plot_container.get_current_xlim()
        self.data_widget.update_main_plot(t0=xmin, t1=xmax)
        
        
        




    MOTIF_ID_TO_KEY = {}

    # Row 1: 1-0 (Motifs 1-10)
    number_keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    for i, key in enumerate(number_keys):
        motif_id = i + 1 if key != '0' else 10
        MOTIF_ID_TO_KEY[motif_id] = key

    # Row 2: Q-P (Motifs 11-20)
    qwerty_row = ['q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p']
    for i, key in enumerate(qwerty_row):
        motif_id = i + 11
        MOTIF_ID_TO_KEY[motif_id] = key.upper()  # Display as uppercase for clarity

    # Row 3: A-; (Motifs 21-30)
    home_row = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';']
    for i, key in enumerate(home_row):
        motif_id = i + 21
        MOTIF_ID_TO_KEY[motif_id] = key.upper() if key != ';' else ';'  # Keep ; as is

    # Also provide reverse mapping for key to motif_id
    KEY_TO_MOTIF_ID = {v.lower(): k for k, v in MOTIF_ID_TO_KEY.items()}
    
    def _populate_motifs_table(self):    
        """Populate the motifs table with loaded mappings."""
        self.motifs_table.setRowCount(len(self.motif_mappings))
        for row, (motif_id, data) in enumerate(self.motif_mappings.items()):
            # ID column
            id_item = QTableWidgetItem(str(motif_id))
            id_item.setData(Qt.UserRole, motif_id)
            self.motifs_table.setItem(row, 0, id_item)

            # Name column with keyboard shortcut
            shortcut = self.MOTIF_ID_TO_KEY.get(motif_id, "?")
            name_with_shortcut = f"{data['name']} (Press {shortcut})"
            name_item = QTableWidgetItem(name_with_shortcut)
            self.motifs_table.setItem(row, 1, name_item)

            # Color column
            color_item = QTableWidgetItem()
            color = data["color"]
            qcolor = QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            color_item.setBackground(qcolor)
            self.motifs_table.setItem(row, 2, color_item)

    def _on_table_selection_changed(self):
        """Handle table row selection changes by activating the selected motif."""
        selected_rows = self.motifs_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            id_item = self.motifs_table.item(row, 0)
            if id_item:
                motif_id = id_item.data(Qt.UserRole)
                self.activate_motif(motif_id)



    def activate_motif(self, motif_key):
        """Activate a motif by shortcu1t: select row, set up for labeling, and scroll to row."""

        # Convert key to motif ID using centralized1 mapping
        motif_id = self.KEY_TO_MOTIF_ID.get(str(motif_key).lower(), motif_key)
        # Check if motif ID is valid
        if motif_id not in self.motif_mappings:
            print(f"No motif defined for key {motif_key}")
            return
        # Set selected motif and start labeling
        self.selected_motif_id = motif_id

        #  Find and select the corresponding row in the table
        for row in range(self.motifs_table.rowCount()):
            item = self.motifs_table.item(row, 0)  # ID column
            if item and item.data(Qt.UserRole) == motif_id:
                self.motifs_table.selectRow(row)
                self.motifs_table.scrollToItem(item)
                break
        self.ready_for_label_click = True
        self.first_click = None
        self.second_click = None
        
            
            

    def _on_plot_clicked(self, click_info):
        """Handle mouse clicks on the lineplot widget.

        Args:
            click_info: dict with 'x' (time coordinate) and 'button' (Qt button constant)
        """

        x_clicked = click_info["x"]
        button = click_info["button"]

        if x_clicked is None or not self.app_state.ready:
            return


        ds_kwargs = self.app_state.get_ds_kwargs()

        try:
            if (self.pred_show_predictions.isChecked() and
                self.app_state.pred_ds is not None):

        
                main_data, _ = sel_valid(self.app_state.label_ds.labels, ds_kwargs)
                secondary_data, _ = sel_valid(self.app_state.pred_ds.labels, ds_kwargs)
                is_prediction_main = False
   
            else:
                main_data, _ = sel_valid(self.app_state.label_ds.labels, ds_kwargs)
                secondary_data = None
                is_prediction_main = False
 

            if button == Qt.LeftButton and not self.ready_for_label_click:
                result = self._check_motif_click(x_clicked, main_data, secondary_data, is_prediction_main)
    

        except Exception as e:
            print(f"Error in plot click handling: {e}")
            return
            

        # Handle right-click - play video of motif if clicking on one
        if button == Qt.RightButton and self.app_state.video_folder is not None:
            frame = int(x_clicked * self.app_state.ds.fps)
            self.data_widget.video.seek_to_frame(frame)

        
        # Handle left-click for labeling/editing (only in label mode)
        elif button == Qt.LeftButton and self.ready_for_label_click:
    
            # Snap to nearest changepoint if available
            x_clicked_idx = int(x_clicked * self.app_state.ds.fps)  # Convert to frame index
            
            if self.changepoint_correction_checkbox.isChecked():
                x_snapped = self._snap_to_changepoint(x_clicked_idx)
            else:
                x_snapped = x_clicked_idx

            if self.first_click is None:
                # First click - just store the position
                self.first_click = x_snapped
            else:
                # Second click - store position and automatically apply
                self.second_click = x_snapped
                self._apply_motif()  # Automatically apply after two clicks



    def _check_motif_click(self, x_clicked: float, main_data: np.ndarray,
                          secondary_data: np.ndarray = None, is_prediction_main: bool = False) -> bool:
        """Check if the click is on an existing motif and select it if so.

        Args:
            x_clicked: X coordinate of the click
            main_data: Primary data array to check for motifs
            secondary_data: Secondary data array (optional)
            is_prediction_main: Whether the main data is predictions or labels
        """
        # Check if there's a motif at this position
        frame_idx = int(x_clicked * self.app_state.ds.fps)

        if frame_idx >= len(main_data):
            print(f"Frame index {frame_idx} out of bounds for data length {len(main_data)}")
            return False

        motif_id = int(main_data[frame_idx])

        if motif_id != 0:
            motif_start = frame_idx
            motif_end = frame_idx

            # Find start of motif
            while motif_start > 0 and main_data[motif_start - 1] == motif_id:
                motif_start -= 1

            # Find end of motif
            while motif_end < len(main_data) - 1 and main_data[motif_end + 1] == motif_id:
                motif_end += 1

            self.current_motif_id = motif_id
            self.current_motif_pos = [motif_start, motif_end]
            self.current_motif_is_prediction = is_prediction_main
            self.highlight_spaceplot.emit(motif_start, motif_end)

            self.selected_motif_id = motif_id

            return True
        else:
            return False

    def _snap_to_changepoint(self, x_clicked_idx: float) -> float:
        """Snap the clicked x-coordinate to the nearest changepoint."""

        ds_kwargs = self.app_state.get_ds_kwargs()

        cp_ds = self.app_state.ds.sel(**ds_kwargs).filter_by_attrs(type="changepoints")
        if len(cp_ds.data_vars) == 0:
            return x_clicked_idx

        feature_sel = self.app_state.features_sel
        ds_kwargs = self.app_state.get_ds_kwargs()
        snapped_val = snap_to_nearest_changepoint(x_clicked_idx, self.app_state.ds, feature_sel, **ds_kwargs)
        return snapped_val

    def _apply_motif(self):
        """Apply the selected motif to the selected time range."""
        if self.first_click is None or self.second_click is None:
            return
        


        ds_kwargs = self.app_state.get_ds_kwargs()
        labels, filt_kwargs = sel_valid(self.app_state.label_ds.labels, ds_kwargs)

        start_idx = self.first_click
        end_idx = self.second_click
        self.highlight_spaceplot.emit(start_idx, end_idx)




        if hasattr(self, 'old_motif_pos') and self.old_motif_pos is not None:
            old_start, old_end = self.old_motif_pos
            labels[old_start : old_end + 1] = 0

            
            # Clean up edit mode variables
            self.old_motif_pos = None
            self.old_motif_id = None
            self.current_motif_pos = None
            self.current_motif_id = None
            self.current_motif_is_prediction = False

        if labels[end_idx] != 0 and not labels[end_idx+1] == 0:
            end_idx = end_idx - 1
            

        labels[start_idx : end_idx + 1] = self.selected_motif_id
        
        






        self.first_click = None
        self.second_click = None
        self.ready_for_label_click = False
        
        labels = remove_small_blocks(labels, min_motif_len=3)
    
        self.app_state.label_dt.sel(trials=self.app_state.trials_sel).labels.loc[filt_kwargs] = labels
        self._human_verification_true(mode="single_trial")

        self._mark_changes_unsaved()

        self.data_widget.update_motif_plot(ds_kwargs)
        self.refresh_motif_shapes_layer()
        
            
            




    def _delete_motif(self):
        if self.current_motif_pos is None:
            return
        
    

        start, end = self.current_motif_pos


        ds_kwargs = self.app_state.get_ds_kwargs()


        labels, filt_kwargs = sel_valid(self.app_state.label_ds.labels, ds_kwargs)



        labels[start : end + 1] = 0


        self.current_motif_pos = None
        self.current_motif_id = None
        self.current_motif_is_prediction = False

    
        self.app_state.label_dt.sel(trials=self.app_state.trials_sel).labels.loc[filt_kwargs] = labels
        

        self._mark_changes_unsaved()
        self.data_widget.update_motif_plot(ds_kwargs)
        self.refresh_motif_shapes_layer()

    def _edit_motif(self):
        """Enter edit mode for adjusting motif boundaries."""
        if self.current_motif_pos is None:
            print("No motif selected. Right-click on a motif first to select it.")
            return
        



        # Store the old motif info for later cleanup
        self.old_motif_pos = self.current_motif_pos.copy()
        self.old_motif_id = self.current_motif_id
        
        # Enter editing mode - user needs to click twice to set new boundaries
        self.ready_for_label_click = True
        self.first_click = None
        self.second_click = None
        

        return

    def _play_segment(self):
        if self.current_motif_pos is None:
            print("No motif selected for playback")
            return
        
        if self.app_state.sync_state not in ["napari_video_mode", "napari_video_mode_paused"]:
            self.app_state.sync_state = "napari_video_mode" # fallback

        
        if (
            self.app_state.sync_state != "napari_video_mode"
            or not self.current_motif_id
            or len(self.current_motif_pos) != 2
        ):
            print(f"Playback conditions not met - sync_state: {self.app_state.sync_state}, motif_id: {self.current_motif_id}, pos_len: {len(self.current_motif_pos) if self.current_motif_pos else 0}")
            return



        start_frame = self.current_motif_pos[0]
        end_frame = self.current_motif_pos[1]



        if hasattr(self.data_widget, 'video') and self.data_widget.video:
            self.data_widget.video.play_segment(start_frame, end_frame)
        else:
            print("No video available for playback.")





        




    def _add_motif_shapes_layer(self):
        """Add single box overlay with dynamically updating text."""
 
        
        ds_kwargs = self.app_state.get_ds_kwargs()
        

        label_ds = self.app_state.label_ds
        labels, _ = sel_valid(label_ds.labels, ds_kwargs)

   


        try:
            layer = self.viewer.layers[0]
            
            if layer.data.ndim == 2:
                height, width = layer.data.shape
            
            elif layer.data.ndim == 3:
                height, width = layer.data.shape[1:3]
                
            else:
                height, width = 100, 100
            
        except (IndexError, AttributeError):
            print("No video layer found for motif shapes overlay.")
            return None
        
        
        
        
        box_width, box_height = 180, 50
        x = width - box_width - 5
        y = height - box_height - 5
        

        rect = np.array([[[y, x],
                        [y, x + box_width],
                        [y + box_height, x + box_width],
                        [y + box_height, x]]])
        

        labels_array = np.asarray(labels)
        frame_to_text = {}
        frame_to_color = {}
        
        for idx, label in enumerate(labels_array):
            if label in self.motif_mappings:
                if label == 0:
                    frame_to_text[idx] = ""
                else:
                    frame_to_text[idx] = self.motif_mappings[label]["name"]
                
                color = self.motif_mappings[label]["color"]
                frame_to_color[idx] = color.tolist() if hasattr(color, 'tolist') else list(color)
        


        shapes_layer = self.viewer.add_shapes(
            rect,
            shape_type='rectangle',
            name="motif_labels",
            face_color='white',
            edge_color='black',
            edge_width=2,
            opacity=0.9,
            text={'string': [''], 'color': [[0, 0, 0]], 'size': 20, 'anchor': 'center'}
        )
   
        
        shapes_layer.z_index = 1000
        

        shapes_layer.metadata = {
            'frame_to_text': frame_to_text,
            'frame_to_color': frame_to_color
        }
        
    
        def update_motif_text(event=None):
            current_frame = self.viewer.dims.current_step[0]
            if current_frame in frame_to_text:
                shapes_layer.text = {
                    'string': [frame_to_text[current_frame]],
                    'color': [frame_to_color[current_frame]],
                    'size': 18,
                    'anchor': 'center'
                }
            else:
                shapes_layer.text = {'string': [''], 'color': [[0, 0, 0]]}
        
    
        self.viewer.dims.events.current_step.connect(update_motif_text)
        
    
        update_motif_text()
        
  
        return shapes_layer
    
    def _remove_motif_shapes_layer(self):
        """Remove existing motif shapes layer if it exists."""
        if "motif_labels" in self.viewer.layers:
            self.viewer.layers.remove("motif_labels")


    def refresh_motif_shapes_layer(self):
        """Refresh the entire motif shapes layer - call when labels change."""
        self._remove_motif_shapes_layer()
        self._add_motif_shapes_layer()