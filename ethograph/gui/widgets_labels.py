"""Widget for labeling segments in movement data."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from napari.utils.notifications import show_info, show_warning
from napari.viewer import Viewer
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ethograph import TrialTree
from ethograph.features.changepoints import snap_to_nearest_changepoint
from ethograph.utils.data_utils import sel_valid
from ethograph.utils.labels import load_label_mapping, purge_small_blocks
from ethograph.utils.paths import get_project_root

from .app_constants import (
    LABELS_TABLE_MAX_HEIGHT,
    LABELS_TABLE_ROW_HEIGHT,
    LABELS_TABLE_ID_COLUMN_WIDTH,
    LABELS_TABLE_COLOR_COLUMN_WIDTH,
    LABELS_OVERLAY_BOX_WIDTH,
    LABELS_OVERLAY_BOX_HEIGHT,
    LABELS_OVERLAY_BOX_MARGIN,
    LABELS_OVERLAY_TEXT_SIZE,
    LABELS_OVERLAY_FALLBACK_SIZE,
    DEFAULT_LAYOUT_SPACING,
    Z_INDEX_LABELS_OVERLAY,
    MIN_LABEL_LEN
)


class LabelsWidget(QWidget):
    """Widget for labeling movement labels in time series data."""
    
    highlight_spaceplot = Signal(float, float)

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.app_state = app_state



        self.plot_container = None  # Will be set after creation
        self.meta_widget = None  # Will be set after creation
        self.changepoints_widget = None  # Will be set after creation

        # Make widget focusable for keyboard events
        self.setFocusPolicy(Qt.StrongFocus)

        # Remove Qt event filter and key event logic
        # Instead, rely on napari's @viewer.bind_key for global shortcuts
        # Shortcut bindings are now handled outside the widget

        # Labeling state
        self._mappings: dict[int, dict[str, Any]] = {}
        self.ready_for_label_click = False
        self.ready_for_play_click = False
        self.first_click = None
        self.second_click = None
        self.selected_labels_id = 0

        # Current  selection for editing
        self.current_labels_pos: list[int] | None = None  # [start, end] idx of selected 
        self.current_labels_id: int | None = None  # ID of currently selected 
        self.current_labels_is_prediction: bool = False  # Whether selected  is from predictions

        # Edit mode state  
        self.old_labels_pos: list[int] | None = None  # Original position when editing
        self.old_labels_id: int | None = None  # Original ID when editing
        
        # Frame tracking for  display
        self.previous_frame: int | None = None


        # UI components
        self.labels_table = None

        self._setup_ui()



        mapping_path = get_project_root() / "configs" / "mapping.txt"
        self._mappings = load_label_mapping(mapping_path)
        self._populate_labels_table()



    def _mark_changes_unsaved(self):
        """Mark that changes have been made and are not saved."""
        self.app_state.changes_saved = False

    def set_plot_container(self, plot_container):
        """Set the plot container reference and connect click handler to all plots."""
        self.plot_container = plot_container
        plot_container.set_label_mappings(self._mappings)

        for plot in [plot_container.line_plot,
                     plot_container.spectrogram_plot,
                     plot_container.audio_trace_plot]:
            if plot is not None:
                plot.plot_clicked.connect(self._on_plot_clicked)

    def set_meta_widget(self, meta_widget):
        """Set reference to the meta widget for layout refresh."""
        self.meta_widget = meta_widget

    def plot_all_labels(self, time_data, labels, predictions=None):
        """Plot all labels for current trial and keypoint based on current labels state.

        Delegates to PlotContainer for centralized, synchronized label drawing
        across all plot types.

        Args:
            time_data: Time array for x-axis
            labels: Primary label data to plot as main rectangles
            predictions: Optional prediction data to plot as small rectangles at top
        """
        if labels is None or self.plot_container is None:
            return

        show_predictions = (
            predictions is not None and
            self.pred_show_predictions.isChecked() and
            hasattr(self.app_state, 'pred_ds') and
            self.app_state.pred_ds is not None
        )

        self.plot_container.draw_all_labels(
            time_data, labels,
            predictions=predictions,
            show_predictions=show_predictions
        )

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        self.setLayout(layout)

        # Create toggle buttons for collapsible sections
        self._create_toggle_buttons()
        layout.addWidget(self.toggle_widget)

        # Create labels table
        self._create_labels_table_and_edit_buttons()

        # Create control buttons
        self._create_control_buttons()

        # Add collapsible sections
        layout.addWidget(self.labels_table)
        layout.addWidget(self.controls_widget)

        # Set initial state: table visible, controls hidden
        self.table_toggle.setText("📋 Table ✓")
        self.controls_toggle.setText("🎛️ Controls")
        self.controls_widget.hide()

        layout.addStretch()

    def _create_toggle_buttons(self):
        """Create toggle buttons for collapsible sections."""
        self.toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        self.toggle_widget.setLayout(toggle_layout)

        # Table toggle button
        self.table_toggle = QPushButton("📋 Table")
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
        """Toggle labels table visibility (mutually exclusive with controls)."""
        if self.table_toggle.isChecked():
            # Show table, hide controls
            self.labels_table.show()
            self.controls_widget.hide()
            self.table_toggle.setText("📋 Table ✓")
            self.controls_toggle.setText("🎛️ Controls")
            self.controls_toggle.setChecked(False)
        else:
            # If trying to uncheck table, force controls to be checked instead
            self.controls_widget.show()
            self.labels_table.hide()
            self.controls_toggle.setText("🎛️ Controls ✓")
            self.table_toggle.setText("📋 Table")
            self.controls_toggle.setChecked(True)
        self._refresh_layout()

    def _toggle_controls(self):
        """Toggle controls visibility (mutually exclusive with table)."""
        if self.controls_toggle.isChecked():
            # Show controls, hide table
            self.controls_widget.show()
            self.labels_table.hide()
            self.controls_toggle.setText("🎛️ Controls ✓")
            self.table_toggle.setText("📋 Table")
            self.table_toggle.setChecked(False)
        else:
            # If trying to uncheck controls, force table to be checked instead
            self.labels_table.show()
            self.controls_widget.hide()
            self.table_toggle.setText("📋 Table ✓")
            self.controls_toggle.setText("🎛️ Controls")
            self.table_toggle.setChecked(True)
        self._refresh_layout()

    def _refresh_layout(self):
        """Force layout recalculation by toggling the collapsible widget."""
        if self.meta_widget and hasattr(self.meta_widget, 'collapsible_widgets'):
            # Labels widget is at index 3 (0: Documentation, 1: I/O, 2: Data controls, 3: Label controls)
            labels_collapsible = self.meta_widget.collapsible_widgets[3]
            labels_collapsible.collapse()
            QApplication.processEvents()
            labels_collapsible.expand()

    def _create_labels_table_and_edit_buttons(self):
        """Create the labels table showing available  types in two columns."""
        self.labels_table = QTableWidget()
        self.labels_table.setColumnCount(6)
        self.labels_table.setHorizontalHeaderLabels(["ID", "Name (Shortcut)", "C", "ID", "Name (Shortcut)", "C"])

        self.labels_table.verticalHeader().setVisible(False)

        header = self.labels_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Fixed)

        self.labels_table.setColumnWidth(0, LABELS_TABLE_ID_COLUMN_WIDTH)
        self.labels_table.setColumnWidth(2, LABELS_TABLE_COLOR_COLUMN_WIDTH)
        self.labels_table.setColumnWidth(3, LABELS_TABLE_ID_COLUMN_WIDTH)
        self.labels_table.setColumnWidth(5, LABELS_TABLE_COLOR_COLUMN_WIDTH)

        self.labels_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.labels_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.labels_table.verticalHeader().setDefaultSectionSize(LABELS_TABLE_ROW_HEIGHT)
        self.labels_table.setMaximumHeight(LABELS_TABLE_MAX_HEIGHT)
        self.labels_table.setStyleSheet("""
            QTableWidget { gridline-color: transparent; }
            QTableWidget::item { padding: 0px 2px; }
            QHeaderView::section { padding: 0px 2px; }
        """)

        self.labels_table.itemSelectionChanged.connect(self._on_table_selection_changed)



    def _create_control_buttons(self):
        """Create control buttons for labeling operations."""
        
        self.controls_widget = QWidget()
        layout = QVBoxLayout()
        self.controls_widget.setLayout(layout)


        # Mapping file row
        mapping_row = QWidget()
        mapping_layout = QHBoxLayout()
        mapping_row.setLayout(mapping_layout)

        self.mapping_file_path_edit = QLineEdit()
        default_mapping_path = str(get_project_root() / "configs" / "mapping.txt")
        self.mapping_file_path_edit.setText(default_mapping_path)
        self.mapping_file_path_edit.setToolTip("Path to mapping.txt file")
        self.mapping_file_path_edit.returnPressed.connect(
            lambda: self._reload_mapping(self.mapping_file_path_edit.text())
        )
        mapping_layout.addWidget(self.mapping_file_path_edit)

        self.browse_mapping_btn = QPushButton("Browse")
        self.browse_mapping_btn.clicked.connect(self._browse_mapping_file)
        mapping_layout.addWidget(self.browse_mapping_btn)

        self.temp_labels_button = QPushButton("Create temporary labels")
        self.temp_labels_button.setToolTip("Create custom labels for this session only")
        self.temp_labels_button.clicked.connect(self._create_temporary_labels)
        mapping_layout.addWidget(self.temp_labels_button)

        layout.addWidget(mapping_row)

        # Predictions file row
        pred_row = QWidget()
        pred_layout = QHBoxLayout()
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


        # Human verification row
        hv_row = QHBoxLayout()
        hv_row.addWidget(QLabel("Apply human verification to:"))

        self.human_verify_trial_btn = QPushButton("Single Trial")
        self.human_verify_trial_btn.clicked.connect(lambda: self._human_verification_true("single_trial"))
        hv_row.addWidget(self.human_verify_trial_btn)

        self.human_verify_all_trials_btn = QPushButton("All Trials")
        self.human_verify_all_trials_btn.clicked.connect(lambda: self._human_verification_true("all_trials"))
        hv_row.addWidget(self.human_verify_all_trials_btn)

        hv_row.addStretch()
        layout.addLayout(hv_row)



        bottom_row = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_row.setLayout(bottom_layout)

        self.save_labels_button = QPushButton("Save labels file")
        self.save_labels_button.setToolTip("Shortcut: (Ctrl + S). Save file in labels\\... folder")
        self.save_labels_button.clicked.connect(lambda: self.app_state.save_labels())
        bottom_layout.addWidget(self.save_labels_button)

        self.save_button = QPushButton("Merge labels and save sesssion")
        self.save_button.setToolTip("Takes current labels and saves to original sesssion file")
        self.save_button.clicked.connect(lambda: self.app_state.save_file())
        bottom_layout.addWidget(self.save_button)

        bottom_layout.addStretch()
        layout.addWidget(bottom_row)


    def _browse_mapping_file(self):
        """Browse for a mapping.txt file and reload  mappings."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select mapping.txt file",
            str(get_project_root() / "configs"),
            "Text files (*.txt);;All Files (*)"
        )
        if file_path:
            self.mapping_file_path_edit.setText(file_path)
            self._reload_mapping(file_path)

    def _reload_mapping(self, mapping_path: str):
        """Reload  mappings from the specified path."""
        try:
            self._mappings = load_label_mapping(Path(mapping_path))
            if self.plot_container:
                self.plot_container.set_label_mappings(self._mappings)
            if self.changepoints_widget:
                self.changepoints_widget.set_motif_mappings(self._mappings)
            self._populate_labels_table()
            self.refresh_labels_shapes_layer()
            show_info(f"Loaded {len(self._mappings) - 1} labels from {Path(mapping_path).name}")
        except FileNotFoundError:
            show_warning(f"Mapping file not found: {mapping_path}")

    def _create_temporary_labels(self):
        """Open dialog to create temporary labels for this session."""
        dialog = TemporaryLabelsDialog(self)
        if dialog.exec_():
            labels = dialog.get_labels()
            if labels:
                mapping_path = get_project_root() / "configs" / "mapping_temporary.txt"
                with open(mapping_path, "w") as f:
                    f.write("0 background\n")
                    for i, label in enumerate(labels, start=1):
                        f.write(f"{i} {label}\n")

                self.mapping_file_path_edit.setText(str(mapping_path))
                self._mappings = load_label_mapping(mapping_path)
                if self.plot_container:
                    self.plot_container.set_label_mappings(self._mappings)
                if self.changepoints_widget:
                    self.changepoints_widget.set_motif_mappings(self._mappings)
                self._populate_labels_table()
                self.refresh_labels_shapes_layer()
                show_info(f"Loaded {len(labels)} temporary labels")

    def _human_verification_true(self, mode=None):
        """Mark current trial as human verified."""
        if self.app_state.label_dt is None or self.app_state.trials_sel is None:
            return
        if mode == "single_trial":
            self.app_state.label_dt.trial(self.app_state.trials_sel).attrs['human_verified'] = np.int8(1)
        elif mode == "all_trials":
            for trial in self.app_state.label_dt.trials:
                self.app_state.label_dt.trial(trial).attrs['human_verified'] = np.int8(1)

        self._update_human_verified_status()
        self.app_state.verification_changed.emit()

        
    def _update_human_verified_status(self):
        default_style = ""
        verified_style = "background-color: green; color: white;"

        if self.app_state.label_dt is None or self.app_state.trials_sel is None:
            self.human_verify_trial_btn.setStyleSheet(default_style)
            self.human_verify_all_trials_btn.setStyleSheet(default_style)
            return

        attrs = self.app_state.label_dt.trial(self.app_state.trials_sel).attrs
        if attrs.get('human_verified', None) == True:
            self.human_verify_trial_btn.setStyleSheet(verified_style)
        else:
            self.human_verify_trial_btn.setStyleSheet(default_style)

        all_verified = all(
            self.app_state.label_dt.trial(t).attrs.get('human_verified', None) == True
            for t in self.app_state.label_dt.trials
        )
        if all_verified:
            self.human_verify_all_trials_btn.setStyleSheet(verified_style)
        else:
            self.human_verify_all_trials_btn.setStyleSheet(default_style)  
        


    def _import_predictions_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select prediction .nc file", "", "NetCDF files (*.nc);;All Files (*)")
        if file_path:
            if 'predictions' not in os.path.basename(file_path):
                show_warning("Filename must include 'predictions' .")
                return
            self.app_state.pred_dt = TrialTree.open(file_path)
            self.app_state.pred_ds = self.app_state.pred_dt.trial(self.app_state.trials_sel)
            self.pred_show_predictions.setEnabled(True)
            self.pred_show_predictions.setChecked(True)
            self.pred_file_path_edit.setText(file_path)

        self.app_state.labels_modified.emit()
        self.refresh_labels_shapes_layer()

    def _on_pred_show_predictions_changed(self):
        self.app_state.labels_modified.emit()
            
            

    LABELS_ID_TO_KEY = {}

    # Row 1: 1-0 (Labels 1-10)
    number_keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    for i, key in enumerate(number_keys):
        _id = i + 1 if key != '0' else 10
        LABELS_ID_TO_KEY[_id] = key

    # Row 2: Q-P (Labels 11-20)
    qwerty_row = ['q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p']
    for i, key in enumerate(qwerty_row):
        _id = i + 11
        LABELS_ID_TO_KEY[_id] = key.upper()  # Display as uppercase for clarity

    # Row 3: A-; (Labels 21-30)
    home_row = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';']
    for i, key in enumerate(home_row):
        _id = i + 21
        LABELS_ID_TO_KEY[_id] = key.upper() if key != ';' else ';'  # Keep ; as is

    # Also provide reverse mapping for key to _id
    KEY_TO_LABELS_ID = {v.lower(): k for k, v in LABELS_ID_TO_KEY.items()}
    
    def _populate_labels_table(self):
        """Populate the labels table with loaded mappings in two columns."""
        items = [(k, v) for k, v in self._mappings.items() if k != 0]
        half = (len(items) + 1) // 2
        self.labels_table.setRowCount(half)

        for i, (_id, data) in enumerate(items):
            row = i % half
            col_offset = 0 if i < half else 3

            id_item = QTableWidgetItem(str(_id))
            id_item.setData(Qt.UserRole, _id)
            self.labels_table.setItem(row, col_offset, id_item)

            shortcut = self.LABELS_ID_TO_KEY.get(_id, "?")
            name_with_shortcut = f"{data['name']} ({shortcut})"
            name_item = QTableWidgetItem(name_with_shortcut)
            name_item.setData(Qt.UserRole, _id)
            self.labels_table.setItem(row, col_offset + 1, name_item)

            color_item = QTableWidgetItem()
            color = data["color"]
            qcolor = QColor(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            color_item.setBackground(qcolor)
            color_item.setData(Qt.UserRole, _id)
            self.labels_table.setItem(row, col_offset + 2, color_item)

    def _on_table_selection_changed(self):
        """Handle table cell selection changes by activating the selected ."""
        selected = self.labels_table.selectedItems()
        if selected:
            item = selected[0]
            _id = item.data(Qt.UserRole)
            if _id is not None:
                self.activate_label(_id)



    def activate_label(self, _key):
        """Activate a  by shortcut: select cell, set up for labeling, and scroll to it."""
        _id = self.KEY_TO_LABELS_ID.get(str(_key).lower(), _key)
        if _id not in self._mappings:
            return

        self.selected_labels_id = _id

        for row in range(self.labels_table.rowCount()):
            for col in [0, 3]:  # Check both ID columns
                item = self.labels_table.item(row, col)
                if item and item.data(Qt.UserRole) == _id:
                    self.labels_table.setCurrentItem(item)
                    self.labels_table.scrollToItem(item)
                    self.ready_for_label_click = True
                    self.first_click = None
                    self.second_click = None
                    return
        
            
            

    def _on_plot_clicked(self, click_info):
        """Handle mouse clicks on the lineplot widget.

        Args:
            click_info: dict with 'x' (time coordinate) and 'button' (Qt button constant)
        """

        t_clicked = click_info["x"]
        button = click_info["button"]

        if t_clicked is None or not self.app_state.ready:
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
                result = self._check_labels_click(t_clicked, main_data, secondary_data, is_prediction_main)
    

        except (KeyError, IndexError, ValueError, AttributeError) as e:
            print(f"Error in plot click handling: {e}")
            return
            

        # Handle right-click - seek video to clicked position
        if button == Qt.RightButton and self.app_state.video_folder is not None:
            frame = int(t_clicked * self.app_state.ds.fps)
            if hasattr(self.app_state, 'video') and self.app_state.video:
                self.app_state.video.seek_to_frame(frame)

        
        # Handle left-click for labeling/editing (only in label mode)
        elif button == Qt.LeftButton and self.ready_for_label_click:
    
            # Snap to nearest changepoint if available
            label_idx = int(t_clicked * self.app_state.label_sr)  
            
            if self.changepoints_widget and self.changepoints_widget.is_changepoint_correction_enabled():
                label_idx_snapped = self._snap_to_changepoint(label_idx)
            else:
                label_idx_snapped = label_idx

            if self.first_click is None:
                # First click - just store the position
                self.first_click = label_idx_snapped
            else:
                # Second click - store position and automatically apply
                self.second_click = label_idx_snapped
                self._apply_label()  # Automatically apply after two clicks



    def _check_labels_click(self, x_clicked: float, main_data: np.ndarray,
                          secondary_data: np.ndarray = None, is_prediction_main: bool = False) -> bool:
        """Check if the click is on an existing  and select it if so.

        Args:
            x_clicked: X coordinate of the click
            main_data: Primary data array to check for labels
            secondary_data: Secondary data array (optional)
            is_prediction_main: Whether the main data is predictions or labels
        """
        # Check if there's a  at this position
        label_idx = int(x_clicked * self.app_state.label_sr)

        if label_idx >= len(main_data):
            print(f"Frame index {label_idx} out of bounds for data length {len(main_data)}")
            return False

        _id = int(main_data[label_idx])

        if _id != 0:
            _start = label_idx
            _end = label_idx
            
            

            # Find start of 
            while _start > 0 and main_data[_start - 1] == _id:
                _start -= 1

            # Find end of 
            while _end < len(main_data) - 1 and main_data[_end + 1] == _id:
                _end += 1

            self.current_labels_id = _id
            self.current_labels_pos = [_start, _end]
            self.current_labels_is_prediction = is_prediction_main
            
            _start_t = _start / self.app_state.label_sr
            _end_t = _end / self.app_state.label_sr
            self.highlight_spaceplot.emit(_start_t, _end_t)

            self.selected_labels_id = _id

            return True
        else:
            return False

    def _snap_to_changepoint(self, x_clicked_idx: float) -> float:
        """Snap the clicked x-coordinate to the nearest changepoint.

        Considers both dataset changepoints and audio changepoints (if visible).
        """
        best_snapped = x_clicked_idx
        best_distance = float('inf')

        ds_kwargs = self.app_state.get_ds_kwargs()

        cp_ds = self.app_state.ds.sel(**ds_kwargs).filter_by_attrs(type="changepoints")
        if len(cp_ds.data_vars) > 0:
            feature_sel = self.app_state.features_sel
            ds_kwargs = self.app_state.get_ds_kwargs()
            ds_snapped = snap_to_nearest_changepoint(x_clicked_idx, self.app_state.ds, feature_sel, **ds_kwargs)
            ds_distance = abs(ds_snapped - x_clicked_idx)
            if ds_distance < best_distance:
                best_snapped = ds_snapped
                best_distance = ds_distance

        if getattr(self.app_state, 'show_changepoints', False):
            onsets = getattr(self.app_state, 'audio_changepoint_onsets', None)
            offsets = getattr(self.app_state, 'audio_changepoint_offsets', None)
            if onsets is not None and offsets is not None:
                label_sr = self.app_state.label_sr
                if label_sr and label_sr > 0:
                    all_audio_cp_times = np.concatenate([onsets, offsets])
                    audio_cp_indices = (all_audio_cp_times * label_sr).astype(int)
                    if len(audio_cp_indices) > 0:
                        nearest_idx = np.argmin(np.abs(audio_cp_indices - x_clicked_idx))
                        audio_snapped = audio_cp_indices[nearest_idx]
                        audio_distance = abs(audio_snapped - x_clicked_idx)
                        if audio_distance < best_distance:
                            best_snapped = audio_snapped

        return best_snapped

    def _apply_label(self):
        """Apply the selected  to the selected time range."""
        
        if self.first_click is None or self.second_click is None:
            return
        


        ds_kwargs = self.app_state.get_ds_kwargs()
        labels, filt_kwargs = sel_valid(self.app_state.label_ds.labels, ds_kwargs)


        start_idx = self.first_click
        end_idx = self.second_click
        print(f"Applying  {self.selected_labels_id} from {start_idx} to {end_idx}")
        print(f"Labels length: {len(labels)}")
    
        start_t = start_idx / self.app_state.label_sr
        end_t = end_idx / self.app_state.label_sr
        self.highlight_spaceplot.emit(start_t, end_t)




        if hasattr(self, 'old_labels_pos') and self.old_labels_pos is not None:
            old_start, old_end = self.old_labels_pos
            labels[old_start : old_end + 1] = 0

            
            # Clean up edit mode variables
            self.old_labels_pos = None
            self.old_labels_id = None
            self.current_labels_pos = None
            self.current_labels_id = None
            self.current_labels_is_prediction = False

        if labels[end_idx] != 0 and not labels[end_idx+1] == 0:
            end_idx = end_idx - 1
            

        labels[start_idx : end_idx + 1] = self.selected_labels_id

        # Auto-select the newly created/edited  for immediate playback with 'v'
        self.current_labels_pos = [start_idx, end_idx]
        self.current_labels_id = self.selected_labels_id
        self.current_labels_is_prediction = False





        self.first_click = None
        self.second_click = None
        self.ready_for_label_click = False

   
        labels = purge_small_blocks(labels, MIN_LABEL_LEN)



        self.app_state.label_dt.trial(self.app_state.trials_sel).labels.loc[filt_kwargs] = labels

        self._human_verification_true(mode="single_trial")
        self._mark_changes_unsaved()
        self.app_state.labels_modified.emit()
        self._seek_to_frame(start_idx)
        self.refresh_labels_shapes_layer()

        

    def _seek_to_frame(self, label_idx: int):
        """Seek video and update time marker to the specified label index.

        Args:
            label_idx: Index into the label array (at label_sr rate), not a video frame.
        """
        # Convert label index to time
        current_time = label_idx / self.app_state.label_sr

        if hasattr(self.app_state, 'video') and self.app_state.video:
            # Convert time to video frame for seek
            video_frame = int(current_time * self.app_state.ds.fps)
            self.app_state.video.seek_to_frame(video_frame)
        elif self.plot_container:
            self.plot_container.current_plot.update_time_marker(current_time)


    def _delete_label(self):
        if self.current_labels_pos is None:
            return
        
    

        start, end = self.current_labels_pos


        ds_kwargs = self.app_state.get_ds_kwargs()


        labels, filt_kwargs = sel_valid(self.app_state.label_ds.labels, ds_kwargs)



        labels[start : end + 1] = 0


        self.current_labels_pos = None
        self.current_labels_id = None
        self.current_labels_is_prediction = False

    
        self.app_state.label_dt.trial(self.app_state.trials_sel).labels.loc[filt_kwargs] = labels

        self._mark_changes_unsaved()
        self.app_state.labels_modified.emit()
        self.refresh_labels_shapes_layer()

    def _edit_label(self):
        """Enter edit mode for adjusting  boundaries."""
        if self.current_labels_pos is None:
            print("No  selected. Right-click on a  first to select it.")
            return
        



        # Store the old  info for later cleanup
        self.old_labels_pos = self.current_labels_pos.copy()
        self.old_labels_id = self.current_labels_id
        
        # Enter editing mode - user needs to click twice to set new boundaries
        self.ready_for_label_click = True
        self.first_click = None
        self.second_click = None
        

        return

    def _play_segment(self):
        if self.current_labels_pos is None:
            print("No  selected for playback")
            return

        if not self.current_labels_id or len(self.current_labels_pos) != 2:
            print(f"Playback conditions not met - _id: {self.current_labels_id}, pos_len: {len(self.current_labels_pos) if self.current_labels_pos else 0}")
            return

        # Label idxs -> Time -> Frame idxs
        start_time = self.current_labels_pos[0] / self.app_state.label_sr
        end_time = self.current_labels_pos[1] / self.app_state.label_sr
        start_frame = int(start_time * self.app_state.ds.fps)
        end_frame = int(end_time * self.app_state.ds.fps)

        if hasattr(self.app_state, 'video') and self.app_state.video:
            self.app_state.video.play_segment(start_frame, end_frame)
        else:
            print("No video available for playback.")




    def _add_labels_shapes_layer(self):
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
                height, width = LABELS_OVERLAY_FALLBACK_SIZE

        except (IndexError, AttributeError):
            print("No video layer found for  shapes overlay.")
            return None

        box_width, box_height = LABELS_OVERLAY_BOX_WIDTH, LABELS_OVERLAY_BOX_HEIGHT
        x = width - box_width - LABELS_OVERLAY_BOX_MARGIN
        y = height - box_height - LABELS_OVERLAY_BOX_MARGIN
        

        rect = np.array([[[y, x],
                        [y, x + box_width],
                        [y + box_height, x + box_width],
                        [y + box_height, x]]])
        

        labels_array = np.asarray(labels)

        shapes_layer = self.viewer.add_shapes(
            rect,
            shape_type='rectangle',
            name="_labels",
            face_color='white',
            edge_color='black',
            edge_width=2,
            opacity=0.9,
            text={'string': [''], 'color': [[0, 0, 0]], 'size': LABELS_OVERLAY_TEXT_SIZE, 'anchor': 'center'}
        )

        shapes_layer.z_index = Z_INDEX_LABELS_OVERLAY

        # Store labels array and conversion factors for on-demand lookup
        video_fps = self.app_state.ds.fps if hasattr(self.app_state, 'ds') and self.app_state.ds else 30.0
        label_sr = getattr(self.app_state, 'label_sr')

        shapes_layer.metadata = {
            'labels_array': labels_array,
            'video_fps': video_fps,
            'label_sr': label_sr,
            '_mappings': self._mappings,
        }

        def update_labels_text(event=None):
            # Convert video frame to data index on-demand
            video_frame = self.viewer.dims.current_step[0]
            time_s = video_frame / shapes_layer.metadata['video_fps']
            label_idx = int(time_s * shapes_layer.metadata['label_sr'])

            labels_arr = shapes_layer.metadata['labels_array']
            mappings = shapes_layer.metadata['_mappings']

            if 0 <= label_idx < len(labels_arr):
                label = int(labels_arr[label_idx])
                if label in mappings and label != 0:
                    color = mappings[label]["color"]
                    color_list = color.tolist() if hasattr(color, 'tolist') else list(color)
                    shapes_layer.text = {
                        'string': [mappings[label]["name"]],
                        'color': [color_list],
                        'size': LABELS_OVERLAY_TEXT_SIZE,
                        'anchor': 'center'
                    }
                    return

            shapes_layer.text = {'string': [''], 'color': [[0, 0, 0]]}
        
    
        self.viewer.dims.events.current_step.connect(update_labels_text)
        
    
        update_labels_text()
        
  
        return shapes_layer
    
    def _remove_labels_shapes_layer(self):
        """Remove existing  shapes layer if it exists."""
        if "_labels" in self.viewer.layers:
            self.viewer.layers.remove("_labels")


    def refresh_labels_shapes_layer(self):
        """Refresh labels data without recreating the layer."""
        if "_labels" not in self.viewer.layers:
            self._add_labels_shapes_layer()
            return

        # Update both labels array and  mappings in existing layer's metadata
        shapes_layer = self.viewer.layers["_labels"]
        ds_kwargs = self.app_state.get_ds_kwargs()
        labels, _ = sel_valid(self.app_state.label_ds.labels, ds_kwargs)
        shapes_layer.metadata['labels_array'] = np.asarray(labels)
        shapes_layer.metadata['_mappings'] = self._mappings


class TemporaryLabelsDialog(QDialog):
    """Dialog for creating temporary labels for the current session."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Temporary Labels")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel("Enter label names (one per line):")
        layout.addWidget(info_label)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "label1\nlabel2\nlabel3\n...\n\n(background is added automatically as label 0)"
        )
        layout.addWidget(self.text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_labels(self):
        """Parse and return the list of label names."""
        text = self.text_edit.toPlainText()
        labels = [line.strip().replace(" ", "_") for line in text.split("\n") if line.strip()]
        return labels