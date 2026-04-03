"""Widget for labeling segments in movement data."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from napari.viewer import Viewer
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import ethograph as eto
from ethograph.gui.notify import notify
from ethograph.features.changepoints import snap_to_nearest_changepoint_time
from ethograph.labels.intervals import load_label_mapping
from ethograph.labels.plots import plot_confidence_pdf
from ethograph.labels.predictions import PredictionsStore
from ethograph.labels.tsv_store import save_labels_tsv
from ethograph.labels.intervals import (
    add_interval,
    delete_interval,
    empty_intervals,
    find_interval_at,
    get_interval_bounds,
)
from ethograph.utils.paths import find_mapping_file


logger = logging.getLogger(__name__)

from .app_constants import (
    LABELS_TABLE_MAX_HEIGHT,
    LABELS_TABLE_ROW_HEIGHT,
    LABELS_TABLE_ID_COLUMN_WIDTH,
    LABELS_TABLE_COLOR_COLUMN_WIDTH,
    DEFAULT_LAYOUT_SPACING,
)




class LabelsWidget(QWidget):
    """Widget for labeling movement labels in time series data."""
    
    highlight_spaceplot = Signal(float, float)

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.app_state = app_state

        self.data_widget = None  # Will be set after creation
        
        

        self.plot_container = None  # Will be set after creation
        self.meta_widget = None  # Will be set after creation
        self.changepoints_widget = None  # Will be set after creation
        self.io_widget = None  # Will be set after creation

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
        self.selected_labels = 0

        # Current  selection for editing (interval DataFrame index)
        self.current_labels_pos: int | None = None  # DataFrame index of selected interval
        self.current_labels: int | None = None  # ID of currently selected
        self.current_labels_is_prediction: bool = False  # Whether selected  is from predictions

        # Edit mode state
        self.old_labels_pos: int | None = None  # Original interval index when editing
        self.old_labels: int | None = None  # Original ID when editing
        
        # Frame tracking for  display
        self.previous_frame: int | None = None


        # UI components
        self.labels_table = None

        self._setup_ui()



        mapping_path = find_mapping_file()
        self._mappings = load_label_mapping(mapping_path) if mapping_path else {}
        self.app_state._label_mappings = self._mappings
        self._populate_labels_table()

    def refresh_mapping_for_data_dir(self, data_dir: Path | str):
        """Re-resolve mapping.txt now that a data directory is known.

        Called by DataWidget after loading a .nc file so that a local
        ``data_dir/.ethograph/mapping.txt`` is picked up when present.
        """
        mapping_path = find_mapping_file(data_dir)
        if mapping_path is None:
            return
        current_path = (
            Path(self.io_widget.mapping_file_path_edit.text())
            if self.io_widget else None
        )
        if current_path == mapping_path:
            return
        self._reload_mapping(str(mapping_path))
        if self.io_widget:
            self.io_widget.mapping_file_path_edit.setText(str(mapping_path))

    def set_data_widget(self, data_widget):
        """Set reference to the data widget for plot updates."""
        self.data_widget = data_widget


    def _mark_changes_unsaved(self):
        """Mark that changes have been made and are not saved."""
        self.app_state.changes_saved = False

    def set_plot_container(self, plot_container):
        """Set the plot container reference and connect click handler to all plots."""
        self.plot_container = plot_container
        plot_container.set_label_mappings(self._mappings)

        for plot in [plot_container.line_plot,
                     plot_container.spectrogram_plot,
                     plot_container.audio_trace_plot,
                     plot_container.heatmap_plot,
                     plot_container.neo_trace_plot,
                     plot_container.ephys_trace_plot]:
            if plot is not None:
                plot.plot_clicked.connect(self._on_plot_clicked)

    def set_meta_widget(self, meta_widget):
        """Set reference to the meta widget for layout refresh."""
        self.meta_widget = meta_widget

    def plot_all_labels(self, intervals_df, predictions_df=None):
        """Plot all labels for current trial based on interval data.

        Delegates to PlotContainer for centralized, synchronized label drawing
        across all plot types.

        Args:
            intervals_df: DataFrame with onset_s, offset_s, labels, individual columns
            predictions_df: Optional prediction intervals DataFrame
        """
        if intervals_df is None or self.plot_container is None:
            return

        show_predictions = (
            predictions_df is not None and
            self.data_widget is not None and
            self.data_widget.pred_show_predictions.isChecked()
        )

        self.plot_container.draw_all_labels(
            intervals_df,
            predictions_df=predictions_df,
            show_predictions=show_predictions,
        )

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        self.setLayout(layout)

        # Create labels table
        self._create_labels_table_and_edit_buttons()
        layout.addWidget(self.labels_table)

        layout.addStretch()

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
            QTableWidget { gridline-color: transparent; background: #444; color: #fff; }
            QTableWidget::item { padding: 0px 2px; color: #fff; }
            QTableWidget::item:selected { background: #ffe066; color: #000; }
            QHeaderView::section { padding: 0px 2px; background: #888; color: #fff; }
        """)

        self.labels_table.itemSelectionChanged.connect(self._on_table_selection_changed)



    def _browse_mapping_file(self):
        """Browse for a mapping.txt file and reload mappings."""
        current = find_mapping_file()
        start_dir = str(current.parent) if current else str(Path.home() / ".ethograph")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select mapping.txt file",
            start_dir,
            "Text files (*.txt);;All Files (*)"
        )
        if file_path:
            self.io_widget.mapping_file_path_edit.setText(file_path)
            self._reload_mapping(file_path)

    def _reload_mapping(self, mapping_path: str):
        """Reload  mappings from the specified path."""
        try:
            self._mappings = load_label_mapping(Path(mapping_path))
            self.app_state._label_mappings = self._mappings
            if self.plot_container:
                self.plot_container.set_label_mappings(self._mappings)
            if self.changepoints_widget:
                self.changepoints_widget.set_motif_mappings(self._mappings)
            if self.data_widget and self.data_widget.navigation_widget:
                self.data_widget.navigation_widget.set_mappings(self._mappings)
            self._populate_labels_table()
            self.refresh_labels_shapes_layer()
            if self.data_widget:
                self.data_widget.update_main_plot(preserve_x_range=True)
            notify(f"Loaded {len(self._mappings) - 1} labels from {Path(mapping_path).name}")
        except FileNotFoundError:
            notify(f"Mapping file not found: {mapping_path}", "warning")

    def _create_temporary_labels(self):
        """Open dialog to create temporary labels for this session."""
        dialog = TemporaryLabelsDialog(self)
        if dialog.exec_():
            labels = dialog.get_labels()
            if labels:
                from ethograph.utils.paths import default_config_dir
                data_dir = Path(self.app_state.nc_file_path).parent if self.app_state.nc_file_path else None
                config_dir = default_config_dir(data_dir)
                mapping_path = config_dir / "mapping_temporary.txt"
                mapping_path.parent.mkdir(parents=True, exist_ok=True)
                with open(mapping_path, "w") as f:
                    f.write("0 background\n")
                    for i, label in enumerate(labels, start=1):
                        f.write(f"{i} {label}\n")

                self.io_widget.mapping_file_path_edit.setText(str(mapping_path))
                self._mappings = load_label_mapping(mapping_path)
                self.app_state._label_mappings = self._mappings
                if self.plot_container:
                    self.plot_container.set_label_mappings(self._mappings)
                if self.changepoints_widget:
                    self.changepoints_widget.set_motif_mappings(self._mappings)
                if self.data_widget and self.data_widget.navigation_widget:
                    self.data_widget.navigation_widget.set_mappings(self._mappings)
                self._populate_labels_table()
                self.refresh_labels_shapes_layer()
                if self.data_widget:
                    self.data_widget.update_main_plot(preserve_x_range=True)
                notify(f"Loaded {len(labels)} temporary labels")

    def _import_predictions_from_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select predictions folder (.npy files)")
        if not folder:
            return
        individual = self.app_state.individuals_sel or "default"
        threshold = self.io_widget.pred_confidence_threshold_spin.value()
        try:
            store = PredictionsStore(folder)
            labels_df, confidence_levels = store.load_all(
                self.app_state.dt, individual, confidence_threshold=threshold,
            )
        except (FileNotFoundError, ValueError, AssertionError) as e:
            notify(str(e), severity="error")
            return

        folder_path = Path(folder)
        labels_dir = folder_path.parent
        tsv_path = labels_dir / f"{folder_path.name}.tsv"
        save_labels_tsv(tsv_path, labels_df)

        self.app_state.pred_labels_df = labels_df
        self.app_state.pred_store = store
        self.app_state.pred_confidence_threshold = threshold
        self.app_state.pred_confidence_levels = confidence_levels

        if self.data_widget:
            self.data_widget.pred_show_predictions.setEnabled(True)
            self.data_widget.pred_show_predictions.setChecked(True)
        self.io_widget.pred_confidence_pdf_btn.setEnabled(True)
        self.io_widget.pred_file_path_edit.setText(folder)

        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
            if self.data_widget.plot_container:
                self.data_widget.plot_container.labels_redraw_needed.emit()
        self.refresh_labels_shapes_layer()

    def _on_confidence_threshold_changed(self, _value):
        self.app_state.pred_confidence_threshold = self.io_widget.pred_confidence_threshold_spin.value()
        self.app_state.pred_segment_confidence_threshold = self.io_widget.pred_segment_confidence_threshold_spin.value()

    def _plot_confidence_pdf(self):
        import os
        store = getattr(self.app_state, "pred_store", None)
        labels_df = getattr(self.app_state, "pred_labels_df", None)
        if store is None or labels_df is None or labels_df.empty:
            notify("No predictions loaded.", severity="warning")
            return
        try:
            # Load all confidence arrays at click time for the PDF (one-off)
            confidence_map = {
                trial: store.get_confidence(trial, self.app_state.dt)
                for trial in self.app_state.trials
            }
            pdf_path, highlighted = plot_confidence_pdf(
                confidence_map, labels_df, self.app_state.dt, self._mappings,
                confidence_threshold=self.app_state.pred_confidence_threshold,
                segment_confidence_threshold=self.app_state.pred_segment_confidence_threshold,
            )
            self.app_state.pred_confidence_levels = {
                t: "low" if is_low else "high" for t, is_low in highlighted.items()
            }
            os.startfile(str(pdf_path))
        except Exception as e:
            notify(str(e), severity="error")

    labels_TO_KEY = {}

    # Row 1: 1-0 (Labels 1-10)
    number_keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    for i, key in enumerate(number_keys):
        _id = i + 1 if key != '0' else 10
        labels_TO_KEY[_id] = key

    # Row 2: Q-P (Labels 11-20)
    qwerty_row = ['q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p']
    for i, key in enumerate(qwerty_row):
        _id = i + 11
        labels_TO_KEY[_id] = key.upper()  # Display as uppercase for clarity

    # Row 3: A-; (Labels 21-30)
    home_row = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';']
    for i, key in enumerate(home_row):
        _id = i + 21
        labels_TO_KEY[_id] = key.upper() if key != ';' else ';'  # Keep ; as is

    # Also provide reverse mapping for key to _id
    KEY_TO_labels = {v.lower(): k for k, v in labels_TO_KEY.items()}
    
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

            shortcut = self.labels_TO_KEY.get(_id, "?")
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
        """Activate a label by shortcut: select cell, set up for labeling, and scroll to it."""
        _id = self.KEY_TO_labels.get(str(_key).lower(), _key)
        if _id not in self._mappings:
            return

        self.selected_labels = _id
        self.ready_for_label_click = True
        self.first_click = None
        self.second_click = None

        self.labels_table.blockSignals(True)
        for row in range(self.labels_table.rowCount()):
            for col in [0, 3]:
                item = self.labels_table.item(row, col)
                if item and item.data(Qt.UserRole) == _id:
                    self.labels_table.setCurrentItem(item)
                    self.labels_table.scrollToItem(item)
                    self.labels_table.blockSignals(False)
                    return
        self.labels_table.blockSignals(False)
        
            
            

    def _current_individual(self) -> str:
        """Return the currently selected individual name for interval operations."""
        ind = getattr(self.app_state, 'individuals_sel', None)
        if ind is not None and ind not in ("", "None"):
            return str(ind)
        if self.app_state.ds is not None and 'individuals' in self.app_state.ds.coords:
            return str(self.app_state.ds.coords['individuals'].values[0])
        return "default"

    def _on_plot_clicked(self, click_info):
        """Handle mouse clicks on the lineplot widget.

        Args:
            click_info: dict with 'x' (time coordinate) and 'button' (Qt button constant)
        """

        t_clicked = click_info["x"]
        button = click_info["button"]

        if t_clicked is None or not self.app_state.ready:
            return

        individual = self._current_individual()

        try:
            if button == Qt.LeftButton and not self.ready_for_label_click:
                self._check_labels_click(t_clicked, individual)

        except (KeyError, IndexError, ValueError, AttributeError) as e:
            logger.error("Error in plot click handling: %s", e)
            return

        # Without video, any click (not in label mode) jumps the time marker
        if getattr(self.app_state, 'video', None) is None and not self.ready_for_label_click:
            if self.plot_container is not None:
                self.plot_container.update_time_marker_by_time(t_clicked)
            if button == Qt.LeftButton:
                return

        # Handle right-click - seek video to clicked position
        if button == Qt.RightButton and self.app_state.video:
            frame = self.app_state.video.time_to_frame(t_clicked)
            self.app_state.video.seek_to_frame(frame)

        # Handle left-click for labeling/editing (only in label mode)
        elif button == Qt.LeftButton and self.ready_for_label_click:

            # Snap to nearest changepoint if available (in time domain)
            if self.changepoints_widget and self.changepoints_widget.is_changepoint_correction_enabled():
                t_snapped = self._snap_to_changepoint_time(t_clicked)
            else:
                t_snapped = t_clicked

            if self.first_click is None:
                self.first_click = t_snapped
            else:
                self.second_click = t_snapped
                self._apply_label()



    def _check_labels_click(self, t_clicked: float, individual: str) -> bool:
        """Check if the click is on an existing interval and select it.

        Args:
            t_clicked: Time in seconds of the click
            individual: Individual name to check
        """
        df = self.app_state.label_intervals
        if df is None or df.empty:
            return False

        idx = find_interval_at(df, t_clicked, individual)
        if idx is not None:
            onset_s, offset_s, labels = get_interval_bounds(df, idx)
            self.current_labels = labels
            self.current_labels_pos = idx
            self.current_labels_is_prediction = False
            self.highlight_spaceplot.emit(onset_s, offset_s)
            self.selected_labels = labels
            return True
        return False

    def _snap_to_changepoint_time(self, t_clicked: float) -> float:
        """Snap the clicked time (seconds) to the nearest changepoint time.

        Works entirely in the time domain. Also considers audio changepoints.
        """
        store = getattr(self.app_state, 'feature_store', None)
        if store is not None:
            feature = getattr(self.app_state, 'features_sel', None)
            cp_times = store.get_cp_times(feature)
            if len(cp_times) == 0:
                return t_clicked
            nearest_idx = np.argmin(np.abs(cp_times - t_clicked))
            return float(cp_times[nearest_idx])

        time_coord = self.app_state.time_coord
        if time_coord is None:
            return t_clicked

        ds_kwargs = self.app_state.get_ds_kwargs()
        feature_sel = self.app_state.features_sel

        snapped = snap_to_nearest_changepoint_time(
            t_clicked, self.app_state.ds, feature_sel, time_coord.values, **ds_kwargs
        )

        return snapped

    def _apply_label(self):
        """Apply the selected label to the selected time range using intervals."""
        if self.first_click is None or self.second_click is None:
            return

        onset_s = min(self.first_click, self.second_click)
        offset_s = max(self.first_click, self.second_click)
        individual = self._current_individual()

        self.highlight_spaceplot.emit(onset_s, offset_s)

        df = self.app_state.label_intervals
        if df is None:
            df = empty_intervals()

        # If editing, delete the old interval first
        if self.old_labels_pos is not None:
            if self.old_labels_pos in df.index:
                df = delete_interval(df, self.old_labels_pos)
            self.old_labels_pos = None
            self.old_labels = None

        df = add_interval(df, onset_s, offset_s, self.selected_labels, individual)
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(self.app_state.trials_sel, df)
        
        # Post purge/stich step 
        self.changepoints_widget.cp_correction_from_labelling()
        df = self.app_state.label_intervals

        # Auto-select the newly created interval for immediate playback
        new_idx = find_interval_at(df, (onset_s + offset_s) / 2, individual)
        self.current_labels_pos = new_idx
        self.current_labels = self.selected_labels
        self.current_labels_is_prediction = False

        self.first_click = None
        self.second_click = None
        self.ready_for_label_click = False

        if self.io_widget:
            self.io_widget._human_verification_true(mode="single_trial")
        self._mark_changes_unsaved()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
        self._seek_to_frame(onset_s)
        self.refresh_labels_shapes_layer()

        

    def _seek_to_frame(self, time_s: float):
        """Seek video and update time marker to the specified time in seconds."""
        if hasattr(self.app_state, 'video') and self.app_state.video:
            video_frame = self.app_state.video.time_to_frame(time_s)
            self.app_state.video.seek_to_frame(video_frame)
        elif self.plot_container:
            self.plot_container.update_time_marker_by_time(time_s)


    def _delete_label(self):
        if self.current_labels_pos is None:
            return

        df = self.app_state.label_intervals
        if df is None or df.empty:
            return

        if self.current_labels_pos not in df.index:
            self.current_labels_pos = None
            return

        onset_s, _, _ = get_interval_bounds(df, self.current_labels_pos)
        df = delete_interval(df, self.current_labels_pos)
        self.app_state.label_intervals = df
        self.app_state.set_trial_intervals(self.app_state.trials_sel, df)

        self.current_labels_pos = None
        self.current_labels = None
        self.current_labels_is_prediction = False

        self._mark_changes_unsaved()
        if self.data_widget:
            self.data_widget.update_main_plot(preserve_x_range=True)
        self.refresh_labels_shapes_layer()

    def _edit_label(self):
        """Enter edit mode for adjusting interval boundaries."""
        if self.current_labels_pos is None:
            logger.warning("No label selected. Click on a label first to select it.")
            return

        self.old_labels_pos = self.current_labels_pos
        self.old_labels = self.current_labels

        self.ready_for_label_click = True
        self.first_click = None
        self.second_click = None

    def _play_segment(self):
        if self.current_labels_pos is None:
            logger.warning("No label selected for playback")
            return

        df = self.app_state.label_intervals
        if df is None or self.current_labels_pos not in df.index:
            return

        onset_s, offset_s, _ = get_interval_bounds(df, self.current_labels_pos)

        if self.app_state.video:
            start_frame = self.app_state.video.time_to_frame(onset_s)
            end_frame = self.app_state.video.time_to_frame(offset_s)
            self.app_state.video.play_segment(start_frame, end_frame)
        else:
            self._play_audio_segment(onset_s, offset_s)

    def _play_audio_segment(self, onset_s: float, offset_s: float):
        if self.plot_container and hasattr(self.plot_container, 'audio_player'):
            self.plot_container.audio_player.play_segment(onset_s, offset_s)




    def _get_canvas_widget(self):
        """Return the Qt widget that holds the napari OpenGL canvas."""
        qt_viewer = getattr(self.viewer.window, '_qt_viewer', None)
        if qt_viewer is None:
            return None
        # _qt_viewer.canvas.native is the actual OpenGL widget
        canvas = getattr(qt_viewer, 'canvas', None)
        native = getattr(canvas, 'native', None) if canvas else None
        return native or qt_viewer

    def _add_labels_shapes_layer(self):
        """Add a Qt QLabel overlay on the napari canvas.

        Much faster than a napari Shapes layer: setText() is a
        trivial Qt repaint instead of a full OpenGL shapes render pass.
        """
        if getattr(self, '_label_overlay', None) is not None:
            return

        canvas_widget = self._get_canvas_widget()
        if canvas_widget is None:
            return

        overlay = QLabel(canvas_widget)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        overlay.setAlignment(Qt.AlignCenter)
        overlay.setStyleSheet(self._overlay_stylesheet("white"))
        overlay.setFixedHeight(36)
        overlay.hide()
        self._label_overlay = overlay
        self._label_overlay_last_text = ""

        def _reposition_overlay():
            if self._label_overlay is None:
                return
            parent = self._label_overlay.parent()
            if parent is None:
                return
            pw = parent.width()
            ow = self._label_overlay.sizeHint().width()
            self._label_overlay.move((pw - ow) // 2, 8)

        self._reposition_overlay = _reposition_overlay

        # Track canvas resizes to keep the overlay centred
        orig_resize = canvas_widget.resizeEvent

        def _on_canvas_resize(event):
            orig_resize(event)
            _reposition_overlay()

        canvas_widget.resizeEvent = _on_canvas_resize

        def _update_labels_text(event=None):
            overlay = getattr(self, '_label_overlay', None)
            if overlay is None:
                return
            if getattr(self, '_label_overlay_hidden', False):
                overlay.hide()
                return
            video = getattr(self.app_state, 'video', None)
            video_frame = self.viewer.dims.current_step[0]
            if video:
                time_s = video.frame_to_time(video_frame)
            elif hasattr(self.app_state, 'video_fps') and self.app_state.video_fps:
                time_s = video_frame / self.app_state.video_fps
            else:
                return
            df = self.app_state.label_intervals
            ind = self._current_individual()
            mappings = self._mappings

            text = ""
            css_color = "white"
            if df is not None and not df.empty:
                idx = find_interval_at(df, time_s, ind)
                if idx is not None:
                    _, _, labels = get_interval_bounds(df, idx)
                    if labels in mappings and labels != 0:
                        text = mappings[labels]["name"]
                        color = mappings[labels]["color"]
                        if hasattr(color, 'tolist'):
                            color = color.tolist()
                        r, g, b = (int(c * 255) for c in color[:3])
                        css_color = f"rgb({r},{g},{b})"

            if text == self._label_overlay_last_text:
                return
            self._label_overlay_last_text = text

            if not text:
                overlay.hide()
                return
            overlay.setStyleSheet(self._overlay_stylesheet(css_color))
            overlay.setText(text)
            overlay.adjustSize()
            overlay.setFixedHeight(36)
            _reposition_overlay()
            overlay.show()
            overlay.raise_()

        self.viewer.dims.events.current_step.connect(_update_labels_text)
        self._update_labels_text = _update_labels_text
        _update_labels_text()

    @staticmethod
    def _overlay_stylesheet(color: str) -> str:
        return (
            "QLabel {"
            "  background: rgba(0, 0, 0, 160);"
            f"  color: {color};"
            "  font-size: 22px;"
            "  font-weight: bold;"
            "  padding: 4px 16px;"
            "  border-radius: 6px;"
            "}"
        )

    def _remove_labels_shapes_layer(self):
        """Remove the Qt label overlay."""
        overlay = getattr(self, '_label_overlay', None)
        if overlay is not None:
            overlay.setParent(None)
            overlay.deleteLater()
            self._label_overlay = None
            self._label_overlay_last_text = ""

    def refresh_labels_shapes_layer(self):
        """Refresh: ensure overlay exists, then force an update."""
        if getattr(self, '_label_overlay', None) is None:
            self._add_labels_shapes_layer()
            return
        self._label_overlay_last_text = ""
        if hasattr(self, '_update_labels_text'):
            self._update_labels_text()


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