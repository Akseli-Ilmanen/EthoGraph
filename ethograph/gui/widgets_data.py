"""Widget for selecting start/stop times and playing a segment in napari."""

import logging
import os
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import xarray as xr
from napari.viewer import Viewer
from qtpy.QtCore import QSortFilterProxyModel, Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)



import ethograph as eto
from ethograph.gui.notify import notify, notify_dialog
from ethograph.labels.intervals import get_interval_bounds
from ethograph.gui.modality import FileSource
from ethograph.gui.plots_timeseriessource import compute_trial_alignment



from .app_constants import (
    DEFAULT_LAYOUT_MARGIN,
    DEFAULT_LAYOUT_SPACING,
    SIDEBAR_AFTER_LOAD_WIDTH_RATIO,
)
from .data_loader import load_dataset
from .makepretty import (
    ElidedDelegate,
    clean_display_labels,
    find_combo_index,
    get_combo_value,
    set_combo_to_value,
)
from .plots_ephystrace import get_loader as get_ephys_loader
from .plots_space import SpacePlot
from .plots_spectrogram import SharedAudioCache
from .pose_render import (
    PoseDisplayManager,
    strip_common_prefix,
)
from .video_manager import VideoManager,  is_url

logger = logging.getLogger(__name__)


@dataclass
class _PanelDef:
    """Declarative description of a panel toggle checkbox."""
    name: str                           # internal identifier / checkbox attr prefix
    label: str                          # displayed checkbox text
    row: int                            # UI row in panels_groupbox (1, 2, or 3)
    state_attr: str | None = None       # app_state attribute to sync with visibility
    container_method: str | None = None # plot_container.method(visible) to call
    autoscale_plot: str | None = None   # plot_container.X for autoscale on show
    audio_row: bool = False             # part of the hidden-when-no-audio widget group
    on_toggle: str | None = None        # self.method(visible) called after standard actions


_PANEL_DEFS: list[_PanelDef] = [
    _PanelDef("audiotrace",   "AudioTrace",   row=1, audio_row=True,
              state_attr="audiotrace_visible",
              container_method="set_audiotrace_visible",
              autoscale_plot="audio_trace_plot"),
    _PanelDef("spectrogram",  "Spectrogram",  row=1, audio_row=True,
              state_attr="spectrogram_visible",
              container_method="set_spectrogram_visible",
              autoscale_plot="spectrogram_plot"),
    _PanelDef("neo_viewer",   "Neo-Viewer",   row=1,
              container_method="set_neo_visible",
              on_toggle="_on_neo_panel_toggle"),
    _PanelDef("phy_viewer",   "Phy-Viewer",   row=1,
              state_attr="ephys_visible",
              container_method="set_ephys_visible",
              on_toggle="_on_phy_panel_toggle"),
    _PanelDef("featureplot",  "FeaturePlot",  row=1,
              state_attr="featureplot_visible",
              container_method="set_featureplot_visible"),
    _PanelDef("video_viewer", "VideoViewer",  row=1,
              state_attr="video_viewer_visible",
              on_toggle="_on_video_viewer_toggle"),
    _PanelDef("pose_markers", "PoseMarkers",  row=1,
              state_attr="pose_markers_visible",
              on_toggle="_on_pose_markers_toggle"),
]


def make_searchable(combo_box: QComboBox) -> None:
    combo_box.setFocusPolicy(Qt.StrongFocus)
    combo_box.setEditable(True)
    combo_box.setInsertPolicy(QComboBox.NoInsert)

    filter_model = QSortFilterProxyModel(combo_box)
    filter_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
    filter_model.setSourceModel(combo_box.model())

    completer = QCompleter(filter_model, combo_box)
    completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
    combo_box.setCompleter(completer)
    combo_box.lineEdit().textEdited.connect(filter_model.setFilterFixedString)


class DataPanel(QWidget):
    """Visible panel for the 'Data' collapsible section in the sidebar.

    Organised into three tabs: Main, Pose, Audio.
    """

    ENERGY_DISPLAY_NAMES = {
        "energy_lowpass": "SOS lowpass envelope",
        "energy_highpass": "SOS highpass envelope",
        "energy_band": "SOS bandpass envelope",
        "energy_meansquared": "Vocalpy meansquared (amplitude)",
        "energy_ava": "Vocalpy AVA (spectral power)",
    }

    def __init__(self, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self._update_pose_callback = None
        layout = QVBoxLayout()
        layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        layout.setContentsMargins(DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN)
        self.setLayout(layout)

        self._create_toggle_buttons(layout)

        self.main_panel = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.main_panel.setLayout(main_layout)
        self._create_main_section(main_layout)
        layout.addWidget(self.main_panel)

        self.pose_panel = QWidget()
        pose_layout = QVBoxLayout()
        pose_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        pose_layout.setContentsMargins(2, 2, 2, 2)
        self.pose_panel.setLayout(pose_layout)
        self._create_pose_section(pose_layout)
        layout.addWidget(self.pose_panel)

        self.audio_panel = QWidget()
        audio_layout = QVBoxLayout()
        audio_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        audio_layout.setContentsMargins(2, 2, 2, 2)
        self.audio_panel.setLayout(audio_layout)
        self._create_audio_section(audio_layout)
        layout.addWidget(self.audio_panel)

        self._show_panel("main")

    def _create_toggle_buttons(self, parent_layout):
        toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_widget.setLayout(toggle_layout)

        self.main_toggle = QPushButton("Main")
        self.main_toggle.setCheckable(True)
        self.main_toggle.clicked.connect(lambda: self._show_panel("main"))
        toggle_layout.addWidget(self.main_toggle)

        self.pose_toggle = QPushButton("Pose")
        self.pose_toggle.setCheckable(True)
        self.pose_toggle.clicked.connect(lambda: self._show_panel("pose"))
        toggle_layout.addWidget(self.pose_toggle)

        self.audio_toggle = QPushButton("Audio")
        self.audio_toggle.setCheckable(True)
        self.audio_toggle.clicked.connect(lambda: self._show_panel("audio"))
        toggle_layout.addWidget(self.audio_toggle)

        parent_layout.addWidget(toggle_widget)

    def _show_panel(self, panel_name):
        panels = {
            "main": (self.main_panel, self.main_toggle),
            "pose": (self.pose_panel, self.pose_toggle),
            "audio": (self.audio_panel, self.audio_toggle),
        }
        for name, (panel, toggle) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)

    def _create_main_section(self, parent_layout):
        self.coords_groupbox = QGroupBox("Xarray coords")
        self.coords_groupbox_layout = QFormLayout()
        self.coords_groupbox_layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        self.coords_groupbox_layout.setContentsMargins(
            DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN,
        )
        self.coords_groupbox.setLayout(self.coords_groupbox_layout)
        parent_layout.addWidget(self.coords_groupbox)

        self.slot_groupbox = QGroupBox("Space/Cameras")
        slot_vbox = QVBoxLayout()
        slot_vbox.setSpacing(2)
        slot_vbox.setContentsMargins(4, 4, 4, 4)
        self.slot_layout = QHBoxLayout()
        self.slot_layout.setSpacing(5)
        self.slot_row2_layout = QHBoxLayout()
        self.slot_row2_layout.setSpacing(5)
        slot_vbox.addLayout(self.slot_layout)
        slot_vbox.addLayout(self.slot_row2_layout)
        self.slot_groupbox.setLayout(slot_vbox)
        self.slot_groupbox.hide()
        parent_layout.addWidget(self.slot_groupbox)

        self.panels_groupbox = QGroupBox("Plot panels")
        panels_vbox = QVBoxLayout()
        panels_vbox.setSpacing(2)
        panels_vbox.setContentsMargins(4, 4, 4, 4)
        self.panels_groupbox.setLayout(panels_vbox)

        for i in range(1, 6):
            row = QHBoxLayout()
            row.setSpacing(10)
            setattr(self, f"panels_row{i}_layout", row)
            panels_vbox.addLayout(row)

        parent_layout.addWidget(self.panels_groupbox)

        self.overlays_groupbox = QGroupBox("Overlays")
        self.overlays_layout = QHBoxLayout()
        self.overlays_layout.setSpacing(15)
        self.overlays_groupbox.setLayout(self.overlays_layout)
        parent_layout.addWidget(self.overlays_groupbox)

        parent_layout.addStretch()

    def _create_pose_section(self, parent_layout):
        self.pose_groupbox = QGroupBox("Pose controls")
        pose_layout = QVBoxLayout()
        pose_layout.setSpacing(2)
        pose_layout.setContentsMargins(4, 4, 4, 4)
        self.pose_groupbox.setLayout(pose_layout)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence >="))
        self.pose_hide_threshold_spin = QDoubleSpinBox()
        self.pose_hide_threshold_spin.setObjectName("pose_hide_threshold_spin")
        self.pose_hide_threshold_spin.setRange(0.0, 1.0)
        self.pose_hide_threshold_spin.setSingleStep(0.1)
        self.pose_hide_threshold_spin.setDecimals(1)
        self.pose_hide_threshold_spin.setFixedWidth(60)
        self.pose_hide_threshold_spin.setToolTip(
            "Hide pose markers with confidence below this value (0.0-1.0)"
        )
        self.pose_hide_threshold_spin.setValue(self.app_state.pose_hide_threshold)
        threshold_layout.addWidget(self.pose_hide_threshold_spin)

        threshold_layout.addWidget(QLabel("Size:"))
        self.pose_point_size_spin = QDoubleSpinBox()
        self.pose_point_size_spin.setObjectName("pose_point_size_spin")
        self.pose_point_size_spin.setRange(1.0, 50.0)
        self.pose_point_size_spin.setSingleStep(1.0)
        self.pose_point_size_spin.setDecimals(0)
        self.pose_point_size_spin.setFixedWidth(55)
        self.pose_point_size_spin.setValue(10.0)
        threshold_layout.addWidget(self.pose_point_size_spin)

        threshold_layout.addStretch()
        pose_layout.addLayout(threshold_layout)

        row2 = QHBoxLayout()
        self.pose_show_text_checkbox = QCheckBox("Show text")
        self.pose_show_text_checkbox.setChecked(False)
        self.pose_show_text_checkbox.setToolTip("Show keypoint/individual labels on pose markers")
        row2.addWidget(self.pose_show_text_checkbox)

        self.rotate_btn = QPushButton("Rotate video/pose by 90°")
        self.rotate_btn.setToolTip("Rotate all video and pose layers by 90° clockwise")
        row2.addWidget(self.rotate_btn)
        row2.addStretch()
        pose_layout.addLayout(row2)

        # Select All / Deselect All
        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        select_all_btn.clicked.connect(lambda: self._set_all_keypoints_checked(True))
        deselect_all_btn.clicked.connect(lambda: self._set_all_keypoints_checked(False))
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(deselect_all_btn)
        btn_row.addStretch()
        pose_layout.addLayout(btn_row)

        # Keypoints table (inline, scrollable)
        self.keypoints_table = QTableWidget(0, 2)
        self.keypoints_table.setHorizontalHeaderLabels(["Show", "Keypoint"])
        self.keypoints_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.keypoints_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.keypoints_table.verticalHeader().setVisible(False)
        pose_layout.addWidget(self.keypoints_table, stretch=1)

        self.pose_groupbox.hide()
        parent_layout.addWidget(self.pose_groupbox, stretch=1)

    def _create_audio_section(self, parent_layout):
        group = QGroupBox("Energy envelope")
        grid = QGridLayout()
        group.setLayout(grid)

        grid.addWidget(QLabel("Energy metric:"), 0, 0)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(self.ENERGY_DISPLAY_NAMES.values())
        grid.addWidget(self.metric_combo, 0, 1, 1, 2)

        self.energy_configure_btn = QPushButton("Configure...")
        self.energy_configure_btn.setToolTip("Open parameter editor for selected energy metric")
        grid.addWidget(self.energy_configure_btn, 1, 0, 1, 3)

        parent_layout.addWidget(group)
        parent_layout.addStretch()

        self._restore_energy_selections()

    def _restore_energy_selections(self):
        metric = self.app_state.get_with_default("energy_metric")
        display = self.ENERGY_DISPLAY_NAMES.get(metric, "SOS lowpass envelope")
        self.metric_combo.setCurrentText(display)

    def energy_display_to_key(self, display_text: str) -> str:
        for key, val in self.ENERGY_DISPLAY_NAMES.items():
            if val == display_text:
                return key
        return "energy_lowpass"

    def _set_all_keypoints_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        self.keypoints_table.blockSignals(True)
        for row in range(self.keypoints_table.rowCount()):
            item = self.keypoints_table.item(row, 0)
            if item:
                item.setCheckState(state)
        self.keypoints_table.blockSignals(False)
        if self._update_pose_callback:
            self._update_pose_callback()


class DataWidget(QWidget):
    """Orchestrator widget — loads data, manages selections, updates plots."""

    def __init__(
        self,
        napari_viewer: Viewer,
        app_state,
        meta_widget,
        io_widget,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        layout = QFormLayout()
        layout.setSpacing(DEFAULT_LAYOUT_SPACING)
        layout.setContentsMargins(DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN, DEFAULT_LAYOUT_MARGIN)
        self.setLayout(layout)
        self.app_state = app_state
        self.meta_widget = meta_widget
        self.io_widget = io_widget
        self.plot_container = None
        self.labels_widget = None
        self.plot_settings_widget = None
        self.ephys_widget = None
        self.audio_player = None
        self.video_path = None
        self.audio_path = None
        self.space_plot = None

        self.combos = {}
        self.all_checkboxes = {}
        self.controls = []

        self.source_software = None
        self.file_path = None

        self.video_mgr = VideoManager(napari_viewer, app_state)
        self.video_mgr.set_frame_changed_callback(self._on_primary_frame_changed)
        self.pose_mgr: PoseDisplayManager | None = None  # created after set_data_panel
        self.app_state.audio_video_sync = None
        self.catalog = None  # DataCatalog set after load

    def set_data_panel(self, panel: DataPanel):
        self.data_panel = panel
        self.coords_groupbox = panel.coords_groupbox
        self.coords_groupbox_layout = panel.coords_groupbox_layout
        self.slot_groupbox = panel.slot_groupbox
        self.slot_layout = panel.slot_layout
        self.slot_row2_layout = panel.slot_row2_layout
        self.panels_groupbox = panel.panels_groupbox
        self.panels_row1_layout = panel.panels_row1_layout
        self.panels_row2_layout = panel.panels_row2_layout
        self.panels_row3_layout = panel.panels_row3_layout
        self.panels_row4_layout = panel.panels_row4_layout
        self.panels_row5_layout = panel.panels_row5_layout
        self.overlays_groupbox = panel.overlays_groupbox
        self.overlays_layout = panel.overlays_layout
        self.pose_groupbox = panel.pose_groupbox
        self.pose_hide_threshold_spin = panel.pose_hide_threshold_spin
        self.pose_show_text_checkbox = panel.pose_show_text_checkbox
        self.pose_point_size_spin = panel.pose_point_size_spin
        self.keypoints_table = panel.keypoints_table

        self.pose_mgr = PoseDisplayManager(self.viewer, self.app_state, self.video_mgr, self)
        self.app_state.keypoints_changed.connect(self.populate_keypoints)

        panel.pose_hide_threshold_spin.valueChanged.connect(self._on_pose_hide_threshold_changed)
        panel.pose_show_text_checkbox.stateChanged.connect(self._on_pose_text_toggled)
        panel.pose_point_size_spin.valueChanged.connect(self._on_pose_point_size_changed)
        panel.rotate_btn.clicked.connect(self.pose_mgr.on_rotate_video_pose)
        panel._update_pose_callback = self.update_pose

        panel.energy_configure_btn.clicked.connect(self._open_energy_params)

    def populate_keypoints(self, keypoint_names: list[str]) -> None:
        try:
            self.keypoints_table.cellChanged.disconnect(self._on_keypoint_toggled)
        except (TypeError, RuntimeError):
            pass
        self.keypoints_table.blockSignals(True)
        self.keypoints_table.setRowCount(len(keypoint_names))
        for row, name in enumerate(keypoint_names):
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox_item.setCheckState(Qt.Checked)
            self.keypoints_table.setItem(row, 0, checkbox_item)

            name_item = QTableWidgetItem(str(name))
            name_item.setFlags(Qt.ItemIsEnabled)
            self.keypoints_table.setItem(row, 1, name_item)
        self.keypoints_table.blockSignals(False)
        self.keypoints_table.cellChanged.connect(self._on_keypoint_toggled)
        self.pose_groupbox.show()

    def get_hidden_keypoints(self) -> set[str]:
        hidden: set[str] = set()
        for row in range(self.keypoints_table.rowCount()):
            checkbox_item = self.keypoints_table.item(row, 0)
            name_item = self.keypoints_table.item(row, 1)
            if checkbox_item and name_item:
                if checkbox_item.checkState() != Qt.Checked:
                    hidden.add(name_item.text())
        return hidden

    def _on_keypoint_toggled(self, row: int, column: int):
        if column != 0:
            return
        self.update_pose()

    def _on_pose_hide_threshold_changed(self, value: float):
        self.app_state.pose_hide_threshold = value
        self.update_pose()

    def _on_pose_text_toggled(self, state: int):
        self.pose_mgr.apply_pose_style()

    def _on_pose_point_size_changed(self, value: float):
        self.pose_mgr.apply_pose_style()

    def set_references(
        self, plot_container, labels_widget, plot_settings_widget,
        navigation_widget, changepoints_widget=None,
        ephys_widget=None, layout_mgr=None, trials_widget=None,
    ):
        self.plot_container = plot_container
        self.labels_widget = labels_widget
        self.plot_settings_widget = plot_settings_widget
        self.navigation_widget = navigation_widget
        self.changepoints_widget = changepoints_widget
        self.ephys_widget = ephys_widget
        self.layout_mgr = layout_mgr
        self.trials_widget = trials_widget

        if trials_widget is not None:
            trials_widget.trials_filtered.connect(self._on_trials_filtered)

        plot_container.time_marker_updated.connect(self._on_time_marker_updated)

        if changepoints_widget is not None:
            changepoints_widget.request_plot_update.connect(self._on_plot_update_request)

        plot_container.plot_changed.connect(self._on_feature_plot_type_changed)

    def _on_plot_update_request(self):
        if not self.app_state.ready or not self.plot_container:
            return
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _on_feature_plot_type_changed(self, plot_type: str):
        self._update_sort_button_state()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _cleanup_load_state(self):
        dt = getattr(self.app_state, 'dt', None)
        if dt is not None:
            dt.close()
            self.app_state.dt = None
        self.app_state.ds = None
        self.app_state.data_loader = None
        self.app_state._all_labels_df = None
        self.app_state.labels_confidence_ds = None
        self.catalog = None
        self.app_state.ready = False

    def _cancel_load(self, reason: str):
        notify_dialog(reason, "warning", "Load cancelled", self)
        self._cleanup_load_state()




    def on_load_clicked(self):
        if not self.app_state.nc_file_path:
            notify_dialog("Please select a data file (.nc, .nwb, .npz) or folder", "warning", "Load cancelled", self)
            return

        nc_file_path = self.io_widget.get_nc_file_path()

        self.app_state.has_video = bool(self.app_state.video_folder)
        self.app_state.has_pose = bool(self.app_state.pose_folder)

        try:
            self.app_state.dt, all_labels_df, catalog = load_dataset(
                nc_file_path,
                require_fps=self.app_state.has_pose,
                progress_callback=getattr(self.app_state, "_progress_callback", None),
                max_trials=getattr(self.app_state, "_dandi_max_trials", None),
                dandiset_id=getattr(self.app_state, "_dandi_dandiset_id", None),
                import_labels=self.io_widget.import_labels_checkbox.isChecked(),
            )
            self.catalog = catalog
            nwb_video_folder = self.app_state.dt.attrs.get("nwb_video_folder")
            if nwb_video_folder and not self.app_state.video_folder:
                from .dialog_video_downsample import offer_downsample
                nwb_video_folder = offer_downsample(str(nwb_video_folder), parent=self)
                self.app_state.video_folder = nwb_video_folder
        except (OSError, ValueError, KeyError) as e:
            self._cancel_load(f"Failed to load dataset: {e}")
            return

        self.app_state.trial_conditions = self.catalog.trial_conditions

        # Extract data_loader from pynapple/NWB loader (xarray loader created later after ds is set)
        self._pending_loader = self.app_state.dt.attrs.pop("data_loader", None)

        dt_attrs = self.app_state.dt.attrs

        # NWB-embedded pose: position variables already in ds, no pose_folder needed
        has_nwb_pose = "nwb_source" in dt_attrs and bool(dt_attrs.get("nwb_pose_keys"))
        if has_nwb_pose:
            self.app_state.has_video = True
            self.app_state.has_pose = True

        # Remote cameras: URLs in session-level cameras — no video folder needed
        cameras_list = self.app_state.dt.cameras
        if any(is_url(c) for c in cameras_list):
            self.app_state.has_video = True

        # NWB-embedded ephys: path stored in attrs, no local ephys file needed
        nwb_ephys_series = dt_attrs.get("nwb_ephys_series")
        nwb_ephys_path = dt_attrs.get("nwb_ephys_path")
        if nwb_ephys_series and nwb_ephys_path:
            if not self.app_state.ephys_path:
                self.app_state.ephys_path = nwb_ephys_path
            display_name = f"{nwb_ephys_series} (NWB)"
            self.app_state.ephys_source_map[display_name] = (nwb_ephys_path, "0", 0)
            self.app_state.ephys_stream_sel = display_name

        self.app_state.has_audio = bool(self.app_state.dt.mics and self.app_state.audio_folder)
        self.app_state.has_neo = bool(self.app_state.ephys_path) or bool(nwb_ephys_series and nwb_ephys_path)
        self.app_state.has_neurons = bool(self.app_state.neurons_path)

        # Populate ephys_source_map (streams are NOT added to features list)
        if self.app_state.ephys_path:
            try:
                self.io_widget._expand_ephys_with_streams(
                    self.app_state.ephys_path, self.app_state.ds,
                )
            except (OSError, ValueError, KeyError) as e:
                self._cancel_load(f"Failed to load ephys features: {e}")
                return
            

        downsample_factor = self.io_widget.get_downsample_factor()
        if downsample_factor is not None:
            self.app_state.dt = eto.downsample_trialtree(self.app_state.dt, downsample_factor)
            self.app_state.downsample_factor_used = downsample_factor
            logger.info("Downsampled data by factor %d", downsample_factor)
        else:
            self.app_state.downsample_factor_used = None

        self.io_widget.disable_downsample_controls()

        trials = self.app_state.dt.trials

        self.app_state._all_labels_df = all_labels_df

        self.app_state.trials = sorted(trials)
        self.app_state.ds = self.app_state.dt.trial(self.app_state.trials[0])

        # Now ds is set — create or finalize DataLoader
        store = self._pending_loader
        if store is None:
            from ethograph.io.catalog import XarrayLoader, catalog_from_xarray
            cat = catalog_from_xarray(self.app_state.ds, self.app_state.dt)
            store = XarrayLoader(self.app_state.ds, cat)
        self.app_state.data_loader = store
        self._pending_loader = None

        # Set trials_sel early so _expand_mics_with_channels / get_media
        # can resolve filenames during UI creation.
        trial = getattr(self.app_state, 'trials_sel', None)
        try:
            is_nan = np.isnan(trial)
        except (TypeError, ValueError):
            is_nan = False
        if not trial or is_nan or trial not in self.app_state.trials:
            self.app_state.trials_sel = self.app_state.trials[0]

        missing = self._validate_media_files()
        if missing:

            self._cancel_load(
                "Missing media files for first trial:\n" + "\n".join(missing)
            )
            return

        self._create_trial_controls()

        self._restore_or_set_defaults()
        self._set_controls_enabled(True)
        self.app_state.ready = True

        self.io_widget.on_load_complete()
        self.labels_widget.refresh_mapping_for_data_dir(Path(nc_file_path).parent)
        self.changepoints_widget.setEnabled(True)
        self.plot_settings_widget.set_enabled_state()
        if self.ephys_widget:
            self.ephys_widget.setEnabled(True)
            self.ephys_widget.populate_ephys_default_path()

        self.meta_widget.configure_layout_for_data()

        # Re-apply sidebar cap after load-triggered dock/layout rearrangement.
        if getattr(self, 'layout_mgr', None) is not None:
            QTimer.singleShot(
                0,
                lambda: self.layout_mgr.set_sidebar_default_width(self.meta_widget, SIDEBAR_AFTER_LOAD_WIDTH_RATIO),
            )

        self.update_trials_combo()
        self._load_trial_with_fallback()

        if self.navigation_widget:
            self.navigation_widget.set_mappings(self.labels_widget._mappings)
            self.navigation_widget.refresh_after_load()

        self.view_mode_combo.show()

    # ------------------------------------------------------------------
    # Trials combo
    # ------------------------------------------------------------------

    def update_trials_combo(self) -> None:
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
            bg_color = QColor(144, 238, 144) if is_verified else QColor(255, 182, 193)
            combo.setItemData(index, bg_color, Qt.BackgroundRole)
            text_color = QColor(0, 100, 0) if is_verified else QColor(139, 0, 0)
            combo.setItemData(index, text_color, Qt.ForegroundRole)

        combo.setCurrentText(str(self.app_state.trials_sel))
        combo.blockSignals(False)

    def _collect_trial_status(self) -> Dict[int, int]:
        trial_status = {}
        for trial in self.app_state.trials:
            is_verified = self.app_state.get_trial_meta(trial).get('human_verified', 0)
            trial_status[trial] = bool(is_verified)
        return trial_status

    def _on_trials_filtered(self, filtered_trials: list) -> None:
        """Handle TrialsWidget filter changes."""
        if not self.app_state.ready:
            return
        self.update_trials_combo()
        if self.app_state.trials_sel not in filtered_trials and filtered_trials:
            self.app_state.set_key_sel("trials", filtered_trials[0])
            self.app_state.trial_changed.emit()
        self.update_main_plot()

    # ------------------------------------------------------------------
    # Create controls (populates main panel groupboxes)
    # ------------------------------------------------------------------


    def _create_trial_controls(self):
        self.io_widget.create_device_controls(self.catalog)
        self.navigation_widget.setup_trial_conditions(self.catalog)
        self.navigation_widget.set_data_widget(self)

        if getattr(self, "trials_widget", None) is not None:
            self.trials_widget.setup(self.app_state.dt.metadata_df)

        for combo_name, combo_spec in self.catalog.combos.items():
            if not combo_spec.values:
                continue
            self._create_combo_widget(combo_name, list(combo_spec.values))

        # Restore camera combos
        has_nwb_pose = "nwb_source" in self.app_state.dt.attrs and (
            "position" in self.app_state.ds.data_vars
            or any(k.startswith("position_") for k in self.app_state.ds.data_vars)
        )
        cameras = self.app_state.dt.cameras
        has_remote_cameras = any(is_url(c) for c in cameras)
        if has_nwb_pose or has_remote_cameras:
            self.app_state.has_video = True
        slot_layout = self.slot_layout

        # Slot 1: Layers / Space Plot toggle
        self.space_view_combo = QComboBox()
        self.space_view_combo.setObjectName("space_view_combo")
        view_items = ["Layers", "Space Plot"]
        self.space_view_combo.addItems(view_items)
        self.space_view_combo.currentTextChanged.connect(self._on_space_view_changed)
        self.controls.append(self.space_view_combo)
        slot_layout.addWidget(self.space_view_combo)

        # Slot 2: Camera selection (first camera)
        if self.app_state.has_video and len(cameras) > 0:
            self.primary_camera_combo = QComboBox()
            self.primary_camera_combo.setObjectName("primary_camera_combo")
            self.primary_camera_combo.addItems([str(c) for c in cameras])
            self.primary_camera_combo.setItemDelegate(ElidedDelegate(parent=self.primary_camera_combo))
            self.primary_camera_combo.currentTextChanged.connect(self._on_primary_camera_changed)
            self.controls.append(self.primary_camera_combo)
            slot_layout.addWidget(self.primary_camera_combo)

        if len(cameras) > 1:
            from .video_manager import MAX_EXTRA_CAMERAS
            cam_names = [str(c) for c in cameras]
            n_extra = min(MAX_EXTRA_CAMERAS, len(cameras) - 1)
            self._extra_camera_combos: list[QComboBox] = []
            for i in range(n_extra):
                combo = QComboBox()
                combo.setObjectName(f"extra_camera_combo_{i}")
                combo.addItems(["None"] + cam_names)
                combo.setItemDelegate(ElidedDelegate(parent=combo))
                combo.setCurrentIndex(0)
                combo.currentTextChanged.connect(
                    lambda _text, idx=i: self._on_extra_camera_combo_changed(idx)
                )
                self._extra_camera_combos.append(combo)
                self.controls.append(combo)
                slot_layout.addWidget(combo)

        if "keypoints" in self.app_state.ds.coords:
            keypoint_names = strip_common_prefix([str(k) for k in self.app_state.ds.coords["keypoints"].values])
            self.populate_keypoints(keypoint_names)

        slot_layout.addStretch()
        if self.app_state.has_video or self.app_state.has_audio:
            self.slot_groupbox.show()

        self._setup_panel_checkboxes()

    def _setup_panel_checkboxes(self):
        self._audio_row_widgets = []

        has_audio = bool(self.app_state.has_audio)
        has_neo = bool(self.app_state.has_neo)
        has_phy = bool(self.app_state.has_neurons)
        has_features = bool(self.catalog and self.catalog.features)

        # Whether this panel's data is available in the current session.
        # When data is NOT available the checkbox is forced unchecked and signals are
        # blocked so the saved preference is preserved for future sessions with that data.
        data_available = {
            "video_viewer": bool(self.app_state.has_video),
            "pose_markers": bool(self.app_state.has_pose),
            "featureplot":  has_features,
            "audiotrace":   has_audio,
            "spectrogram":  has_audio,
            "neo_viewer":   has_neo,
            "phy_viewer":   has_phy,
        }
        # Saved user preference (ignored when data unavailable).
        # neo_viewer has no state_attr so always defaults to True when available.
        saved_checked = {
            "video_viewer": self.app_state.video_viewer_visible,
            "pose_markers": self.app_state.pose_markers_visible,
            "featureplot":  self.app_state.featureplot_visible,
            "audiotrace":   self.app_state.audiotrace_visible,
            "spectrogram":  self.app_state.spectrogram_visible,
            "neo_viewer":   True,
            "phy_viewer":   self.app_state.ephys_visible,
        }
        initial_shown = {
            "video_viewer": True,
            "pose_markers": True,
            "featureplot":  True,
            "audiotrace":   True,
            "spectrogram":  True,
            "neo_viewer":   has_neo,
            "phy_viewer":   has_phy,
        }

        for defn in _PANEL_DEFS:
            checkbox = QCheckBox(defn.label)
            checkbox.setObjectName(f"{defn.name}_checkbox")
            setattr(self, f"{defn.name}_checkbox", checkbox)

            # Set checkbox to saved state WITHOUT firing signals (avoids
            # triggering plot updates before data is loaded).
            checkbox.blockSignals(True)
            if data_available[defn.name]:
                checkbox.setChecked(saved_checked[defn.name])
            else:
                checkbox.setChecked(False)
            checkbox.blockSignals(False)

            # Apply the effective visibility to the container directly.
            effective = data_available[defn.name] and saved_checked[defn.name]
            if defn.container_method and self.plot_container:
                getattr(self.plot_container, defn.container_method)(effective)
            if defn.on_toggle:
                getattr(self, defn.on_toggle)(effective)

            # Connect signal AFTER initial state is applied.
            checkbox.stateChanged.connect(
                lambda state, n=defn.name: self._on_panel_toggled(n, state)
            )

            if not initial_shown[defn.name]:
                checkbox.hide()
            if defn.audio_row:
                self._audio_row_widgets.append(checkbox)

        # Row 1: audio panel checkboxes
        self.panels_row1_layout.addWidget(self.audiotrace_checkbox)
        self.panels_row1_layout.addWidget(self.spectrogram_checkbox)
        self.panels_row1_layout.addStretch()

        # Row 2: mic selector
        if has_audio:
            mic_names = self.catalog.mics if self.catalog else []
            expanded = self._expand_mics_with_channels(mic_names)
            self.mics_combo = QComboBox()
            self.mics_combo.setObjectName("mics_combo")
            self.mics_combo.addItems(expanded)
            self.mics_combo.currentTextChanged.connect(self._on_mics_changed)
            self.controls.append(self.mics_combo)
            self._mic_label = QLabel("Mic:")
            self.panels_row2_layout.addWidget(self._mic_label)
            self.panels_row2_layout.addWidget(self.mics_combo)
            self.panels_row2_layout.addStretch()
            self._audio_row_widgets.extend([self._mic_label, self.mics_combo])
            if expanded:
                self.app_state.set_key_sel("mics", expanded[0])

        # Row 3: feature panel checkbox + view controls
        self.panels_row3_layout.addWidget(self.featureplot_checkbox)
        self.panels_row3_layout.addWidget(QLabel("View:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.setObjectName("view_mode_combo")
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        self.view_mode_combo.hide()
        self.controls.append(self.view_mode_combo)
        self.panels_row3_layout.addWidget(self.view_mode_combo)
        self.sort_channels_btn = QPushButton("Sort channels")
        self.sort_channels_btn.setToolTip("Sort heatmap channels by activity in selected label interval")
        self.sort_channels_btn.setEnabled(False)
        self.sort_channels_btn.clicked.connect(self._on_sort_channels_clicked)
        self.panels_row3_layout.addWidget(self.sort_channels_btn)
        self.panels_row3_layout.addStretch()

        # Row 4: Neo-Viewer checkbox + Neo stream combo
        self.panels_row4_layout.addWidget(self.neo_viewer_checkbox)
        self._neo_stream_label = QLabel("Stream:")
        self.neo_stream_combo = QComboBox()
        self.neo_stream_combo.setObjectName("neo_stream_combo")
        self.neo_stream_combo.currentTextChanged.connect(self._on_neo_stream_changed)
        self.panels_row4_layout.addWidget(self._neo_stream_label)
        self.panels_row4_layout.addWidget(self.neo_stream_combo)
        self._neo_stream_label.hide()
        self.neo_stream_combo.hide()
        self.panels_row4_layout.addStretch()

        # Row 5: Phy-Viewer checkbox + neural view combo
        self.panels_row5_layout.addWidget(self.phy_viewer_checkbox)
        self._neural_view_label = QLabel("View:")
        self.neural_view_combo = QComboBox()
        self.neural_view_combo.setObjectName("neural_view_combo")
        self.neural_view_combo.addItems(["Multi Trace", "Raster"])
        self.neural_view_combo.currentTextChanged.connect(self._on_neural_view_changed)
        self.panels_row5_layout.addWidget(self._neural_view_label)
        self.panels_row5_layout.addWidget(self.neural_view_combo)
        self._neural_view_label.hide()
        self.neural_view_combo.hide()
        self.panels_row5_layout.addStretch()

        # slot_groupbox row 2: video viewer + pose markers
        self.slot_row2_layout.addWidget(self.video_viewer_checkbox)
        self.slot_row2_layout.addWidget(self.pose_markers_checkbox)
        self.slot_row2_layout.addStretch()

        if has_neo and self.ephys_widget:
            self._populate_neo_stream_combo()

        if has_phy and self.ephys_widget:
            self._neural_view_label.show()
            self.neural_view_combo.show()
            self.ephys_widget.configure_ephys_trace_plot()

        if not has_audio:
            for w in self._audio_row_widgets:
                w.hide()
        self.video_mgr.set_audio_row_widgets(self._audio_row_widgets)

        # Overlays
        overlays_layout = self.overlays_layout
        self.show_labels_checkbox = QCheckBox("Labels")
        self.show_labels_checkbox.setChecked(self.app_state.labels_visible)
        self.show_labels_checkbox.stateChanged.connect(self._on_labels_overlay_toggled)
        overlays_layout.addWidget(self.show_labels_checkbox)

        self.pred_show_predictions = QCheckBox("Predictions")
        self.pred_show_predictions.setChecked(False)
        self.pred_show_predictions.setEnabled(False)
        self.pred_show_predictions.stateChanged.connect(self._on_pred_show_predictions_changed)
        overlays_layout.addWidget(self.pred_show_predictions)

        self.show_confidence_checkbox = QCheckBox("Confidence")
        self.show_confidence_checkbox.setChecked(False)
        self.show_confidence_checkbox.stateChanged.connect(self._update_confidence_overlay)
        overlays_layout.addWidget(self.show_confidence_checkbox)

        self.show_envelope_checkbox = QCheckBox("Envelope")
        self.show_envelope_checkbox.setChecked(False)
        self.show_envelope_checkbox.stateChanged.connect(self._on_envelope_overlay_changed)
        self.show_envelope_checkbox.hide()
        overlays_layout.addWidget(self.show_envelope_checkbox)

        overlays_layout.addStretch()
        self._set_controls_enabled(False)

    # ------------------------------------------------------------------
    # Combo / checkbox handlers
    # ------------------------------------------------------------------

    def refresh_lineplot(self):
        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _update_confidence_overlay(self):
        if not self.app_state.ready or self.plot_container is None:
            return
        if not self.show_confidence_checkbox.isChecked():
            self.plot_container.hide_confidence_plot()
            return
        trial = self.app_state.trials_sel
        store = getattr(self.app_state, "pred_store", None)
        if store is not None:
            trial_confidence = store.get_confidence(trial, self.app_state.dt)
            if trial_confidence is not None:
                time_coord = self.app_state.time_coord.values
                n = min(len(trial_confidence), len(time_coord))
                self.plot_container.show_confidence_plot(trial_confidence[:n], time_coord[:n])
                return
        label_ds = getattr(self.app_state, "labels_confidence_ds", None)
        if label_ds is not None and "labels_confidence" in getattr(label_ds, "data_vars", {}):
            ds_kwargs = self.app_state.get_ds_kwargs()
            try:
                label_confidence, _ = eto.sel_valid(label_ds.labels_confidence, ds_kwargs)
            except (KeyError, AttributeError, ValueError):
                label_confidence = None
            if label_confidence is not None and len(label_confidence) > 0:
                self.plot_container.show_confidence_plot(label_confidence)
                return
        self.plot_container.hide_confidence_plot()

    def _on_pred_show_predictions_changed(self):
        self.refresh_lineplot()

    def cycle_neural_view(self):
        if not hasattr(self, 'neural_view_combo') or not self.neural_view_combo.isVisible():
            return
        next_index = (self.neural_view_combo.currentIndex() + 1) % self.neural_view_combo.count()
        self.neural_view_combo.setCurrentIndex(next_index)

    def _on_neural_view_changed(self, mode: str):
        if not self.app_state.ready or not mode:
            return
        if self.ephys_widget:
            self.ephys_widget.set_neural_view(mode)

    # ------------------------------------------------------------------
    # Neo-Viewer panel
    # ------------------------------------------------------------------

    def _populate_neo_stream_combo(self):
        """Populate the Neo stream combo with available streams, greying out
        any stream that matches kilosort params (n_channels, sample_rate)."""
        source_map = getattr(self.app_state, 'ephys_source_map', {})
        if not source_map:
            self._neo_stream_label.hide()
            self.neo_stream_combo.hide()
            return

        ks_params = None
        if self.ephys_widget:
            ks_params = getattr(self.ephys_widget, '_kilosort_params', None)

        self.neo_stream_combo.blockSignals(True)
        self.neo_stream_combo.clear()

        for display_name, (filepath, stream_id, _ch) in source_map.items():
            self.neo_stream_combo.addItem(display_name)

            # Grey out streams matching kilosort params
            if ks_params:
                loader = get_ephys_loader(filepath, stream_id)
                if loader is not None:
                    ks_sr = ks_params.get("sample_rate", 0)
                    ks_nch = ks_params.get("n_channels_dat", 0)
                    if (loader.n_channels == ks_nch
                            and abs(loader.rate - ks_sr) < 1.0):
                        idx = self.neo_stream_combo.count() - 1
                        model = self.neo_stream_combo.model()
                        item = model.item(idx)
                        if item:
                            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)

        self.neo_stream_combo.blockSignals(False)

        show_combo = self.neo_stream_combo.count() > 0
        self._neo_stream_label.setVisible(show_combo)
        self.neo_stream_combo.setVisible(show_combo)

        if show_combo:
            # Select first enabled item
            for i in range(self.neo_stream_combo.count()):
                item = self.neo_stream_combo.model().item(i)
                if item and (item.flags() & Qt.ItemIsEnabled):
                    self.neo_stream_combo.setCurrentIndex(i)
                    break

    def _on_neo_stream_changed(self, stream_name: str):
        if not self.app_state.ready or not stream_name:
            return
        self._configure_neo_panel(stream_name)

    def _configure_neo_panel(self, stream_name: str | None = None):
        """Configure the Neo-Viewer panel with the selected stream."""
        if not self.plot_container:
            return
        neo_cb = getattr(self, 'neo_viewer_checkbox', None)
        if neo_cb is not None and not neo_cb.isChecked():
            return

        if stream_name is None:
            stream_name = self.neo_stream_combo.currentText() if hasattr(self, 'neo_stream_combo') else ""
        if not stream_name:
            return

        source_map = getattr(self.app_state, 'ephys_source_map', {})
        if stream_name not in source_map:
            return

        filepath, stream_id, channel_idx = source_map[stream_name]
        loader = get_ephys_loader(filepath, stream_id)
        if loader is None:
            return

        if self.app_state.dt is not None:
            self.app_state.dt.attrs["ephys_stream_id"] = stream_id

        neo_plot = self.plot_container.neo_trace_plot
        neo_plot.set_loader(loader, channel_idx)

        neo_plot.set_source(FileSource("neo", loader, start_time=0.0))

        if self.plot_container._panel_visible["neo"]:
            xmin, xmax = self.plot_container.get_current_xlim()
            neo_plot.update_plot_content(xmin, xmax)
            neo_plot.auto_channel_spacing()
            neo_plot.auto_gain()
            neo_plot.autoscale()

    def cycle_view_mode(self):
        if not hasattr(self, 'view_mode_combo') or not self.view_mode_combo.isVisible():
            return
        next_index = (self.view_mode_combo.currentIndex() + 1) % self.view_mode_combo.count()
        self.view_mode_combo.setCurrentIndex(next_index)

    def _on_mics_changed(self, mic_name):
        if not self.app_state.ready or not mic_name:
            return
        self.app_state.set_key_sel("mics", mic_name)
        self.update_audio()
        self.plot_container.clear_audio_cache()
        self.plot_container.update_audio_panels()
        current_plot = self.plot_container.get_current_plot()
        xmin, xmax = current_plot.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _get_audio_channel_count(self, audio_path):
        try:
            from audioio import AudioLoader
            with AudioLoader(audio_path) as loader:
                if loader.shape is None:
                    return 1
                return loader.channels if hasattr(loader, 'channels') else (loader.shape[1] if len(loader.shape) > 1 else 1)
        except (ImportError, OSError, ValueError):
            return 1

    def _expand_mics_with_channels(self, mic_labels):
        self.app_state.audio_source_map.clear()
        expanded_items = []
        audio_folder = self.app_state.audio_folder
        dt = getattr(self.app_state, 'dt', None)
        trial_id = getattr(self.app_state, 'trials_sel', None)
        if trial_id is None and dt is not None and dt.trials:
            trial_id = dt.trials[0]

        if not audio_folder or dt is None:
            for mic in mic_labels:
                display_name = str(mic)
                self.app_state.audio_source_map[display_name] = (str(mic), 0)
                expanded_items.append(display_name)
            return expanded_items

        for mic_label in mic_labels:
            mic_file = dt.get_media(trial_id, "audio", str(mic_label))
            if not mic_file:
                continue
            try:
                audio_path = os.path.join(audio_folder, mic_file)
                n_channels = self._get_audio_channel_count(audio_path)
                if n_channels > 1:
                    for ch in range(n_channels):
                        display_name = f"{mic_file} (Ch {ch + 1})"
                        self.app_state.audio_source_map[display_name] = (mic_file, ch)
                        expanded_items.append(display_name)
                else:
                    self.app_state.audio_source_map[mic_file] = (mic_file, 0)
                    expanded_items.append(mic_file)
            except (OSError, ValueError):
                self.app_state.audio_source_map[mic_file] = (mic_file, 0)
                expanded_items.append(mic_file)
        return expanded_items

    def update_mics_combo_for_trial(self, ds):
        combo = getattr(self, 'mics_combo', None)
        if combo is None:
            return
        new_items = self.app_state.dt.mics
        if not new_items:
            return
        new_items = np.array(new_items, dtype=str)
        prev_index = combo.currentIndex()
        combo.blockSignals(True)
        combo.clear()
        expanded = self._expand_mics_with_channels(new_items)
        combo.addItems(expanded)
        if prev_index < combo.count():
            combo.setCurrentIndex(prev_index)
        else:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)
        self.app_state.set_key_sel("mics", combo.currentText())

    def _update_device_sels_for_trial(self, ds):
        cameras = self.app_state.dt.cameras

        primary = getattr(self, 'primary_camera_combo', None)
        if primary is not None and cameras:
            prev_index = primary.currentIndex()
            primary.blockSignals(True)
            primary.clear()
            primary.addItems(cameras)
            if prev_index < primary.count():
                primary.setCurrentIndex(prev_index)
            else:
                primary.setCurrentIndex(0)
            primary.blockSignals(False)
            self.app_state.primary_camera = primary.currentText()

        self._update_extra_camera_combo_items(cameras)



    def _on_labels_overlay_toggled(self, state):
        visible = state == Qt.Checked
        self.app_state.labels_visible = visible
        if self.plot_container:
            if visible:
                ds_kwargs = self.app_state.get_ds_kwargs()
                self.update_label_plot(ds_kwargs)
            else:
                for plot in self.plot_container._get_all_plots():
                    self.plot_container._clear_labels_on_plot(plot)

    def _is_autoscale_on(self) -> bool:
        return (
            self.plot_settings_widget is not None
            and self.plot_settings_widget.autoscale_checkbox.isChecked()
        )

    def _on_panel_toggled(self, name: str, state: int):
        """Central handler for all panel visibility checkboxes."""
        visible = state == Qt.Checked
        defn = next(d for d in _PANEL_DEFS if d.name == name)

        if defn.state_attr:
            setattr(self.app_state, defn.state_attr, visible)

        if defn.container_method and self.plot_container:
            getattr(self.plot_container, defn.container_method)(visible)

        if visible and defn.autoscale_plot and self._is_autoscale_on() and self.plot_container:
            plot = getattr(self.plot_container, defn.autoscale_plot)
            plot.vb.enableAutoRange(x=False, y=True)
            if hasattr(plot, '_apply_y_constraints'):
                plot._apply_y_constraints()

        if defn.on_toggle:
            getattr(self, defn.on_toggle)(visible)

    def _on_neo_panel_toggle(self, visible: bool):
        if visible and self.plot_container:
            self._configure_neo_panel()

    def _on_phy_panel_toggle(self, visible: bool):
        if not visible or not self.plot_container:
            return
        mode = self.neural_view_combo.currentText() if hasattr(self, 'neural_view_combo') else "Multi Trace"
        self.plot_container.set_neural_panel_mode("raster" if mode == "Raster" else "trace")
        if self._is_autoscale_on():
            self.plot_container.ephys_trace_plot.vb.enableAutoRange(x=False, y=True)

    def _on_video_viewer_toggle(self, visible: bool):
        if hasattr(self, 'layout_mgr') and self.layout_mgr:
            self.layout_mgr.set_video_viewer_visible(visible)

    def _on_pose_markers_toggle(self, visible: bool):
        if visible:
            self.update_pose()
        elif self.pose_mgr is not None:
            self.pose_mgr._remove_pose_layers()

    def _on_ephys_toggled(self, state):
        self._on_panel_toggled("phy_viewer", state)

    def _update_view_mode_items(self, feature_sel: str):
        """Update view_mode_combo items based on available data.

        Feature view controls what the bottom (feature) panel shows.
        Audio/Ephys heatmap modes compute envelope from raw data.
        """
        current_text = self.view_mode_combo.currentText()
        self.view_mode_combo.blockSignals(True)
        self.view_mode_combo.clear()

        has_audio = self.app_state.has_audio or bool(self.app_state.audio_path)
        items = ["LinePlot", "Heatmap"]
        if has_audio:
            items.append("Heatmap (Audio)")
        if self.app_state.has_neo:
            items.append("Heatmap (Ephys)")
        self.view_mode_combo.addItems(items)

        idx = self.view_mode_combo.findText(current_text)
        if idx >= 0:
            self.view_mode_combo.setCurrentIndex(idx)
        self.view_mode_combo.blockSignals(False)

    def _on_view_mode_changed(self, mode: str):
        if not self.app_state.ready or not self.plot_container:
            return

        self.app_state.feature_view_mode = mode

        if mode.startswith("Heatmap"):
            self.plot_container.switch_to_heatmap()
            self.plot_container.heatmap_plot._clear_buffer()
            self.plot_container.heatmap_plot._channel_range = None
        else:
            self.plot_container.switch_to_lineplot()

        self._update_sort_button_state()

        xmin, xmax = self.plot_container.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)

    def _update_sort_button_state(self):
        btn = getattr(self, 'sort_channels_btn', None)
        if btn is None:
            return
        enabled = self.plot_container is not None and self.plot_container.is_heatmap()
        btn.setEnabled(enabled)

    def _on_sort_channels_clicked(self):
        if not self.labels_widget or not self.plot_container:
            return

        idx = self.labels_widget.current_labels_pos
        if idx is None:
            notify("Select a label interval first", "warning")
            return

        df = self.app_state.label_intervals
        if df is None or idx not in df.index:
            notify("Select a label interval first", "warning")
            return

        onset_s, offset_s, _ = get_interval_bounds(df, idx)

        heatmap = self.plot_container.heatmap_plot
        data = heatmap.get_normalized_data_for_range(onset_s, offset_s)
        if data is None or data.size == 0:
            notify("No heatmap data available for the selected interval", "warning")
            return

        channel_sums = np.nansum(np.abs(data), axis=0)
        sort_order = np.argsort(channel_sums)[::-1]
        heatmap.set_sort_order(sort_order)

    def _configure_ephys_trace_plot(self):
        if self.ephys_widget:
            self.ephys_widget.configure_ephys_trace_plot()

    def _hide_ephys_channel_controls(self):
        if self.ephys_widget:
            self.ephys_widget.hide_ephys_channel_controls()

    def _apply_view_mode_for_feature(self):
        mode = self.view_mode_combo.currentText()
        if mode.startswith("Heatmap"):
            self.plot_container.switch_to_heatmap()
        else:
            self.plot_container.switch_to_lineplot()


    def _on_envelope_overlay_changed(self):
        if not self.plot_container:
            return
        if self.show_envelope_checkbox.isChecked():
            self.plot_container.show_envelope_overlay()
        else:
            self.plot_container.hide_envelope_overlay()

    def _open_energy_params(self):
        from .dialog_function_params import open_function_params_dialog
        key = self.data_panel.energy_display_to_key(
            self.data_panel.metric_combo.currentText(),
        )
        if key:
            result = open_function_params_dialog(key, self.app_state, parent=self.data_panel)
            if result is not None:
                self._on_energy_apply()

    def _on_energy_apply(self):
        metric_key = self.data_panel.energy_display_to_key(
            self.data_panel.metric_combo.currentText(),
        )
        self.app_state.energy_metric = metric_key

        if not self.plot_container:
            return

        from .dialog_busy_progress import BusyProgressDialog

        if (
            hasattr(self, 'show_envelope_checkbox')
            and not self.show_envelope_checkbox.isChecked()
        ):
            self.show_envelope_checkbox.setChecked(True)
            return

        self.plot_container.hide_envelope_overlay()
        dialog = BusyProgressDialog("Computing energy envelope...", parent=self.data_panel)
        dialog.execute_blocking(self.plot_container.show_envelope_overlay)

    def _set_controls_enabled(self, enabled: bool):
        for control in self.controls:
            control.setEnabled(enabled)
        self.io_widget.set_controls_enabled(enabled)
        self.app_state.ready = enabled

    def _create_combo_widget(self, key, vars):
        excluded_from_all = {"individuals", "features", "colors", "cameras", "mics"}
        show_all_checkbox = key not in excluded_from_all

        combo = QComboBox()
        combo.setObjectName(f"{key}_combo")
        combo.currentIndexChanged.connect(self._on_combo_changed)
        if key == "colors":
            raw_items = ["None"] + [str(var) for var in vars]
        else:
            raw_items = [str(var) for var in vars]
        display_items = clean_display_labels(raw_items)
        for display, raw in zip(display_items, raw_items):
            combo.addItem(display, raw)

        make_searchable(combo)

        target_layout = self.coords_groupbox_layout

        if show_all_checkbox:
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)

            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row_layout.addWidget(combo)

            all_checkbox = QCheckBox("All")
            all_checkbox.setObjectName(f"{key}_all_checkbox")
            all_checkbox.setToolTip(f"Show all {key} traces on the plot")
            all_checkbox.stateChanged.connect(lambda state, k=key: self._on_all_checkbox_changed(k, state))
            row_layout.addWidget(all_checkbox)

            self.all_checkboxes[key] = all_checkbox
            self.controls.append(all_checkbox)

            target_layout.addRow(f"{key.capitalize()}:", row_widget)
        else:
            target_layout.addRow(f"{key.capitalize()}:", combo)

        self.combos[key] = combo
        self.controls.append(combo)

        return combo

    def _on_combo_changed(self):
        if not self.app_state.ready:
            return

        combo = self.sender()
        name = combo.objectName()
        key = name[:-6] if name.endswith("_combo") else None

        if key:
            selected_value = get_combo_value(combo)
            self.app_state.set_key_sel(key, selected_value)

            if key == "features":
                self._update_view_mode_items(selected_value)
                self.view_mode_combo.show()
                self._apply_view_mode_for_feature()

            
            current_plot = self.plot_container.get_current_plot()
            xmin, xmax = current_plot.get_current_xlim()
            self.update_main_plot(t0=xmin, t1=xmax)

            if key in ["individuals", "keypoints", "colors"]:
                if self.space_plot and self.space_plot.isVisible():
                    self.space_plot.refresh()

            if key == "cluster_id" and self.ephys_widget:
                try:
                    self.ephys_widget.select_cluster_in_table(int(selected_value))
                except (ValueError, TypeError):
                    pass

            if key == "individuals":
                self.labels_widget.refresh_labels_shapes_layer()


    def _on_all_checkbox_changed(self, key: str, state: int):
        if not self.app_state.ready:
            return

        combo = self.combos.get(key)
        if combo is None:
            return

        is_checked = state == Qt.Checked

        if is_checked:
            for other_key, other_checkbox in self.all_checkboxes.items():
                if other_key != key and other_checkbox.isChecked():
                    other_checkbox.blockSignals(True)
                    other_checkbox.setChecked(False)
                    other_checkbox.blockSignals(False)
                    other_combo = self.combos.get(other_key)
                    if other_combo:
                        other_combo.setEnabled(True)
                        self.app_state.set_key_sel(other_key, get_combo_value(other_combo))
                    self._update_all_checkbox_state(other_key, False)

        combo.setEnabled(not is_checked)
        self._update_all_checkbox_state(key, is_checked)

        if is_checked:
            self.app_state.set_key_sel(key, None)
        else:
            self.app_state.set_key_sel(key, get_combo_value(combo))

        current_plot = self.plot_container.get_current_plot()
        xmin, xmax = current_plot.get_current_xlim()
        self.update_main_plot(t0=xmin, t1=xmax)
        if self.space_plot and self.space_plot.isVisible():
            self.space_plot.refresh()

    def _on_channel_all_changed(self, state: int):
        if not self.app_state.ready:
            return
        is_checked = state == Qt.Checked
        for key, checkbox in self.all_checkboxes.items():
            if checkbox.isChecked() != is_checked:
                checkbox.setChecked(is_checked)

    def _update_all_checkbox_state(self, key: str, is_checked: bool):
        states = self.app_state.all_checkbox_states.copy()
        if is_checked:
            states[key] = True
        else:
            states.pop(key, None)
        self.app_state.all_checkbox_states = states

    def _restore_or_set_defaults(self):
        if not self.catalog:
            return
        for key, spec in self.catalog.combos.items():
            combo = self.io_widget.combos.get(key) or self.combos.get(key)
            vals = list(spec.values)

            if combo is not None and vals:
                saved_value = self.app_state.get_key_sel(key) if self.app_state.key_sel_exists(key) else None
                vals_str = [str(v) for v in vals]

                if saved_value in vals_str:
                    set_combo_to_value(combo, saved_value)
                elif saved_value and key == "mics":
                    match = next((v for v in vals_str if v.startswith(str(saved_value))), None)
                    if match:
                        set_combo_to_value(combo, match)
                        self.app_state.set_key_sel(key, match)
                    else:
                        set_combo_to_value(combo, vals_str[0])
                        self.app_state.set_key_sel(key, vals_str[0])
                else:
                    if key == "features" and "speed" in vals_str:
                        set_combo_to_value(combo, "speed")
                        self.app_state.set_key_sel(key, "speed")
                    else:
                        set_combo_to_value(combo, vals_str[0])
                        self.app_state.set_key_sel(key, vals_str[0])


        if self.app_state.key_sel_exists("trials"):
            saved_trial = self.app_state.get_key_sel("trials")
            self.app_state.set_key_sel("trials", saved_trial)
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials_sel))
        else:
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials[0]))
            self.app_state.trials_sel = self.app_state.trials[0]

        space_plot_type = getattr(self.app_state, 'space_plot_type', 'Layers')
        # Migrate old values to simplified combo
        if space_plot_type in ("Space 2D", "Space 3D", "space_2D", "space_3D", "PCA 2D", "PCA 3D"):
            space_plot_type = "Space Plot"
        if hasattr(self, 'space_view_combo'):
            self.space_view_combo.setCurrentText(space_plot_type)

        saved_camera = self.app_state.primary_camera
        if saved_camera and hasattr(self, 'primary_camera_combo'):
            idx = self.primary_camera_combo.findText(saved_camera)
            if idx >= 0:
                self.primary_camera_combo.setCurrentIndex(idx)

        saved_extra = self.app_state.extra_cameras
        if saved_extra and hasattr(self, '_extra_camera_combos'):
            for i, cam_name in enumerate(saved_extra):
                if i >= len(self._extra_camera_combos):
                    break
                combo = self._extra_camera_combos[i]
                idx = combo.findText(cam_name)
                if idx >= 0:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)

        # Normalize stale *_sel values against currently loaded options.
        self._normalize_saved_sel_values()

        checkbox_states = self.app_state.all_checkbox_states or {}
        for key, is_checked in checkbox_states.items():
            checkbox = self.all_checkboxes.get(key)
            combo = self.combos.get(key)
            if checkbox and is_checked:
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
                if combo:
                    combo.setEnabled(False)
                self.app_state.set_key_sel(key, None)

    def _normalize_saved_sel_values(self) -> None:
        def _normalize_from_combo(key: str, combo: QComboBox) -> None:
            if combo is None or combo.count() == 0:
                return
            saved_value = self.app_state.get_key_sel(key) if self.app_state.key_sel_exists(key) else None
            if saved_value is None:
                self.app_state.set_key_sel(key, get_combo_value(combo))
                return
            if find_combo_index(combo, str(saved_value)) >= 0:
                return
            combo.setCurrentIndex(0)
            fallback = get_combo_value(combo)
            logger.warning("Saved %s_sel '%s' not found; reverting to '%s'", key, saved_value, fallback)
            self.app_state.set_key_sel(key, fallback)

        for key, combo in self.combos.items():
            _normalize_from_combo(key, combo)
        for key, combo in self.io_widget.combos.items():
            _normalize_from_combo(key, combo)

        primary_combo = getattr(self, "primary_camera_combo", None)
        if isinstance(primary_combo, QComboBox) and primary_combo.count() > 0:
            saved = self.app_state.primary_camera
            if saved is None or find_combo_index(primary_combo, saved) < 0:
                self.app_state.primary_camera = get_combo_value(primary_combo)

        mics_combo = getattr(self, "mics_combo", None)
        if isinstance(mics_combo, QComboBox):
            _normalize_from_combo("mics", mics_combo)



    # ------------------------------------------------------------------
    # Trial change
    # ------------------------------------------------------------------

    def _load_trial_with_fallback(self) -> None:
        first_trial = self.app_state.trials[0]
        current_trial = getattr(self.app_state, 'trials_sel', None)

        try:
            is_nan = np.isnan(current_trial)
        except (TypeError, ValueError):
            is_nan = False

        if not current_trial or is_nan or current_trial not in self.app_state.trials:
            if current_trial and not is_nan:
                logger.warning("Saved trial %s not in dataset, using %s", current_trial, first_trial)
            self.app_state.trials_sel = first_trial

        self.on_trial_changed()

    def _validate_media_files(self) -> list[str]:
        missing = []
        dt = self.app_state.dt
        first_trial = self.app_state.trials[0]

        video_folder = self.app_state.video_folder
        if video_folder:
            for cam in dt.cameras:
                vid = dt.get_media(first_trial, "video", device=cam) 
                if not vid or is_url(vid):
                    continue
                path = os.path.join(video_folder, vid)
                if not os.path.isfile(path):
                    missing.append(f"Video: {path}")

        audio_folder = self.app_state.audio_folder
        if audio_folder:
            mics = dt.mics
            if not mics:
                notify(
                    "You selected an audio folder, although the .nc "
                    "contains no audio media entries.",
                    "warning",
                )
            else:
                for mic in mics:
                    aud = dt.get_media(first_trial, "audio", device=mic)
                    if not aud:
                        continue
                    path = os.path.join(audio_folder, aud)
                    if not os.path.isfile(path):
                        missing.append(f"Audio: {path}")

        pose_folder = self.app_state.pose_folder
        if pose_folder:
            cameras = dt.cameras
            if not cameras:
                notify(
                    "You selected a pose folder, although the .nc "
                    "contains no pose data.",
                    "warning",
                )
            else:
                for cam in cameras:
                    pose_file = dt.get_media(first_trial, "pose", device=cam)
                    if not pose_file:
                        continue
                    path = os.path.join(pose_folder, pose_file)
                    if not os.path.isfile(path):
                        missing.append(f"Pose: {path}")

        return missing



    def _build_trial_alignment(self, trial_id) -> None:
        self.app_state.trial_alignment = compute_trial_alignment(
            self.app_state.dt,
            trial_id,
            self.app_state.ds,
            video_folder=self.app_state.video_folder,
            audio_folder=self.app_state.audio_folder,
            cameras_sel=self.app_state.primary_camera,
        )

    def _build_restrict_window(self, trial_id) -> None:
        from ethograph.io.restrict import build_trial_window
        alignment = self.app_state.trial_alignment
        if alignment is None:
            return
        extra_t0 = getattr(self.app_state, "restrict_extra_t0", 0.0)
        extra_t1 = getattr(self.app_state, "restrict_extra_t1", 0.0)
        if self.app_state.restrict_mode == "trial":
            self.app_state.restrict_window = build_trial_window(
                alignment, trial_id, extra_t0, extra_t1,
            )

    def on_trial_changed(self):
        trials_sel = self.app_state.trials_sel
        
        if trials_sel not in self.app_state.trials:
            logger.warning("Selected trial '%s' not found in dataset, reverting to first trial '%s'", trials_sel, self.app_state.trials[0])
            trials_sel = self.app_state.trials[0]
            self.app_state.trials_sel = trials_sel
            return
        

        self.app_state.ds = self.app_state.dt.trial(trials_sel)

        # Update data_loader trial index (pynapple: restrict changes;
        # xarray: XarrayLoader.update_ds swaps the backing dataset)
        store = self.app_state.data_loader
        if store is not None:
            trial_idx = self.app_state.trials.index(trials_sel)
            store.set_trial(trial_idx)
            if hasattr(store, 'update_ds'):
                store.update_ds(self.app_state.ds)

        self._update_device_sels_for_trial(self.app_state.ds)
        self.update_mics_combo_for_trial(self.app_state.ds)

        features_combo = self.combos.get("features")
        fallback_feature = features_combo.itemText(0) if features_combo and features_combo.count() else None
        feature_sel = getattr(self.app_state, 'features_sel', fallback_feature)

        if hasattr(self, 'view_mode_combo'):
            self._update_view_mode_items(feature_sel)

        if hasattr(self, 'view_mode_combo'):
            self._apply_view_mode_for_feature()
            view_mode = self.view_mode_combo.currentText()
            if view_mode.startswith("Heatmap"):
                self.plot_container.heatmap_plot._clear_buffer()

        self.app_state.label_intervals = self.app_state.get_trial_intervals(trials_sel)

        self._build_trial_alignment(trials_sel)
        self._build_restrict_window(trials_sel)

        self.app_state.current_frame = 0
        self.update_video()
        self._init_or_update_extra_cameras()
        self.update_audio()
        self.update_pose()
        self.update_label()
        if self.ephys_widget:
            self.ephys_widget.on_trial_changed()
        preserve = getattr(self.app_state, '_preserve_x_range_next', False)
        self.app_state._preserve_x_range_next = False
        self.update_main_plot(preserve_x_range=preserve)
        try:
            self.update_space_plot()
        except Exception:
            logger.debug("update_space_plot failed", exc_info=True)

        self.plot_container.update_time_range_from_data()
        self.plot_container.update_time_marker_by_time(0.0)

        self._update_confidence_overlay()

        if self.io_widget:
            self.io_widget._update_human_verified_status()

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def update_main_plot(self, **kwargs):
        if not self.app_state.ready:
            return

        ds_kwargs = self.app_state.get_ds_kwargs()
        current_plot = self.plot_container.get_current_plot()

        self.plot_container.clear_amplitude_envelope()

        current_plot.update_plot(**kwargs)


        if self.show_envelope_checkbox.isChecked():
            self.plot_container.show_envelope_overlay()

        self.update_label_plot(ds_kwargs)

    def update_label_plot(self, ds_kwargs):
        if not self.app_state.labels_visible:
            if self.plot_container:
                for plot in self.plot_container._get_all_plots():
                    self.plot_container._clear_labels_on_plot(plot)
            return

        intervals_df = self.app_state.label_intervals

        if intervals_df is not None and not intervals_df.empty and "individuals" in ds_kwargs:
            selected_ind = str(ds_kwargs["individuals"])
            intervals_df = intervals_df[intervals_df["individual"] == selected_ind]

        predictions_df = None

        if (
            self.pred_show_predictions.isChecked()
            and self.app_state.pred_labels_df is not None
        ):
            trial = self.app_state.trials_sel
            df = self.app_state.pred_labels_df
            predictions_df = df[df["trial"] == trial] if "trial" in df.columns else df

        self.labels_widget.plot_all_labels(intervals_df, predictions_df)

    # ------------------------------------------------------------------
    # Video / audio / pose / space
    # ------------------------------------------------------------------

    def update_video(self):
        if not self.app_state.ready:
            return
        self.show_envelope_checkbox.show()
        self.video_mgr.update_video(plot_container=self.plot_container)

    def update_audio(self):
        if not self.app_state.ready:
            return
        self.video_mgr.update_audio(plot_container=self.plot_container)


    def update_label(self):
        self.labels_widget.refresh_labels_shapes_layer()

    def toggle_pause_resume(self):
        self.video_mgr.toggle_pause_resume(self.plot_container)

    def _on_time_marker_updated(self, time_s: float):
        if not self.space_plot or not self.space_plot.isVisible():
            return
        self.space_plot.update_time_marker(time_s)
        self._highlight_label_at_time(time_s)

    def _on_xrange_for_space_plot(self, _time_s: float):
        """Debounced re-render of space plot when lineplot x-range changes."""
        if self.space_plot and self.space_plot.isVisible():
            self.space_plot.on_xrange_changed()

    _space_highlight_key: tuple | None = None

    def _highlight_label_at_time(self, time_s: float):
        """If the current time falls inside a label, highlight that segment.

        Only redraws when entering a different label interval.
        """
        label_intervals = self.app_state.label_intervals
        if label_intervals is None or label_intervals.empty:
            self._space_highlight_key = None
            return
        mask = (label_intervals["onset_s"] <= time_s) & (label_intervals["offset_s"] >= time_s)
        hits = label_intervals[mask]
        if hits.empty:
            self._space_highlight_key = None
            return
        row = hits.iloc[0]
        key = (float(row["onset_s"]), float(row["offset_s"]), int(row["labels"]))
        if key == self._space_highlight_key:
            return
        self._space_highlight_key = key

        color = (255, 102, 0)
        mappings = getattr(self.labels_widget, '_mappings', {})
        color = mappings.get(key[2], {}).get("color", color)
        self.space_plot.highlight_time_segment(key[0], key[1], color)

    def _on_primary_frame_changed(self, frame_number: int):
        self.plot_container.update_time_marker_and_window(frame_number)

        video = getattr(self.app_state, 'video', None)
        if video:
            current_time = video.frame_to_time(frame_number)
        else:
            current_time = frame_number / self.app_state.video_fps

        xlim = self.plot_container.get_current_xlim()
        if getattr(self.app_state, 'center_playback', False) or current_time < xlim[0] or current_time > xlim[1]:
            self.plot_container.set_x_range(mode='center', center_on_frame=frame_number)

    def update_pose(self):
        """Refresh primary and extra camera pose layers through PoseDisplayManager."""
        if self.pose_mgr is None or not self.app_state.has_pose:
            return
        if not self.app_state.pose_markers_visible:
            return
        if not hasattr(self.app_state, 'trials_sel') or self.app_state.trials_sel is None:
            return
        self.pose_mgr.update_pose(self.get_hidden_keypoints())


    def closeEvent(self, event):
        SharedAudioCache.clear_cache()
        from .plots_ephystrace import clear_loader_cache
        clear_loader_cache()
        if getattr(self.app_state, 'video', None):
            self.app_state.video.stop()
        super().closeEvent(event)

    def _on_space_view_changed(self, text):
        if not self.app_state.ready:
            return
        self.app_state.space_plot_type = text

        show_layers = text == "Layers"
        self.layout_mgr.toggle_layer_docks_with_anchor(show_layers)

        if text == "Space Plot":
            self.update_space_plot()
        elif self.space_plot:
            self.space_plot.hide()

    def _on_primary_camera_changed(self, camera_name):
        if not self.app_state.ready or not camera_name:
            return
        self.app_state.primary_camera_previous = self.app_state.primary_camera
        self.app_state.primary_camera = camera_name
        self.update_video()
        self.update_pose()

    def _on_extra_camera_combo_changed(self, combo_idx: int):
        if not self.app_state.ready:
            return
        self._apply_extra_cameras()
        self._save_extra_cameras()

    def _apply_extra_cameras(self):
        desired = self._get_desired_extra_cameras()
        current = set(self.video_mgr.extra_widgets.keys())

        for name in current - desired:
            self.video_mgr.remove_camera(name)
            if self.pose_mgr is not None:
                self.pose_mgr.on_camera_removed(name)

        to_add: dict[str, str] = {}
        for name in desired - current:
            video_path = self.video_mgr._resolve_video_path(name, self.app_state.video_folder)
            if video_path:
                to_add[name] = video_path

        from .video_manager import VideoManager
        readers = VideoManager.open_readers_parallel(to_add)

        for name, video_path in to_add.items():
            self.video_mgr.add_camera(
                camera_name=name,
                video_path=video_path,
                layout_mgr=self.layout_mgr,
                meta_widget=self.meta_widget,
                reader=readers.get(name),
            )
            if self.pose_mgr is not None:
                self.pose_mgr.update_extra_camera_pose(name, self.get_hidden_keypoints())

    def _get_desired_extra_cameras(self) -> set[str]:
        if not hasattr(self, '_extra_camera_combos'):
            return set()
        names = set()
        for combo in self._extra_camera_combos:
            text = combo.currentText()
            if text and text != "None":
                names.add(text)
        return names

    def _save_extra_cameras(self):
        if not hasattr(self, '_extra_camera_combos'):
            return
        values = [combo.currentText() for combo in self._extra_camera_combos]
        self.app_state.extra_cameras = [v for v in values if v and v != "None"]

    def _update_extra_camera_combo_items(self, cameras: list[str]):
        if not hasattr(self, '_extra_camera_combos'):
            return
        cam_names = [str(c) for c in cameras]
        for combo in self._extra_camera_combos:
            prev_text = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(["None"] + cam_names)
            idx = combo.findText(prev_text)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

    def _init_or_update_extra_cameras(self):
        if not hasattr(self, '_extra_camera_combos'):
            return
        desired = self._get_desired_extra_cameras()
        to_add: dict[str, str] = {}
        for camera_name in desired:
            widget = self.video_mgr.extra_widgets.get(camera_name)
            if widget is not None and widget.isVisible():
                continue
            video_path = self.video_mgr._resolve_video_path(camera_name, self.app_state.video_folder)
            if video_path:
                to_add[camera_name] = video_path

        from .video_manager import VideoManager
        readers = VideoManager.open_readers_parallel(to_add)

        for camera_name, video_path in to_add.items():
            self.video_mgr.add_camera(
                camera_name=camera_name,
                video_path=video_path,
                layout_mgr=self.layout_mgr,
                meta_widget=self.meta_widget,
                reader=readers.get(camera_name),
            )
            if self.pose_mgr is not None:
                self.pose_mgr.update_extra_camera_pose(camera_name, self.get_hidden_keypoints())



    def update_space_plot(self):
        if not self.app_state.ready:
            return

        plot_type = self.app_state.get_with_default('space_plot_type')
        if plot_type != "Space Plot":
            if self.space_plot:
                self.space_plot.hide()
            return

        if not self.space_plot:
            self.space_plot = SpacePlot(self.viewer, self.app_state)
            self.space_plot.set_plot_container(self.plot_container)
            self.plot_container.time_marker_updated.connect(self._on_xrange_for_space_plot)
            if self.labels_widget:
                self.labels_widget.highlight_spaceplot.connect(self._highlight_positions_in_space_plot)

        store = getattr(self.app_state, 'data_loader', None)
        if store is None and self.app_state.ds is not None:
            from ethograph.io.catalog import XarrayLoader, catalog_from_xarray
            cat = catalog_from_xarray(self.app_state.ds, self.app_state.dt)
            store = XarrayLoader(self.app_state.ds, cat)

        self.space_plot.set_store(store)
        self.space_plot.refresh()
        self.space_plot.show()

    def _highlight_positions_in_space_plot(self, start_time: float, end_time: float):
        if not self.space_plot or not self.space_plot.dock_widget or not self.space_plot.dock_widget.isVisible():
            return

        color = (255, 102, 0)
        label_intervals = self.app_state.label_intervals
        active_ids = self.app_state.active_label_ids
        if label_intervals is not None and not label_intervals.empty:
            mid = (start_time + end_time) / 2.0
            mask = (label_intervals["onset_s"] <= mid) & (label_intervals["offset_s"] >= mid)
            hits = label_intervals[mask]
            if not hits.empty:
                label_id = int(hits.iloc[0]["labels"])
                if active_ids is None or label_id in active_ids:
                    mappings = getattr(self.labels_widget, '_mappings', {})
                    color = mappings.get(label_id, {}).get("color", color)

        self.space_plot.highlight_time_segment(start_time, end_time, color)
