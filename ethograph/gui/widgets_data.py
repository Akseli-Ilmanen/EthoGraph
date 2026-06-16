"""Widget for selecting start/stop times and playing a segment in napari."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
from napari.viewer import Viewer
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
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
from ethograph.io.data_loader import load_dataset
from ethograph.io.plot_sources import FileSource
from ethograph.io.time_model import compute_trial_video_bounds
from ethograph.labels.intervals import get_interval_bounds
from ethograph.utils.qt import (
    ElidedDelegate,
    find_combo_index,
    get_combo_value,
    make_searchable,
    set_combo_to_value,
)

from ..io.ephys_loader import load_ephys
from .app_constants import (
    DEFAULT_LAYOUT_MARGIN,
    DEFAULT_LAYOUT_SPACING,
    SIDEBAR_AFTER_LOAD_WIDTH_RATIO,
)
from .make_pretty import clean_display_labels
from .plots_space import SpacePlot
from .plots_spectrogram import SharedAudioCache
from .pose_render import (
    PoseDisplayManager,
    strip_common_prefix,
)
from .video_manager import VideoManager, is_url

logger = logging.getLogger(__name__)


def _detect_nwb_pose_keys(nwb_path: str | None) -> list[str] | None:
    """Detect PoseEstimation container names from NWB processing modules."""
    if not nwb_path or not Path(nwb_path).exists():
        return None
    try:
        from pynwb import NWBHDF5IO

        keys = []
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            for mod in nwb.processing.values():
                for name, di in mod.data_interfaces.items():
                    if hasattr(di, "pose_estimation_series"):
                        keys.append(name)
        return keys if keys else None
    except Exception:
        return None


@dataclass
class _PanelDef:
    """Declarative description of a panel toggle checkbox."""

    name: str  # internal identifier / checkbox attr prefix
    label: str  # displayed checkbox text
    row: int  # UI row in panels_groupbox (1, 2, or 3)
    state_attr: str | None = None  # app_state attribute to sync with visibility
    container_method: str | None = None  # plot_container.method(visible) to call
    autoscale_plot: str | None = None  # plot_container.X for autoscale on show
    audio_row: bool = False  # part of the hidden-when-no-audio widget group
    on_toggle: str | None = None  # self.method(visible) called after standard actions
    requires: str | None = None  # app_state attr that must be truthy for data to exist


_PANEL_DEFS: list[_PanelDef] = [
    _PanelDef(
        "audiotrace",
        "AudioTrace",
        row=1,
        audio_row=True,
        state_attr="audiotrace_visible",
        container_method="set_audiotrace_visible",
        autoscale_plot="audio_trace_plot",
        on_toggle="_on_audio_panel_toggle",
    ),
    _PanelDef(
        "spectrogram",
        "Spectrogram",
        row=1,
        audio_row=True,
        state_attr="spectrogram_visible",
        container_method="set_spectrogram_visible",
        autoscale_plot="spectrogram_plot",
        on_toggle="_on_audio_panel_toggle",
    ),
    _PanelDef(
        "neo_viewer",
        "Neo-Viewer",
        row=1,
        container_method="set_neo_visible",
        on_toggle="_on_neo_panel_toggle",
        requires="has_neo",
    ),
    _PanelDef(
        "phy_viewer",
        "Phy-Viewer",
        row=1,
        state_attr="ephys_visible",
        container_method="set_ephys_visible",
        on_toggle="_on_phy_panel_toggle",
        requires="has_neurons",
    ),
    _PanelDef(
        "featureplot",
        "FeaturePlot",
        row=1,
        state_attr="featureplot_visible",
        container_method="set_featureplot_visible",
        requires="has_features",
    ),
    _PanelDef(
        "video_viewer",
        "VideoViewer",
        row=1,
        state_attr="video_viewer_visible",
        on_toggle="_on_video_viewer_toggle",
    ),
    _PanelDef(
        "pose_markers",
        "PoseMarkers",
        row=1,
        state_attr="pose_markers_visible",
        on_toggle="_on_pose_markers_toggle",
    ),
]


class _LoadError(Exception):
    """Raised during any loading phase to abort with a user-visible message."""


@dataclass
class _LoadContext:
    """Accumulated load state -- app_state is not mutated until _apply_to_state.

    Each loading phase reads from and writes to this context.  If any phase
    raises ``_LoadError``, the load is cancelled and app_state remains clean.
    """

    result: object  # LoadResult from data_loader.load_dataset
    nc_file_path: str
    catalog: object  # DataCatalog

    # Inferred media availability
    has_audio: bool = False
    has_neo: bool = False
    has_neurons: bool = False
    cameras: list = field(default_factory=list)
    nwb_local: str | None = None

    # NWB-embedded ephys
    nwb_ephys_display: str | None = None
    nwb_ephys_entry: tuple | None = None

    # Dataset (possibly downsampled)
    dt: object = None
    ds: object = None
    trials: list = field(default_factory=list)
    all_labels_df: object = None
    data_loader: object = None
    downsample_factor: int | None = None
    video_folder_override: str | None = None


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
        layout.setContentsMargins(
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
        )
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
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
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
        # Two rows: row 1 holds the label-slot dropdowns, row 2 the
        # secondary toggles (Confidence, Envelope, ...).
        self.overlays_layout = QVBoxLayout()
        self.overlays_layout.setSpacing(2)
        self.overlays_layout.setContentsMargins(4, 4, 4, 4)
        self.overlays_row1_layout = QHBoxLayout()
        self.overlays_row1_layout.setSpacing(15)
        self.overlays_row2_layout = QHBoxLayout()
        self.overlays_row2_layout.setSpacing(15)
        self.overlays_layout.addLayout(self.overlays_row1_layout)
        self.overlays_layout.addLayout(self.overlays_row2_layout)
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
        self.pose_hide_threshold_spin.setToolTip("Hide pose markers with confidence below this value (0.0-1.0)")
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

        row2.addWidget(QLabel("Text size:"))
        self.pose_text_size_spin = QDoubleSpinBox()
        self.pose_text_size_spin.setObjectName("pose_text_size_spin")
        self.pose_text_size_spin.setRange(4.0, 72.0)
        self.pose_text_size_spin.setSingleStep(1.0)
        self.pose_text_size_spin.setDecimals(0)
        self.pose_text_size_spin.setFixedWidth(55)
        self.pose_text_size_spin.setValue(12.0)
        row2.addWidget(self.pose_text_size_spin)

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

        # Pose ↔ Video matching
        self.pose_match_btn = QPushButton("Match Pose ↔ Video")
        self.pose_match_btn.setToolTip(
            "Open dialog to match NWB PoseEstimation containers to video cameras.\n"
            "Required when the NWB has multiple pose containers (e.g. LeftCamera, RightCamera)."
        )
        self.pose_match_btn.clicked.connect(self._on_pose_match_clicked)
        pose_layout.addWidget(self.pose_match_btn)

        # Keypoints table (inline, scrollable)
        self.keypoints_table = QTableWidget(0, 2)
        self.keypoints_table.setHorizontalHeaderLabels(["Show", "Keypoint"])
        self.keypoints_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.keypoints_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.keypoints_table.verticalHeader().setVisible(False)
        pose_layout.addWidget(self.keypoints_table, stretch=1)

        self.pose_groupbox.hide()
        parent_layout.addWidget(self.pose_groupbox, stretch=1)

    def _on_pose_match_clicked(self):
        from .dialog_pose_video_matcher import PoseVideoMatcherDialog

        sio = getattr(self.app_state, "nwb_alignment", None)

        # Saved mapping from local_settings, then NWB acquisition, then detection
        pose_keys = list(getattr(self.app_state, "nwb_pose_keys", None) or [])
        if not pose_keys and sio:
            pose_keys = sio.pose_keys
        if not pose_keys:
            nwb_local = getattr(self.app_state, "nwb_local", None)
            if nwb_local:
                pose_keys = _detect_nwb_pose_keys(nwb_local) or []

        if not pose_keys:
            from ethograph.gui.notify import notify

            notify("No PoseEstimation containers found in the NWB file.", "warning")
            return

        dialog = PoseVideoMatcherDialog(
            video_folder=getattr(self.app_state, "video_folder", "") or "",
            pose_folder=getattr(self.app_state, "pose_folder", "") or "",
            trial_ids=getattr(self.app_state, "trials", []),
            parent=self,
            pose_items=pose_keys,
        )
        if dialog.exec_():
            mapping = dialog.get_mapping()
            ordered_keys = [pose_key for _, pose_key in mapping]
            self.app_state.nwb_pose_keys = ordered_keys
            if self._update_pose_callback:
                self._update_pose_callback()

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
        layout.setContentsMargins(
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
            DEFAULT_LAYOUT_MARGIN,
        )
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
        self.overlays_row1_layout = panel.overlays_row1_layout
        self.overlays_row2_layout = panel.overlays_row2_layout
        self.pose_groupbox = panel.pose_groupbox
        self.pose_hide_threshold_spin = panel.pose_hide_threshold_spin
        self.pose_show_text_checkbox = panel.pose_show_text_checkbox
        self.pose_point_size_spin = panel.pose_point_size_spin
        self.pose_text_size_spin = panel.pose_text_size_spin
        self.keypoints_table = panel.keypoints_table

        self.pose_mgr = PoseDisplayManager(self.viewer, self.app_state, self.video_mgr, self)
        self.app_state.keypoints_changed.connect(self.populate_keypoints)

        panel.pose_hide_threshold_spin.valueChanged.connect(self._on_pose_hide_threshold_changed)
        panel.pose_show_text_checkbox.stateChanged.connect(self._on_pose_text_toggled)
        panel.pose_point_size_spin.valueChanged.connect(self._on_pose_point_size_changed)
        panel.pose_text_size_spin.valueChanged.connect(self._on_pose_text_size_changed)
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

    def _on_pose_text_size_changed(self, value: float):
        self.pose_mgr.apply_pose_style()

    def set_references(
        self,
        plot_container,
        labels_widget,
        plot_settings_widget,
        navigation_widget,
        changepoints_widget=None,
        ephys_widget=None,
        layout_mgr=None,
        trials_widget=None,
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
        dt = getattr(self.app_state, "dt", None)
        if dt is not None:
            dt.close()
            self.app_state.dt = None
        self.app_state.ds = None
        self.app_state.data_loader = None
        self.app_state.source_collection = None
        self.app_state._all_labels_df = None
        self.app_state.labels_confidence_ds = None
        self.catalog = None
        self.app_state.ready = False

    def _cancel_load(self, reason: str):
        notify_dialog(reason, "warning", "Load cancelled", self)
        self._cleanup_load_state()

    def on_load_clicked(self):
        if not self.app_state.nc_file_path:
            notify_dialog(
                "Please select a data file (.nc, .nwb, .npz) or folder",
                "warning",
                "Load cancelled",
                self,
            )
            return

        nc_file_path = self.io_widget.get_nc_file_path()

        # Phases 1-4: pure computation — app_state is untouched.
        try:
            ctx = self._phase_load_data(nc_file_path)
            self._phase_infer_media(ctx)
            self._phase_prepare_dataset(ctx)
            self._phase_validate(ctx)
        except _LoadError as e:
            self._cancel_load(str(e))
            return

        # Phase 5: single commit point — mutate app_state.
        try:
            self._apply_to_state(ctx)
        except _LoadError as e:
            self._cancel_load(str(e))
            return

        # Phase 6: UI setup (app_state is now consistent).
        self._setup_ui_after_load(ctx)

    # ------------------------------------------------------------------
    # Loading phases
    # ------------------------------------------------------------------

    def _phase_load_data(self, nc_file_path: str) -> _LoadContext:
        """Phase 1: Load dataset from disk.  May show downsample dialog."""
        try:
            result = load_dataset(
                nc_file_path,
                progress_callback=getattr(self.app_state, "_progress_callback", None),
                metadata_path=self.app_state.metadata_path,
            )
        except (OSError, ValueError, KeyError) as e:
            logger.exception("load_dataset failed")
            raise _LoadError(f"Failed to load dataset: {type(e).__name__}: {e}") from e

        video_folder_override = None
        if result.nwb_video_folder and not self.app_state.video_folder:
            from .dialog_video_downsample import offer_downsample

            video_folder_override = offer_downsample(str(result.nwb_video_folder), parent=self)

        return _LoadContext(
            result=result,
            nc_file_path=nc_file_path,
            catalog=result.catalog,
            dt=result.dt,
            all_labels_df=result.all_labels_df,
            nwb_local=result.nwb_local,
            video_folder_override=video_folder_override,
        )

    def _phase_infer_media(self, ctx: _LoadContext) -> None:
        """Phase 2: Detect video/audio/pose/ephys availability."""
        result = ctx.result
        sio = result.nwb_alignment

        ctx.has_audio = bool(sio.mics) if sio else False

        # NWB-embedded ephys — detect directly from the NWB file
        if sio:
            eseries = sio.electrical_series()
            if eseries:
                first = eseries[0]
                ctx.nwb_ephys_display = f"{first['name']} (NWB)"
                ctx.nwb_ephys_entry = (first["path"], "0", 0)

        ctx.has_neo = bool(self.app_state.ephys_path) or bool(ctx.nwb_ephys_entry)
        ctx.has_neurons = bool(self.app_state.neurons_path)

    def _phase_prepare_dataset(self, ctx: _LoadContext) -> None:
        """Phase 3: Downsample, resolve trials, build first ds + DataLoader."""
        downsample_factor = self.io_widget.get_downsample_factor()
        if downsample_factor is not None and ctx.dt is not None:
            ctx.dt = eto.downsample_trialtree(ctx.dt, downsample_factor)
            ctx.downsample_factor = downsample_factor
            logger.info("Downsampled data by factor %d", downsample_factor)

        ctx.trials = sorted(ctx.result.trial_ids)
        if ctx.dt is not None:
            ctx.ds = ctx.dt.trial(ctx.trials[0])
        else:
            ctx.ds = xr.Dataset()

        store = ctx.result.data_loader
        if store is None:
            from ethograph.io.catalog import XarrayLoader, catalog_from_xarray

            cat = catalog_from_xarray(ctx.ds, ctx.dt)
            store = XarrayLoader(ctx.ds, cat)
        ctx.data_loader = store

    def _phase_validate(self, ctx: _LoadContext) -> None:
        """Phase 4: Validate media files for the first trial."""
        video_folder = ctx.video_folder_override or self.app_state.video_folder
        missing = self._validate_media_files(
            nwb_alignment=ctx.result.nwb_alignment,
            first_trial=ctx.trials[0],
            video_folder=video_folder,
            audio_folder=self.app_state.audio_folder,
            pose_folder=self.app_state.pose_folder,
        )
        if missing:
            raise _LoadError("Missing media files for first trial:\n" + "\n".join(missing))

    def _apply_to_state(self, ctx: _LoadContext) -> None:
        """Phase 5: Single commit — mutate app_state from accumulated context."""
        # Suppress signal handlers during bulk mutation so that handlers
        # checking app_state.ready don't react to partially-set state.
        self.app_state.ready = False

        result = ctx.result

        self.app_state.dt = ctx.dt
        self.app_state.metadata_df = result.metadata_df
        self.app_state.metadata_path = result.metadata_path
        self.catalog = ctx.catalog

        if ctx.video_folder_override:
            self.app_state.video_folder = ctx.video_folder_override

        self.app_state.trial_conditions = ctx.catalog.trial_conditions
        self.app_state.source_collection = result.source_collection
        self.app_state.nwb_alignment = result.nwb_alignment

        self.app_state.has_audio = ctx.has_audio

        if ctx.nwb_local:
            self.app_state.nwb_local = ctx.nwb_local

        # NWB-embedded ephys
        if ctx.nwb_ephys_display and ctx.nwb_ephys_entry:
            if not self.app_state.ephys_path:
                self.app_state.ephys_path = ctx.nwb_ephys_entry[0]
            self.app_state.ephys_source_map[ctx.nwb_ephys_display] = ctx.nwb_ephys_entry
            self.app_state.ephys_stream_sel = ctx.nwb_ephys_display

        self.app_state.has_neo = ctx.has_neo
        self.app_state.has_neurons = ctx.has_neurons

        # Expand ephys streams (mutates io_widget + app_state.ephys_source_map)
        if self.app_state.ephys_path:
            try:
                self.io_widget._expand_ephys_with_streams(
                    self.app_state.ephys_path,
                    ctx.ds,
                )
            except (OSError, ValueError, KeyError) as e:
                logger.exception("Ephys stream expansion failed")
                raise _LoadError(f"Failed to load ephys features: {type(e).__name__}: {e}") from e

        self.io_widget.disable_downsample_controls()
        self.app_state.downsample_factor_used = ctx.downsample_factor

        self.app_state._all_labels_df = ctx.all_labels_df
        self.app_state._labels_file_path = ctx.result.labels_file_path
        self.app_state.trials = ctx.trials if ctx.trials else [1]
        self.app_state.ds = ctx.ds
        self.app_state.data_loader = ctx.data_loader

        # Set trials_sel early so _expand_mics_with_channels / get_media
        # can resolve filenames during UI creation.
        trial = getattr(self.app_state, "trials_sel", None)
        try:
            is_nan = np.isnan(trial)
        except (TypeError, ValueError):
            is_nan = False
        if not trial or is_nan or trial not in self.app_state.trials:
            self.app_state.trials_sel = self.app_state.trials[0]

    def _setup_ui_after_load(self, ctx: _LoadContext) -> None:
        """Phase 6: Create controls, restore defaults, enable UI."""
        self._create_trial_controls()

        # TrialsWidget._apply_filters may have overwritten app_state.trials
        # with an empty list (e.g. when metadata_df mismatches trial_ids).
        # Restore from the authoritative LoadResult.
        if not self.app_state.trials and ctx.trials:
            self.app_state.trials = ctx.trials

        self._restore_or_set_defaults()
        self._set_controls_enabled(True)
        self.app_state.ready = True

        self.io_widget.on_load_complete()
        self.labels_widget.refresh_mapping_for_data_dir(Path(ctx.nc_file_path).parent)
        self.changepoints_widget.setEnabled(True)
        self.plot_settings_widget.set_enabled_state()
        if self.ephys_widget:
            self.ephys_widget.setEnabled(True)
            self.ephys_widget.populate_ephys_default_path()

        self.meta_widget.configure_layout_for_data()

        # Re-apply sidebar cap after load-triggered dock/layout rearrangement.
        if getattr(self, "layout_mgr", None) is not None:
            QTimer.singleShot(
                0,
                lambda: self.layout_mgr.set_sidebar_default_width(self.meta_widget, SIDEBAR_AFTER_LOAD_WIDTH_RATIO),
            )

        self.update_trials_combo()
        self._load_trial_with_fallback()
        self._disable_empty_panels()

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
            is_verified = self.app_state.get_trial_meta(trial).get("human_verified", 0)
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

    def refresh_trials_confidence(self) -> None:
        """Refresh the Trials tab with current metadata."""
        if getattr(self, "trials_widget", None) is None:
            return
        mdf = self.app_state.metadata_df
        if mdf is None or mdf.empty:
            mdf = pd.DataFrame({"trial": self.app_state.trials})
        self.trials_widget.setup(mdf)

    def _create_trial_controls(self):
        self.io_widget.create_device_controls(self.catalog)
        self.navigation_widget.setup_trial_conditions(self.catalog)
        self.navigation_widget.set_data_widget(self)

        if getattr(self, "trials_widget", None) is not None:
            mdf = (
                self.app_state.metadata_df
                if self.app_state.metadata_df is not None
                else pd.DataFrame({"trial": self.app_state.trials})
            )
            self.trials_widget.setup(mdf)
            self.refresh_trials_confidence()

        for combo_name, combo_spec in self.catalog.combos.items():
            if not combo_spec.values:
                continue
            self._create_combo_widget(combo_name, list(combo_spec.values))

        self._create_colors_combo()

        # Restore camera combos
        cameras = self.app_state.nwb_alignment.cameras
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
                combo.currentTextChanged.connect(lambda _text, idx=i: self._on_extra_camera_combo_changed(idx))
                self._extra_camera_combos.append(combo)
                self.controls.append(combo)
                slot_layout.addWidget(combo)

        if "keypoint" in self.app_state.ds.coords:
            keypoint_names = strip_common_prefix([str(k) for k in self.app_state.ds.coords["keypoint"].values])
            self.populate_keypoints(keypoint_names)

        slot_layout.addStretch()
        self.slot_groupbox.show()

        self._setup_panel_checkboxes()

    def _is_panel_available(self, defn: _PanelDef) -> bool:
        if defn.requires is None:
            return True
        if defn.requires == "has_features":
            return bool(self.catalog and self.catalog.features)
        return bool(getattr(self.app_state, defn.requires, False))

    def _setup_panel_checkboxes(self):
        self._audio_row_widgets = []

        for defn in _PANEL_DEFS:
            available = self._is_panel_available(defn)
            saved = getattr(self.app_state, defn.state_attr) if defn.state_attr else True

            checkbox = QCheckBox(defn.label)
            checkbox.setObjectName(f"{defn.name}_checkbox")
            setattr(self, f"{defn.name}_checkbox", checkbox)

            # Set checkbox to saved state WITHOUT firing signals (avoids
            # triggering plot updates before data is loaded).
            # When data is NOT available the checkbox is forced unchecked —
            # signals are blocked so the saved preference is preserved.
            checkbox.blockSignals(True)
            checkbox.setChecked(available and saved)
            checkbox.blockSignals(False)

            # Apply the effective visibility to the container directly.
            effective = available and saved
            if defn.container_method and self.plot_container:
                getattr(self.plot_container, defn.container_method)(effective)
            if defn.on_toggle:
                getattr(self, defn.on_toggle)(effective)

            # Connect signal AFTER initial state is applied.
            checkbox.stateChanged.connect(lambda state, n=defn.name: self._on_panel_toggled(n, state))

            if not available:
                checkbox.setEnabled(False)
            if defn.audio_row:
                self._audio_row_widgets.append(checkbox)

        # Row 1: audio panel checkboxes
        self.panels_row1_layout.addWidget(self.audiotrace_checkbox)
        self.panels_row1_layout.addWidget(self.spectrogram_checkbox)
        self.panels_row1_layout.addStretch()

        # Row 2: mic selector
        if self.app_state.has_audio:
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
        self._neo_stream_label = QLabel("Preview stream:")
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

        if self.app_state.has_neo and self.ephys_widget:
            self._populate_neo_stream_combo()

        if self.app_state.has_neurons and self.ephys_widget:
            self._neural_view_label.show()
            self.neural_view_combo.show()
            self.ephys_widget.configure_ephys_trace_plot()

        self.video_mgr.set_audio_row_widgets(self._audio_row_widgets)

        # Overlays row 1 — three dropdowns choose what is rendered as labels.
        # Main fills the entire plot (except top strips). Top1 / Top2 are
        # narrow top strips, each independently configurable to a label
        # branch or to the imported predictions. Label sits flush against
        # its combo; an explicit gap separates the three label/combo groups.
        row1 = self.overlays_row1_layout
        row1.setSpacing(2)
        _GROUP_GAP = 6
        _COMBO_WIDTH = 72

        row1.addWidget(QLabel("Main:"))
        self.main_labels_combo = QComboBox()
        self.main_labels_combo.setFixedWidth(_COMBO_WIDTH)
        self.main_labels_combo.currentIndexChanged.connect(lambda _i: self._on_label_slot_changed("main"))
        row1.addWidget(self.main_labels_combo)
        row1.addSpacing(_GROUP_GAP)

        row1.addWidget(QLabel("Top 1:"))
        self.top1_labels_combo = QComboBox()
        self.top1_labels_combo.setFixedWidth(_COMBO_WIDTH)
        self.top1_labels_combo.currentIndexChanged.connect(lambda _i: self._on_label_slot_changed("top1"))
        row1.addWidget(self.top1_labels_combo)
        row1.addSpacing(_GROUP_GAP)

        row1.addWidget(QLabel("Top 2:"))
        self.top2_labels_combo = QComboBox()
        self.top2_labels_combo.setFixedWidth(_COMBO_WIDTH)
        self.top2_labels_combo.currentIndexChanged.connect(lambda _i: self._on_label_slot_changed("top2"))
        row1.addWidget(self.top2_labels_combo)

        row1.addStretch()

        # Overlays row 2 — secondary scalar overlays.
        row2 = self.overlays_row2_layout

        self.show_confidence_checkbox = QCheckBox("Confidence")
        self.show_confidence_checkbox.setChecked(False)
        self.show_confidence_checkbox.stateChanged.connect(self._update_confidence_overlay)
        row2.addWidget(self.show_confidence_checkbox)

        self.show_envelope_checkbox = QCheckBox("Envelope")
        self.show_envelope_checkbox.setChecked(False)
        self.show_envelope_checkbox.stateChanged.connect(self._on_envelope_overlay_changed)
        self.show_envelope_checkbox.hide()
        row2.addWidget(self.show_envelope_checkbox)

        row2.addStretch()

        self._set_controls_enabled(False)

        # Populate the slot dropdowns with whatever branches the labels widget
        # already has — this happens after data load, so branches exist.
        self.refresh_label_slot_dropdowns()

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

    # ------------------------------------------------------------------
    # Label slot dropdowns (Main / Top1 / Top2)
    # ------------------------------------------------------------------

    def _label_slot_options(self, *, allow_predictions: bool, allow_none: bool):
        """Build (display_text, value) pairs for a label slot dropdown.

        Branches are read from the labels widget's mapping. Predictions is
        offered iff a predictions df has been loaded.
        """
        options: list[tuple[str, object]] = []
        if allow_none:
            options.append(("(none)", None))
        if self.labels_widget is not None:
            for branch_idx in sorted(self.labels_widget._branch_sections):
                options.append((f"Branch {branch_idx}", branch_idx))
        if allow_predictions and self.app_state.pred_labels_df is not None:
            options.append(("Predict", "predictions"))
        return options

    def _populate_label_slot_combo(self, combo: QComboBox, options, current_value):
        """Re-fill *combo* with *options*, restoring *current_value* if present."""
        combo.blockSignals(True)
        combo.clear()
        chosen_index = 0
        for i, (text, value) in enumerate(options):
            combo.addItem(text, userData=value)
            if value == current_value:
                chosen_index = i
        if options:
            combo.setCurrentIndex(chosen_index)
        combo.blockSignals(False)

    def refresh_label_slot_dropdowns(self, *, new_branch: int | None = None, predictions_added: bool = False):
        """Repopulate the three slot dropdowns from current branches/predictions.

        Auto-fill rules:
          - new_branch: place the new branch into Top1 if free, else Top2.
            If both are taken, leave them alone.
          - predictions_added: same auto-fill rule, value="predictions".
        """
        if not hasattr(self, "main_labels_combo"):
            return

        state = self.app_state

        # Auto-fill Top1/Top2 before repopulating, so the new value is selected.
        if new_branch is not None and new_branch != state._main_labels_source:
            if state._top1_source is None:
                state._top1_source = new_branch
            elif state._top2_source is None:
                state._top2_source = new_branch
        if predictions_added:
            already_in_slot = state._top1_source == "predictions" or state._top2_source == "predictions"
            if not already_in_slot:
                if state._top1_source is None:
                    state._top1_source = "predictions"
                elif state._top2_source is None:
                    state._top2_source = "predictions"

        main_opts = self._label_slot_options(allow_predictions=False, allow_none=True)
        top_opts = self._label_slot_options(allow_predictions=True, allow_none=True)

        self._populate_label_slot_combo(
            self.main_labels_combo,
            main_opts,
            state._main_labels_source,
        )
        self._populate_label_slot_combo(
            self.top1_labels_combo,
            top_opts,
            state._top1_source,
        )
        self._populate_label_slot_combo(
            self.top2_labels_combo,
            top_opts,
            state._top2_source,
        )

    def set_main_labels_branch(self, branch_idx: int):
        """Programmatic main-slot change (used by Shift+B shortcut)."""
        idx = self.main_labels_combo.findData(branch_idx)
        if idx < 0:
            return
        self.main_labels_combo.setCurrentIndex(idx)

    def toggle_predictions_slot(self):
        """Toggle predictions in/out of the Top1/Top2 slots (Ctrl+Y).

        If predictions is already in a slot, clear it. Otherwise drop it
        into the first free slot (Top1, then Top2). No-op when no
        predictions have been imported.
        """
        if self.app_state.pred_labels_df is None:
            return
        state = self.app_state
        if state._top1_source == "predictions":
            target_combo, target = self.top1_labels_combo, None
            state_attr = "_top1_source"
        elif state._top2_source == "predictions":
            target_combo, target = self.top2_labels_combo, None
            state_attr = "_top2_source"
        elif state._top1_source is None:
            target_combo, target = self.top1_labels_combo, "predictions"
            state_attr = "_top1_source"
        elif state._top2_source is None:
            target_combo, target = self.top2_labels_combo, "predictions"
            state_attr = "_top2_source"
        else:
            return  # both slots taken by something else
        idx = target_combo.findData(target)
        if idx < 0:
            # Combo wasn't populated yet — set state directly and refresh.
            setattr(state, state_attr, target)
            self.refresh_label_slot_dropdowns()
            return
        target_combo.setCurrentIndex(idx)

    def _on_label_slot_changed(self, slot: str):
        """User changed Main / Top1 / Top2 dropdown — persist + redraw."""
        state = self.app_state
        combo = {
            "main": self.main_labels_combo,
            "top1": self.top1_labels_combo,
            "top2": self.top2_labels_combo,
        }[slot]
        new_value = combo.currentData()
        if slot == "main":
            prev = state._main_labels_source
            if isinstance(prev, int) and prev != new_value and self.labels_widget:
                self.labels_widget._previous_main_branch = prev
            state._main_labels_source = new_value
        elif slot == "top1":
            state._top1_source = new_value
        else:
            state._top2_source = new_value

        if self.labels_widget is not None:
            self.labels_widget._sync_active_label_ids()
        if self.app_state.ready:
            ds_kwargs = self.app_state.get_ds_kwargs()
            self.update_label_plot(ds_kwargs)
        if self.labels_widget is not None:
            self.labels_widget.refresh_labels_shapes_layer()

    def cycle_neural_view(self):
        if not hasattr(self, "neural_view_combo") or not self.neural_view_combo.isVisible():
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
        source_map = getattr(self.app_state, "ephys_source_map", {})
        if not source_map:
            self._neo_stream_label.hide()
            self.neo_stream_combo.hide()
            return

        ks_params = None
        if self.ephys_widget:
            ks_params = getattr(self.ephys_widget, "_kilosort_params", None)

        self.neo_stream_combo.blockSignals(True)
        self.neo_stream_combo.clear()

        for display_name, (filepath, stream_id, _ch) in source_map.items():
            self.neo_stream_combo.addItem(display_name)

            # Grey out streams matching kilosort params
            if ks_params:
                try:
                    loader = load_ephys(filepath, stream_id)
                    ks_sr = ks_params.get("sample_rate", 0)
                    ks_nch = ks_params.get("n_channels_dat", 0)
                    if loader.n_channels == ks_nch and abs(loader.rate - ks_sr) < 1.0:
                        idx = self.neo_stream_combo.count() - 1
                        model = self.neo_stream_combo.model()
                        item = model.item(idx)
                        if item:
                            item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                except Exception:
                    pass

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

        if stream_name is None:
            stream_name = self.neo_stream_combo.currentText() if hasattr(self, "neo_stream_combo") else ""
        if not stream_name:
            return

        source_map = getattr(self.app_state, "ephys_source_map", {})
        if stream_name not in source_map:
            return

        # Track the selected stream regardless of panel visibility so that
        # configure_ephys_trace_plot() always knows which stream is active.
        self.app_state.ephys_stream_sel = stream_name

        neo_cb = getattr(self, "neo_viewer_checkbox", None)
        if neo_cb is not None and not neo_cb.isChecked():
            return

        filepath, stream_id, channel_idx = source_map[stream_name]
        try:
            loader = load_ephys(filepath, str(stream_id))
        except Exception:
            return

        neo_starting_time = float(getattr(loader, "starting_time", 0.0) or 0.0)

        neo_plot = self.plot_container.neo_trace_plot
        neo_plot.set_loader(loader, channel_idx)
        neo_plot.set_source(FileSource("neo", loader, start_time=neo_starting_time))

        if self.plot_container._panel_visible["neo"]:
            xmin, xmax = self.plot_container.get_current_xlim()
            neo_plot.update_plot_content(xmin, xmax)
            neo_plot.auto_channel_spacing()
            neo_plot.auto_gain()
            neo_plot.autoscale()

    def cycle_view_mode(self):
        if not hasattr(self, "view_mode_combo") or not self.view_mode_combo.isVisible():
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
                return (
                    loader.channels
                    if hasattr(loader, "channels")
                    else (loader.shape[1] if len(loader.shape) > 1 else 1)
                )
        except (ImportError, OSError, ValueError):
            return 1

    def _expand_mics_with_channels(self, mic_labels):
        self.app_state.audio_source_map.clear()
        expanded_items = []
        audio_folder = self.app_state.audio_folder
        dt = getattr(self.app_state, "dt", None)
        trial_id = getattr(self.app_state, "trials_sel", None)
        if trial_id is None and self.app_state.trials:
            trial_id = self.app_state.trials[0]

        if not audio_folder or dt is None:
            for mic in mic_labels:
                display_name = str(mic)
                self.app_state.audio_source_map[display_name] = (str(mic), 0)
                expanded_items.append(display_name)
            return expanded_items

        for mic_label in mic_labels:
            mic_file = self.app_state.nwb_alignment.get_media(trial_id, "audio", str(mic_label))
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
        combo = getattr(self, "mics_combo", None)
        if combo is None:
            return
        new_items = self.app_state.nwb_alignment.mics
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
        cameras = self.app_state.nwb_alignment.cameras

        primary = getattr(self, "primary_camera_combo", None)
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

    def _is_autoscale_on(self) -> bool:
        return self.plot_settings_widget is not None and self.plot_settings_widget.autoscale_checkbox.isChecked()

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
            if hasattr(plot, "_apply_y_constraints"):
                plot._apply_y_constraints()

        if defn.on_toggle:
            getattr(self, defn.on_toggle)(visible)

    def _on_audio_panel_toggle(self, visible: bool):
        if visible and self.plot_container:
            self.plot_container.update_audio_panels()

    def _on_neo_panel_toggle(self, visible: bool):
        if visible and self.plot_container:
            self._configure_neo_panel()

    def _on_phy_panel_toggle(self, visible: bool):
        if not visible or not self.plot_container:
            return
        mode = self.neural_view_combo.currentText() if hasattr(self, "neural_view_combo") else "Multi Trace"
        self.plot_container.set_neural_panel_mode("raster" if mode == "Raster" else "trace")
        if self._is_autoscale_on():
            self.plot_container.ephys_trace_plot.vb.enableAutoRange(x=False, y=True)

    def _on_video_viewer_toggle(self, visible: bool):
        if hasattr(self, "layout_mgr") and self.layout_mgr:
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
        btn = getattr(self, "sort_channels_btn", None)
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

        if hasattr(self, "show_envelope_checkbox") and not self.show_envelope_checkbox.isChecked():
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

    def _create_colors_combo(self):
        """Create the Colors combo populated with all features, filtered by rgb suffix."""
        all_features = list(self.catalog.features)
        if not all_features:
            return

        combo = QComboBox()
        combo.setObjectName("colors_combo")
        combo.currentIndexChanged.connect(self._on_combo_changed)

        rgb_checkbox = QCheckBox("rgb suffix")
        rgb_checkbox.setObjectName("colors_rgb_suffix_checkbox")
        rgb_checkbox.setToolTip("Only show features with 'rgb' in the name")
        rgb_checkbox.setChecked(True)
        rgb_checkbox.stateChanged.connect(self._on_rgb_suffix_changed)
        self._colors_rgb_checkbox = rgb_checkbox

        self._populate_colors_combo(combo, all_features, rgb_filter=True)
        make_searchable(combo)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(5)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_layout.addWidget(combo)
        row_layout.addWidget(rgb_checkbox)

        self.coords_groupbox_layout.addRow("Colors:", row_widget)
        self.combos["colors"] = combo
        self.controls.append(combo)
        self.controls.append(rgb_checkbox)

    def _populate_colors_combo(self, combo: QComboBox, features: list[str], rgb_filter: bool):
        prev = get_combo_value(combo) if combo.count() > 0 else "None"
        combo.blockSignals(True)
        combo.clear()
        raw_items = ["None"]
        if rgb_filter:
            raw_items += [f for f in features if "rgb" in f.lower()]
        else:
            raw_items += features
        display_items = clean_display_labels(raw_items)
        for display, raw in zip(display_items, raw_items):
            combo.addItem(display, raw)
        # Restore previous selection if still available
        for i in range(combo.count()):
            if combo.itemData(i) == prev:
                combo.setCurrentIndex(i)
                break
        combo.blockSignals(False)

    def _on_rgb_suffix_changed(self, _state: int):
        combo = self.combos.get("colors")
        if combo is None:
            return
        rgb_filter = self._colors_rgb_checkbox.isChecked()
        self._populate_colors_combo(combo, list(self.catalog.features), rgb_filter)
        self.app_state.set_key_sel("colors", get_combo_value(combo))
        if self.app_state.ready and self.plot_container:
            current_plot = self.plot_container.get_current_plot()
            xmin, xmax = current_plot.get_current_xlim()
            self.update_main_plot(t0=xmin, t1=xmax)

    def _create_combo_widget(self, key, vars):
        excluded_from_all = {"individuals", "features", "cameras", "mics"}
        show_all_checkbox = key not in excluded_from_all

        combo = QComboBox()
        combo.setObjectName(f"{key}_combo")
        combo.currentIndexChanged.connect(self._on_combo_changed)
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

            if key in ["individual", "keypoint"]:
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

        # Restore colors combo (not in catalog.combos, created separately)
        colors_combo = self.combos.get("colors")
        if colors_combo is not None and self.app_state.key_sel_exists("colors"):
            saved_color = self.app_state.get_key_sel("colors")
            if saved_color is not None:
                idx = find_combo_index(colors_combo, str(saved_color))
                if idx < 0 and hasattr(self, "_colors_rgb_checkbox"):
                    # Saved value not visible with rgb filter on — disable it
                    self._colors_rgb_checkbox.blockSignals(True)
                    self._colors_rgb_checkbox.setChecked(False)
                    self._colors_rgb_checkbox.blockSignals(False)
                    self._populate_colors_combo(colors_combo, list(self.catalog.features), rgb_filter=False)
                    idx = find_combo_index(colors_combo, str(saved_color))
                if idx >= 0:
                    colors_combo.blockSignals(True)
                    colors_combo.setCurrentIndex(idx)
                    colors_combo.blockSignals(False)
                    self.app_state.set_key_sel("colors", saved_color)

        if self.app_state.key_sel_exists("trials"):
            saved_trial = self.app_state.get_key_sel("trials")
            self.app_state.set_key_sel("trials", saved_trial)
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials_sel))
        else:
            self.navigation_widget.trials_combo.setCurrentText(str(self.app_state.trials[0]))
            self.app_state.trials_sel = self.app_state.trials[0]

        space_plot_type = getattr(self.app_state, "space_plot_type", "Layers")
        # Migrate old values to simplified combo
        if space_plot_type in (
            "Space 2D",
            "Space 3D",
            "space_2D",
            "space_3D",
            "PCA 2D",
            "PCA 3D",
        ):
            space_plot_type = "Space Plot"
        if hasattr(self, "space_view_combo"):
            self.space_view_combo.setCurrentText(space_plot_type)

        saved_camera = self.app_state.primary_camera
        if saved_camera and hasattr(self, "primary_camera_combo"):
            idx = self.primary_camera_combo.findText(saved_camera)
            if idx >= 0:
                self.primary_camera_combo.setCurrentIndex(idx)

        saved_extra = self.app_state.extra_cameras
        if saved_extra and hasattr(self, "_extra_camera_combos"):
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
            logger.warning(
                "Saved %s_sel '%s' not found; reverting to '%s'",
                key,
                saved_value,
                fallback,
            )
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
        current_trial = getattr(self.app_state, "trials_sel", None)

        try:
            is_nan = np.isnan(current_trial)
        except (TypeError, ValueError):
            is_nan = False

        if not current_trial or is_nan or current_trial not in self.app_state.trials:
            if current_trial and not is_nan:
                logger.warning(
                    "Saved trial %s not in dataset, using %s",
                    current_trial,
                    first_trial,
                )
            self.app_state.trials_sel = first_trial

        self.on_trial_changed()

    def _panel_has_data(self, name: str) -> bool:
        """Check whether a panel would actually display data for the current trial."""
        if name in ("audiotrace", "spectrogram"):
            return bool(self.app_state.audio_path)
        if name == "featureplot":
            return bool(self.catalog and self.catalog.features)
        if name == "neo_viewer":
            neo_plot = getattr(self.plot_container, "neo_trace_plot", None)
            return neo_plot is not None and getattr(neo_plot, "_source", None) is not None
        if name == "phy_viewer":
            return bool(self.app_state.has_neurons)
        if name == "video_viewer":
            return bool(self.app_state.video_path)
        if name == "pose_markers":
            return bool(self.app_state.video_path)
        return True

    def _disable_empty_panels(self):
        """Uncheck and hide plot panels that have no data after first trial load."""
        for defn in _PANEL_DEFS:
            checkbox = getattr(self, f"{defn.name}_checkbox", None)
            if checkbox is None or not checkbox.isChecked():
                continue
            if self._panel_has_data(defn.name):
                continue
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            if defn.container_method and self.plot_container:
                getattr(self.plot_container, defn.container_method)(False)
            if defn.on_toggle:
                getattr(self, defn.on_toggle)(False)

    def _validate_media_files(
        self,
        nwb_alignment=None,
        first_trial=None,
        video_folder=None,
        audio_folder=None,
        pose_folder=None,
    ) -> list[str]:
        sio = nwb_alignment if nwb_alignment is not None else self.app_state.nwb_alignment
        if first_trial is None:
            first_trial = self.app_state.trials[0]
        if video_folder is None:
            video_folder = self.app_state.video_folder
        if audio_folder is None:
            audio_folder = self.app_state.audio_folder
        if pose_folder is None:
            pose_folder = self.app_state.pose_folder

        missing = []

        if video_folder:
            for cam in sio.cameras:
                vid = sio.get_media(first_trial, "video", device=cam)
                if not vid or is_url(vid):
                    continue
                path = os.path.join(video_folder, vid)
                if not os.path.isfile(path):
                    missing.append(f"Video: {path}")

        if audio_folder:
            mics = sio.mics
            if not mics:
                notify(
                    "You selected an audio folder, although the .nc contains no audio media entries.",
                    "warning",
                )
            else:
                for mic in mics:
                    aud = sio.get_media(first_trial, "audio", device=mic)
                    if not aud:
                        continue
                    path = os.path.join(audio_folder, aud)
                    if not os.path.isfile(path):
                        missing.append(f"Audio: {path}")

        if pose_folder:
            cameras = sio.cameras
            if not cameras:
                notify(
                    "You selected a pose folder, although the .nc contains no pose data.",
                    "warning",
                )
            else:
                for cam in cameras:
                    pose_file = sio.get_media(first_trial, "pose", device=cam)
                    if not pose_file:
                        continue
                    path = os.path.join(pose_folder, pose_file)
                    if not os.path.isfile(path):
                        missing.append(f"Pose: {path}")

        return missing

    def _build_trial_alignment(self, trial_id) -> None:
        self.app_state.trial_alignment = compute_trial_video_bounds(
            self.app_state.nwb_alignment,
            trial_id,
            self.app_state.ds,
            video_folder=self.app_state.video_folder,
            audio_folder=self.app_state.audio_folder,
            cameras_sel=self.app_state.primary_camera,
            source_collection=getattr(self.app_state, "source_collection", None),
        )

    def _build_restrict_window(self, trial_id) -> None:
        if self.navigation_widget is not None:
            self.navigation_widget._apply_slider_scope()

    def on_trial_changed(self):
        trials_sel = self.app_state.trials_sel

        if trials_sel not in self.app_state.trials:
            logger.warning(
                "Selected trial '%s' not found in dataset, reverting to first trial '%s'",
                trials_sel,
                self.app_state.trials[0],
            )
            trials_sel = self.app_state.trials[0]
            self.app_state.trials_sel = trials_sel
            return

        if self.app_state.dt is not None:
            self.app_state.ds = self.app_state.dt.trial(trials_sel)

        self.pose_frame_path = self.app_state.ds.attrs.get("frame_path")

        # Update data_loader (xarray only: swap the backing dataset)
        store = self.app_state.data_loader
        trial_idx = self.app_state.trials.index(trials_sel)
        if store is not None and hasattr(store, "update_ds"):
            store.update_ds(self.app_state.ds)

        # Sync SourceCollection trial index for time-range queries
        sc = getattr(self.app_state, "source_collection", None)
        if sc is not None:
            for src in sc.sources.values():
                if hasattr(src, "set_trial"):
                    src.set_trial(trial_idx)

        self._update_device_sels_for_trial(self.app_state.ds)
        self.update_mics_combo_for_trial(self.app_state.ds)

        features_combo = self.combos.get("features")
        fallback_feature = features_combo.itemText(0) if features_combo and features_combo.count() else None
        feature_sel = getattr(self.app_state, "features_sel", fallback_feature)

        if hasattr(self, "view_mode_combo"):
            self._update_view_mode_items(feature_sel)

        if hasattr(self, "view_mode_combo"):
            self._apply_view_mode_for_feature()

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

        preserve = getattr(self.app_state, "_preserve_x_range_next", False)
        self.app_state._preserve_x_range_next = False
        self.update_main_plot(preserve_x_range=preserve)
        try:
            self.update_space_plot()
        except Exception:
            logger.debug("update_space_plot failed", exc_info=True)

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
        # Labels are hidden when every slot is None.
        state = self.app_state
        any_slot = (
            state._main_labels_source is not None or state._top1_source is not None or state._top2_source is not None
        )
        if not any_slot:
            if self.plot_container:
                for plot in self.plot_container._get_all_plots():
                    self.plot_container._clear_labels_on_plot(plot)
            return

        intervals_df = self.app_state.label_intervals

        if intervals_df is not None and not intervals_df.empty and "individuals" in ds_kwargs:
            selected_ind = str(ds_kwargs["individuals"])
            intervals_df = intervals_df[intervals_df["individual"] == selected_ind]

        predictions_df = None
        if self.app_state.pred_labels_df is not None:
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
        video = getattr(self.app_state, "video", None)
        if video:
            nav = self.meta_widget.navigation_widget
            nav.connect_video_sync(video)

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
        mappings = getattr(self.labels_widget, "_mappings", {})
        color = mappings.get(key[2], {}).get("color", color)
        self.space_plot.highlight_time_segment(key[0], key[1], color)

    def _on_primary_frame_changed(self, frame_number: int):
        self.plot_container.update_time_marker_and_window(frame_number)

        video = getattr(self.app_state, "video", None)
        if video:
            current_time = video.frame_to_time(frame_number)
        else:
            current_time = frame_number / self.app_state.video_fps

        xlim = self.plot_container.get_current_xlim()
        if getattr(self.app_state, "center_playback", False) or current_time < xlim[0] or current_time > xlim[1]:
            self.plot_container.set_x_range(mode="center", center_on_frame=frame_number)

    def update_pose(self):
        """Refresh primary and extra camera pose layers through PoseDisplayManager."""
        if self.pose_mgr is None:
            return
        if not self.app_state.pose_markers_visible:
            return
        if not hasattr(self.app_state, "trials_sel") or self.app_state.trials_sel is None:
            return
        self.pose_mgr.update_pose(self.get_hidden_keypoints())

    def closeEvent(self, event):
        SharedAudioCache.clear_cache()
        from .plots_ephystrace import clear_loader_cache

        clear_loader_cache()
        if getattr(self.app_state, "video", None):
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
        else:
            if self.space_plot:
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
        if not hasattr(self, "_extra_camera_combos"):
            return set()
        names = set()
        for combo in self._extra_camera_combos:
            text = combo.currentText()
            if text and text != "None":
                names.add(text)
        return names

    def _save_extra_cameras(self):
        if not hasattr(self, "_extra_camera_combos"):
            return
        values = [combo.currentText() for combo in self._extra_camera_combos]
        self.app_state.extra_cameras = [v for v in values if v and v != "None"]

    def _update_extra_camera_combo_items(self, cameras: list[str]):
        if not hasattr(self, "_extra_camera_combos"):
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
        if not hasattr(self, "_extra_camera_combos"):
            return
        desired = self._get_desired_extra_cameras()
        to_add: dict[str, str] = {}
        for camera_name in desired:
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

        plot_type = self.app_state.get_with_default("space_plot_type")
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

        store = getattr(self.app_state, "data_loader", None)
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
                    mappings = getattr(self.labels_widget, "_mappings", {})
                    color = mappings.get(label_id, {}).get("color", color)

        self.space_plot.highlight_time_segment(start_time, end_time, color)
