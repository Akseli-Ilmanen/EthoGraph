"""Changepoints widget - dataset changepoints and audio changepoint detection."""

import numpy as np
import ruptures as rpt
import xarray as xr
import yaml
import audioio as aio

from napari.utils.notifications import show_info, show_warning
from napari.viewer import Viewer
from qtpy.QtCore import Qt, QLocale, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import ethograph as eto
from ethograph.features.changepoints import (
    correct_changepoints,
    correct_changepoints_automatic,
    extract_cp_times,
    snap_to_nearest_changepoint_time,
)
from ethograph.features.audio_changepoints import get_audio_changepoints

from .dialog_function_params import open_function_params_dialog, get_registry
from .makepretty import styled_link


# Maps UI combo text → registry key
_AUDIO_CP_REGISTRY_MAP = {
    "VocalPy meansquared": "meansquared_cp",
    "VocalPy ava": "ava_cp",
    "VocalSeg dynamic thresholding": "vocalseg_cp",
    "VocalSeg continuity filtering": "continuity_cp",
}

_RUPTURES_REGISTRY_MAP = {
    "Pelt": "ruptures_pelt",
    "Binseg": "ruptures_binseg",
    "BottomUp": "ruptures_bottomup",
    "Window": "ruptures_window",
    "Dynp": "ruptures_dynp",
}

_KINEMATIC_REGISTRY_MAP = {
    "troughs": "find_troughs",
    "turning_points": "find_turning_points",
}

_OSCILLATORY_REGISTRY_MAP = {
    "detect_oscillatory_events": "oscillatory_events",
}


def _run_ruptures_in_process(
    signal: np.ndarray,
    method: str,
    params: dict,
) -> tuple[list[int] | None, str | None]:
    try:
        model = params.get("model", "l2")
        min_size = params.get("min_size", 2)
        jump = params.get("jump", 5)

        algo_map = {
            "Pelt": lambda: rpt.Pelt(model=model, min_size=min_size, jump=jump),
            "Binseg": lambda: rpt.Binseg(model=model, min_size=min_size, jump=jump),
            "BottomUp": lambda: rpt.BottomUp(model=model, min_size=min_size, jump=jump),
            "Window": lambda: rpt.Window(
                width=params.get("width", 100), model=model, min_size=min_size, jump=jump
            ),
            "Dynp": lambda: rpt.Dynp(model=model, min_size=min_size, jump=jump),
        }

        if method not in algo_map:
            return (None, f"Unknown method: {method}")

        algo = algo_map[method]().fit(signal)

        if method == "Pelt":
            bkps = algo.predict(pen=params.get("pen", 1.0))
        elif method == "Binseg":
            pen = params.get("pen")
            if pen is not None:
                bkps = algo.predict(pen=pen)
            else:
                bkps = algo.predict(n_bkps=params.get("n_bkps", 5))
        else:
            bkps = algo.predict(n_bkps=params.get("n_bkps", 5))

        return (bkps, None)

    except Exception as e:
        return (None, str(e))


class ChangepointsWidget(QWidget):
    """Changepoints controls - dataset changepoints and audio changepoint detection."""
    
    request_plot_update = Signal()

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None
        self.meta_widget = None
        self.setAttribute(Qt.WA_AlwaysShowToolTips)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_shared_controls(main_layout)
        self._create_toggle_buttons(main_layout)
        self._create_changepoints_panel()
        self._create_ruptures_panel()
        self._create_audio_cp_panel()
        self._create_oscillatory_panel()
        self._create_correction_params_panel()

        main_layout.addWidget(self.changepoints_panel)
        main_layout.addWidget(self.ruptures_panel)
        main_layout.addWidget(self.audio_cp_panel)
        main_layout.addWidget(self.oscillatory_panel)
        main_layout.addWidget(self.correction_params_panel)

        self.changepoints_panel.hide()
        self.ruptures_panel.hide()
        self.audio_cp_panel.hide()
        self.oscillatory_panel.hide()
        self.correction_params_panel.show()
        self.correction_toggle.setText("CP Correction")

        main_layout.addStretch()

        self._restore_or_set_defaults()
        self.setEnabled(False)

    def _update_trial_dataset(self, new_ds: xr.Dataset):
        trial = self.app_state.trials_sel
        self.app_state.dt.update_trial(trial, lambda _: new_ds)
        self.app_state.ds = self.app_state.dt.trial(trial)

    def _ensure_changepoints_visible(self):
        self.show_cp_checkbox.blockSignals(True)
        self.show_cp_checkbox.setChecked(True)
        self.show_cp_checkbox.blockSignals(False)
        self.app_state.show_changepoints = True
        self.request_plot_update.emit()

    def _store_audio_cps_to_ds(
        self, onsets: np.ndarray, offsets: np.ndarray, target_feature: str, method: str
    ):
        ds = self.app_state.ds
        if ds is None:
            return

        new_ds = ds.copy()
        for var in ("audio_cp_onsets", "audio_cp_offsets"):
            if var in new_ds.data_vars:
                new_ds = new_ds.drop_vars(var)

        attrs = {"type": "audio_changepoints", "target_feature": target_feature, "method": method}
        new_ds["audio_cp_onsets"] = xr.DataArray(onsets, dims=["audio_cp"], attrs=attrs)
        new_ds["audio_cp_offsets"] = xr.DataArray(offsets, dims=["audio_cp"], attrs=attrs)
        self._update_trial_dataset(new_ds)

    def _get_audio_cps_from_ds(self) -> tuple[np.ndarray, np.ndarray] | None:
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            return None
        if "audio_cp_onsets" not in ds.data_vars or "audio_cp_offsets" not in ds.data_vars:
            return None
        return ds["audio_cp_onsets"].values, ds["audio_cp_offsets"].values

    def _draw_dataset_changepoints_on_plot(self):
        if self.plot_container:
            cp_by_method, time_array = self._get_dataset_changepoint_indices()
            if cp_by_method is not None:
                self.plot_container.draw_dataset_changepoints(time_array, cp_by_method)

    # =========================================================================
    # Shared controls / toggle buttons
    # =========================================================================

    def _create_shared_controls(self, main_layout):
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(0, 0, 0, 0)

        self.show_cp_checkbox = QCheckBox("Show changepoints")
        self.show_cp_checkbox.setToolTip("Display changepoints on plot")
        self.show_cp_checkbox.setChecked(True)
        self.show_cp_checkbox.stateChanged.connect(self._on_show_changepoints_changed)
        row1_layout.addWidget(self.show_cp_checkbox)

        self.changepoint_correction_checkbox = QCheckBox(
            "Changepoint correction"
        )
        self.changepoint_correction_checkbox.setChecked(self.app_state.apply_changepoint_correction)
        self.changepoint_correction_checkbox.setToolTip(
            "Snap label boundaries to nearest changepoint when creating labels.\n"
            "When enabled, uses full correction parameters.\n"
        )
        self.changepoint_correction_checkbox.stateChanged.connect(self._on_changepoint_correction_changed)
        row1_layout.addWidget(self.changepoint_correction_checkbox)

        row1_layout.addStretch()
        main_layout.addLayout(row1_layout)

    def _create_toggle_buttons(self, main_layout):
        self.toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        self.toggle_widget.setLayout(toggle_layout)

        toggle_defs = [
            ("correction_toggle", "CP Correction", True, self._toggle_correction_params),
            ("cp_toggle", "Kinematic CPs", False, self._toggle_changepoints),
            ("ruptures_toggle", "Ruptures", False, self._toggle_ruptures),
            ("audio_cp_toggle", "Audio CPs", False, self._toggle_audio_cps),
            ("oscillatory_toggle", "Oscillatory", False, self._toggle_oscillatory),
        ]
        for attr, label, checked, callback in toggle_defs:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(checked)
            btn.clicked.connect(callback)
            toggle_layout.addWidget(btn)
            setattr(self, attr, btn)

        main_layout.addWidget(self.toggle_widget)

    def _show_panel(self, panel_name: str):
        panels = {
            "correction": (self.correction_params_panel, self.correction_toggle, "CP Correction"),
            "kinematic": (self.changepoints_panel, self.cp_toggle, "Kinematic CPs"),
            "ruptures": (self.ruptures_panel, self.ruptures_toggle, "Ruptures"),
            "audio_cps": (self.audio_cp_panel, self.audio_cp_toggle, "Audio CPs"),
            "oscillatory": (self.oscillatory_panel, self.oscillatory_toggle, "Oscillatory"),
        }
        for name, (panel, toggle, label) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)

        self._refresh_layout()

    def _toggle_changepoints(self):
        self._show_panel("kinematic" if self.cp_toggle.isChecked() else "correction")

    def _toggle_ruptures(self):
        self._show_panel("ruptures" if self.ruptures_toggle.isChecked() else "correction")

    def _toggle_audio_cps(self):
        self._show_panel("audio_cps" if self.audio_cp_toggle.isChecked() else "correction")

    def _toggle_oscillatory(self):
        self._show_panel("oscillatory" if self.oscillatory_toggle.isChecked() else "correction")
        if self.oscillatory_toggle.isChecked():
            self._update_oscillatory_source_state()

    def _toggle_correction_params(self):
        self._show_panel("correction" if self.correction_toggle.isChecked() else "oscillatory")

    def _refresh_layout(self):
        if self.meta_widget:
            self.meta_widget.refresh_widget_layout(self)

    # =========================================================================
    # Panel creation — simplified with "Configure..." buttons
    # =========================================================================

    def _create_changepoints_panel(self):
        self.changepoints_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 0)
        self.changepoints_panel.setLayout(layout)

        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.setToolTip(
            "Troughs: local minima\n"
            "Turning points: points where gradient is near zero around peaks"
        )
        self.method_combo.addItems(["troughs", "turning_points"])
        row_layout.addWidget(self.method_combo)

        self.kinematic_configure_btn = QPushButton("Configure...")
        self.kinematic_configure_btn.setToolTip("Open parameter editor for selected method")
        self.kinematic_configure_btn.clicked.connect(self._open_kinematic_params)
        row_layout.addWidget(self.kinematic_configure_btn)
        layout.addLayout(row_layout)

        button_layout = QHBoxLayout()

        self.compute_ds_cp_button = QPushButton("Detect")
        self.compute_ds_cp_button.setToolTip(
            "Detect changepoints for current feature and add to dataset"
        )
        self.compute_ds_cp_button.clicked.connect(self._compute_dataset_changepoints)
        button_layout.addWidget(self.compute_ds_cp_button)

        self.clear_ds_cp_button = QPushButton("Clear")
        self.clear_ds_cp_button.setToolTip(
            "Remove all changepoints for current feature"
        )
        self.clear_ds_cp_button.clicked.connect(self._clear_current_feature_changepoints)
        button_layout.addWidget(self.clear_ds_cp_button)

        self.ds_cp_count_label = QLabel("")
        button_layout.addWidget(self.ds_cp_count_label)

        button_layout.addStretch()
        layout.addLayout(button_layout)

    def _create_audio_cp_panel(self):
        self.audio_cp_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.audio_cp_panel.setLayout(layout)

        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Method:"))
        self.audio_cp_method_combo = QComboBox()
        self.audio_cp_method_combo.addItems([
            "VocalPy meansquared", "VocalPy ava",
            "VocalSeg dynamic thresholding", "VocalSeg continuity filtering",
        ])
        self.audio_cp_method_combo.currentTextChanged.connect(self._on_audio_cp_method_changed)
        row_layout.addWidget(self.audio_cp_method_combo)

        self.audio_cp_configure_btn = QPushButton("Configure...")
        self.audio_cp_configure_btn.setToolTip("Open parameter editor for selected method")
        self.audio_cp_configure_btn.clicked.connect(self._open_audio_cp_params)
        row_layout.addWidget(self.audio_cp_configure_btn)
        layout.addLayout(row_layout)

        button_layout = QHBoxLayout()

        self.compute_audio_cp_button = QPushButton("Detect")
        self.compute_audio_cp_button.setToolTip(
            "Detect onset/offset candidates using selected method"
        )
        self.compute_audio_cp_button.clicked.connect(self._compute_audio_changepoints)
        button_layout.addWidget(self.compute_audio_cp_button)

        self.clear_audio_cp_button = QPushButton("Clear")
        self.clear_audio_cp_button.setToolTip(
            "Remove all audio changepoints from the plot"
        )
        self.clear_audio_cp_button.clicked.connect(self._clear_spectral_changepoints)
        button_layout.addWidget(self.clear_audio_cp_button)

        self.audio_cp_count_label = QLabel("")
        button_layout.addWidget(self.audio_cp_count_label)

        button_layout.addStretch()

        self.audio_cp_ref_label = QLabel()
        self.audio_cp_ref_label.setOpenExternalLinks(True)
        button_layout.addWidget(self.audio_cp_ref_label)

        layout.addLayout(button_layout)

        self._on_audio_cp_method_changed(self.audio_cp_method_combo.currentText())

    def _create_ruptures_panel(self):
        self.ruptures_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.ruptures_panel.setLayout(layout)

        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Method:"))
        self.ruptures_method_combo = QComboBox()
        self.ruptures_method_combo.setToolTip(
            "Pelt: Fast, penalty-based (unknown # of changepoints)\n"
            "Binseg: Binary segmentation (fast)\n"
            "BottomUp: Bottom-up segmentation\n"
            "Window: Sliding window method\n"
            "Dynp: Dynamic programming (optimal but slow)"
        )
        self.ruptures_method_combo.addItems(
            ["Pelt", "Binseg", "BottomUp", "Window", "Dynp"]
        )
        row_layout.addWidget(self.ruptures_method_combo)

        self.ruptures_configure_btn = QPushButton("Configure...")
        self.ruptures_configure_btn.setToolTip("Open parameter editor for selected method")
        self.ruptures_configure_btn.clicked.connect(self._open_ruptures_params)
        row_layout.addWidget(self.ruptures_configure_btn)
        layout.addLayout(row_layout)

        button_layout = QHBoxLayout()

        self.compute_ruptures_button = QPushButton("Detect")
        self.compute_ruptures_button.setToolTip(
            "Detect changepoints for current feature using ruptures library"
        )
        self.compute_ruptures_button.clicked.connect(self._compute_ruptures_changepoints)
        button_layout.addWidget(self.compute_ruptures_button)

        self.ruptures_count_label = QLabel("")
        button_layout.addWidget(self.ruptures_count_label)

        button_layout.addStretch()

        ref_label = QLabel(styled_link(
            "https://centre-borelli.github.io/ruptures-docs",
            "Ruptures (Truong et al., 2020)",
        ))
        ref_label.setOpenExternalLinks(True)
        ref_label.setToolTip("Open ruptures documentation")
        button_layout.addWidget(ref_label)

        layout.addLayout(button_layout)

    def _create_oscillatory_panel(self):
        self.oscillatory_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.oscillatory_panel.setLayout(layout)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Source:"))
        self.osc_source_combo = QComboBox()
        self.osc_source_combo.setToolTip(
            "Data source for oscillatory event detection.\n"
            "Ephys Trace: single-channel ephys (multichannel not supported)\n"
            "Audio Trace: single-channel audio waveform\n"
            "Current Feature: currently selected dataset feature"
        )
        self.osc_source_combo.addItems(["Ephys Trace", "Audio Trace", "Current Feature"])
        row1.addWidget(self.osc_source_combo)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Method:"))
        self.osc_method_combo = QComboBox()
        self.osc_method_combo.addItems(["detect_oscillatory_events"])
        row2.addWidget(self.osc_method_combo)

        self.osc_configure_btn = QPushButton("Configure...")
        self.osc_configure_btn.setToolTip("Open parameter editor for oscillatory event detection")
        self.osc_configure_btn.clicked.connect(self._open_oscillatory_params)
        row2.addWidget(self.osc_configure_btn)
        layout.addLayout(row2)

        button_layout = QHBoxLayout()

        self.osc_detect_btn = QPushButton("Detect")
        self.osc_detect_btn.setToolTip("Detect oscillatory events in selected data source")
        self.osc_detect_btn.clicked.connect(self._compute_oscillatory_events)
        button_layout.addWidget(self.osc_detect_btn)

        self.osc_clear_btn = QPushButton("Clear")
        self.osc_clear_btn.setToolTip("Remove all oscillatory events from plots and dataset")
        self.osc_clear_btn.clicked.connect(self._clear_oscillatory_events)
        button_layout.addWidget(self.osc_clear_btn)

        self.osc_count_label = QLabel("")
        button_layout.addWidget(self.osc_count_label)

        button_layout.addStretch()

        ref_label = QLabel(styled_link(
            "https://pynapple.org/user_guide/12_filtering.html#detecting-oscillatory-events",
            "pynapple (Viejo et al., 2023)",
        ))
        ref_label.setOpenExternalLinks(True)
        ref_label.setToolTip("Open pynapple filtering documentation")
        button_layout.addWidget(ref_label)

        layout.addLayout(button_layout)

        self._update_oscillatory_source_state()

    def _update_oscillatory_source_state(self):
        has_ephys = bool(
            getattr(self.app_state, "ephys_path", None)
            or getattr(self.app_state, "kilosort_folder", None)
        )
        model = self.osc_source_combo.model()
        ephys_item = model.item(0)  # "Ephys Trace"
        ephys_item.setEnabled(has_ephys)
        if not has_ephys and self.osc_source_combo.currentIndex() == 0:
            self.osc_source_combo.setCurrentIndex(1)

    # =========================================================================
    # Configure... dialog openers
    # =========================================================================

    def _open_oscillatory_params(self):
        method = self.osc_method_combo.currentText()
        key = _OSCILLATORY_REGISTRY_MAP.get(method)
        if key and open_function_params_dialog(key, self.app_state, parent=self) is not None:
            self._compute_oscillatory_events()

    def _open_kinematic_params(self):
        method = self.method_combo.currentText()
        key = _KINEMATIC_REGISTRY_MAP.get(method)
        if key and open_function_params_dialog(key, self.app_state, parent=self) is not None:
            self._compute_dataset_changepoints()

    def _open_audio_cp_params(self):
        method = self.audio_cp_method_combo.currentText()
        key = _AUDIO_CP_REGISTRY_MAP.get(method)
        if key and open_function_params_dialog(key, self.app_state, parent=self) is not None:
            self._compute_audio_changepoints()

    def _open_ruptures_params(self):
        method = self.ruptures_method_combo.currentText()
        key = _RUPTURES_REGISTRY_MAP.get(method)
        if key and open_function_params_dialog(key, self.app_state, parent=self) is not None:
            self._compute_ruptures_changepoints()

    # =========================================================================
    # Reference label update
    # =========================================================================

    def _on_audio_cp_method_changed(self, method: str):
        if method.startswith("VocalSeg"):
            self.audio_cp_ref_label.setText(styled_link(
                "https://github.com/timsainb/vocalization-segmentation",
                "VocalSeg (Sainburg et al., 2020)",
            ))
            self.audio_cp_ref_label.setToolTip("Open vocalseg GitHub repository")
        else:
            self.audio_cp_ref_label.setText(styled_link(
                "https://vocalpy.readthedocs.io/",
                "VocalPy (Nicholson et al.)",
            ))
            self.audio_cp_ref_label.setToolTip("Open VocalPy documentation")

    # =========================================================================
    # Setters / state
    # =========================================================================

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    # =========================================================================
    # Defaults / parameter persistence
    # =========================================================================

    def _restore_or_set_defaults(self):
        show_cp = getattr(self.app_state, "show_changepoints", False)
        self.show_cp_checkbox.setChecked(show_cp)
        self._load_correction_params_from_file()

    # =========================================================================
    # Parameter extraction from cache
    # =========================================================================

    def _get_cached_params(self, registry_key: str) -> dict:
        cache = getattr(self.app_state, "function_params_cache", None) or {}
        return dict(cache.get(registry_key, {}))

    def _get_audio_cp_params(self) -> dict:
        method = self.audio_cp_method_combo.currentText()
        key = _AUDIO_CP_REGISTRY_MAP.get(method)
        params = self._get_cached_params(key) if key else {}

        if method == "VocalSeg dynamic thresholding":
            params["method"] = "vocalseg"
        elif method == "VocalSeg continuity filtering":
            params["method"] = "continuity"
        elif method == "VocalPy ava":
            params["method"] = "ava"
            nperseg = params.get("nperseg", 1024)
            params["noverlap"] = nperseg // 2
        else:
            params["method"] = "meansquared"

        return params

    def _get_kinematic_params(self) -> dict:
        method = self.method_combo.currentText()
        key = _KINEMATIC_REGISTRY_MAP.get(method)
        return self._get_cached_params(key) if key else {}

    def _get_ruptures_params(self) -> dict:
        method = self.ruptures_method_combo.currentText()
        key = _RUPTURES_REGISTRY_MAP.get(method)
        return self._get_cached_params(key) if key else {}

    # =========================================================================
    # Show / clear changepoints on plot
    # =========================================================================

    def _on_show_changepoints_changed(self, state):
        show = state == Qt.Checked
        self.app_state.show_changepoints = show

        if not show:
            self.changepoint_correction_checkbox.setChecked(False)

        if self.plot_container:
            if show:
                result = self._get_audio_cps_from_ds()
                if result is not None:
                    self.plot_container.draw_audio_changepoints(*result)

                osc_result = self._get_oscillatory_events_from_ds()
                if osc_result is not None:
                    self.plot_container.draw_oscillatory_events(*osc_result)

                self._draw_dataset_changepoints_on_plot()
            else:
                self.plot_container.clear_audio_changepoints()
                self.plot_container.clear_oscillatory_events()
                self.plot_container.clear_dataset_changepoints()

        self.request_plot_update.emit()

    def _get_dataset_changepoint_indices(self):
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            return None, None

        cp_ds = ds.filter_by_attrs(type="changepoints")
        if len(cp_ds.data_vars) == 0:
            return None, None

        cp_by_method = {}
        time_array = None

        for var_name in cp_ds.data_vars:
            cp_da = cp_ds[var_name]
            if time_array is None:
                time_array = eto.get_time_coord(cp_da)

            cp_data = cp_da.values
            if cp_data.ndim > 1:
                cp_data = cp_data.any(axis=tuple(range(1, cp_data.ndim)))

            indices = np.where(cp_data > 0)[0]
            if len(indices) > 0:
                method_name = var_name.split("_")[-1]
                if method_name in cp_by_method:
                    cp_by_method[method_name] = np.unique(
                        np.concatenate([cp_by_method[method_name], indices])
                    )
                else:
                    cp_by_method[method_name] = indices

        if len(cp_by_method) == 0:
            return None, None

        return cp_by_method, time_array

    def _is_audio_waveform_selected(self) -> bool:
        return getattr(self.app_state, "features_sel", None) == "Audio Waveform"

    def _clear_spectral_changepoints(self):
        ds = getattr(self.app_state, "ds", None)
        if ds is not None:
            vars_to_drop = [v for v in ("audio_cp_onsets", "audio_cp_offsets") if v in ds.data_vars]
            if vars_to_drop:
                self._update_trial_dataset(ds.drop_vars(vars_to_drop))

        self.audio_cp_count_label.setText("")

        if self.plot_container:
            self.plot_container.clear_audio_changepoints()

        self.request_plot_update.emit()

    # =========================================================================
    # Audio CP detection (meansquared / ava / vocalseg)
    # =========================================================================

    def _compute_audio_changepoints(self):
        from .dialog_busy_progress import BusyProgressDialog

        audio_path, channel_idx = self.app_state.get_audio_source()
        if not audio_path:
            show_warning("No audio data loaded. Audio CPs require an audio file.")
            return
        data, sample_rate = aio.load_audio(audio_path)
        sample_rate = float(sample_rate)
        if data.ndim > 1:
            data = data[:, channel_idx]

        params = self._get_audio_cp_params()
        method = params.pop("method")
        signal_array = np.asarray(data, dtype=np.float64)

        if method in ("vocalseg", "continuity"):
            n_fft = params.get("n_fft", 1024)
            min_n_fft = int(np.ceil(0.005 * sample_rate))
            if n_fft < min_n_fft:
                params["n_fft"] = min_n_fft
                show_info(f"n_fft raised to {min_n_fft} (minimum for sample rate {sample_rate:.0f} Hz)")
        elif method == "ava":
            nperseg = params.get("nperseg", 1024)
            max_nperseg = max(4, len(signal_array) // 4)
            if nperseg > max_nperseg:
                nperseg = max_nperseg
                params["nperseg"] = nperseg
            params["noverlap"] = nperseg // 2

        def _run():
            return get_audio_changepoints(
                method=method,
                signal=signal_array,
                sr=sample_rate,
                **params,
            )

        dialog = BusyProgressDialog(f"Detecting audio changepoints ({method})...", parent=self)
        result, error = dialog.execute(_run)

        if dialog.was_cancelled:
            return
        if error:
            show_warning(f"Error detecting changepoints: {error}")
            return

        (onsets, offsets), env_time, envelope = result

        if method == "meansquared" and self.plot_container:
            threshold = params.get("threshold", 5000)
            self.plot_container.draw_amplitude_envelope(env_time, envelope, threshold)
        elif method == "ava" and self.plot_container:
            self.plot_container.draw_amplitude_envelope(
                env_time, envelope,
                (params.get("thresh_lowest", 0.1),
                 params.get("thresh_min", 0.2),
                 params.get("thresh_max", 0.3)),
            )

        if len(onsets) == 0 and len(offsets) == 0:
            show_info("No changepoints detected. Try adjusting parameters.")
            return

        self._store_audio_cps_to_ds(onsets, offsets, "Audio Waveform", method)
        self.audio_cp_count_label.setText(f"{len(onsets)}+{len(offsets)}")
        show_info(f"Detected {len(onsets)} onsets, {len(offsets)} offsets")

        if self.plot_container:
            self.plot_container.draw_audio_changepoints(onsets, offsets)

        self._ensure_changepoints_visible()

    # =========================================================================
    # Oscillatory event detection (pynapple)
    # =========================================================================

    def _store_oscillatory_events_to_ds(
        self, onsets: np.ndarray, offsets: np.ndarray, target_feature: str, method: str
    ):
        ds = self.app_state.ds
        if ds is None:
            return
        new_ds = ds.copy()
        for var in ("osc_event_onsets", "osc_event_offsets"):
            if var in new_ds.data_vars:
                new_ds = new_ds.drop_vars(var)
        attrs = {"type": "oscillatory_events", "target_feature": target_feature, "method": method}
        new_ds["osc_event_onsets"] = xr.DataArray(onsets, dims=["osc_event"], attrs=attrs)
        new_ds["osc_event_offsets"] = xr.DataArray(offsets, dims=["osc_event"], attrs=attrs)
        self._update_trial_dataset(new_ds)

    def _get_oscillatory_events_from_ds(self) -> tuple[np.ndarray, np.ndarray] | None:
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            return None
        if "osc_event_onsets" not in ds.data_vars or "osc_event_offsets" not in ds.data_vars:
            return None
        return ds["osc_event_onsets"].values, ds["osc_event_offsets"].values

    def _compute_oscillatory_events(self):
        from .dialog_busy_progress import BusyProgressDialog
        from ethograph.features.oscillatory import detect_oscillatory_events_np

        source = self.osc_source_combo.currentText()
        method = self.osc_method_combo.currentText()
        key = _OSCILLATORY_REGISTRY_MAP.get(method)
        params = self._get_cached_params(key) if key else {}

        if source == "Ephys Trace":
            from .plots_ephystrace import get_loader as get_ephys_loader
            ephys_path, stream_id, channel_idx = self.app_state.get_ephys_source()
            if not ephys_path:
                show_warning("No ephys data loaded")
                return
            loader = get_ephys_loader(ephys_path, stream_id=stream_id)
            if loader is None:
                show_warning("Could not open ephys file")
                return
            sample_rate = float(loader.rate)
            raw = loader[:]
            if raw.ndim > 1:
                raw = raw[:, min(channel_idx, raw.shape[1] - 1)]
            signal_array = np.asarray(raw, dtype=np.float64)
            target_feature = f"ephys_ch{channel_idx}"

        elif source == "Audio Trace":
            audio_path, channel_idx = self.app_state.get_audio_source()
            if not audio_path:
                show_warning("No audio data loaded")
                return
            data, sample_rate = aio.load_audio(audio_path)
            sample_rate = float(sample_rate)
            if data.ndim > 1:
                data = data[:, channel_idx]
            signal_array = np.asarray(data, dtype=np.float64)
            target_feature = "Audio Waveform"

        else:  # Current Feature
            features_sel = self.app_state.features_sel
            if not features_sel or features_sel in ("Audio Waveform", "Ephys trace", "Firing rate"):
                show_warning("Select a standard dataset feature (not Audio/Ephys/Firing rate)")
                return
            ds_kwargs = self.app_state.get_ds_kwargs()
            data, _ = eto.sel_valid(self.app_state.ds[features_sel], ds_kwargs)
            feature_sr = self.app_state.get_feature_sr()
            
            if np.asarray(data).ndim > 1:
                show_warning("Oscillatory event detection requires 1-D data. Select a single dimension.")
                return
            

            signal_array = np.asarray(data, dtype=np.float64).ravel()
            target_feature = features_sel

        def _run():
            return detect_oscillatory_events_np(
                data=signal_array, sr=feature_sr, **params,
            )

        dialog = BusyProgressDialog("Detecting oscillatory events...", parent=self)
        result, error = dialog.execute(_run)

        if dialog.was_cancelled:
            return
        if error:
            show_warning(f"Error detecting oscillatory events: {error}")
            return

        onsets, offsets = result
        if len(onsets) == 0:
            show_info("No oscillatory events detected. Try adjusting parameters (freq_band, thresh_band).")
            return

        self._store_oscillatory_events_to_ds(onsets, offsets, target_feature, method)
        self.osc_count_label.setText(f"{len(onsets)} events")
        show_info(f"Detected {len(onsets)} oscillatory events")

        if self.plot_container:
            self.plot_container.draw_oscillatory_events(onsets, offsets)

        self._ensure_changepoints_visible()

    def _clear_oscillatory_events(self):
        ds = getattr(self.app_state, "ds", None)
        if ds is not None:
            vars_to_drop = [v for v in ("osc_event_onsets", "osc_event_offsets") if v in ds.data_vars]
            if vars_to_drop:
                self._update_trial_dataset(ds.drop_vars(vars_to_drop))

        self.osc_count_label.setText("")

        if self.plot_container:
            self.plot_container.clear_oscillatory_events()

        self.request_plot_update.emit()

    # =========================================================================
    # Kinematic (dataset) changepoint detection
    # =========================================================================

    def _compute_dataset_changepoints(self):
        from ethograph.features.changepoints import (
            find_troughs_binary,
            find_nearest_turning_points_binary,
        )
        from .dialog_busy_progress import BusyProgressDialog

        method = self.method_combo.currentText()
        func_kwargs = self._get_kinematic_params()

        if method == "troughs":
            changepoint_func = find_troughs_binary
            changepoint_name = "troughs"
        else:
            changepoint_func = find_nearest_turning_points_binary
            changepoint_name = "turning_points"

        ds_copy = self.app_state.ds.copy()
        feature = self.app_state.features_sel

        def _run():
            return eto.add_changepoints_to_ds(
                ds=ds_copy,
                target_feature=feature,
                changepoint_name=changepoint_name,
                changepoint_func=changepoint_func,
                **func_kwargs,
            )

        dialog = BusyProgressDialog(f"Detecting {changepoint_name}...", parent=self)
        new_ds, error = dialog.execute(_run)

        if dialog.was_cancelled:
            return
        if error:
            show_warning(f"Error computing changepoints: {error}")
            return

        cp_var_name = f"{feature}_{changepoint_name}"
        self._update_trial_dataset(new_ds)

        n_changepoints = np.sum(new_ds[cp_var_name].values > 0)
        self.ds_cp_count_label.setText(f"{n_changepoints} changepoints")
        show_info(f"Added '{cp_var_name}' with {n_changepoints} changepoints")

        self._ensure_changepoints_visible()
        self._draw_dataset_changepoints_on_plot()

    def _clear_current_feature_changepoints(self):
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            show_warning("No dataset loaded")
            return

        feature = getattr(self.app_state, "features_sel", None)
        if not feature:
            show_warning("No feature selected in Data Controls")
            return

        n_removed = self._clear_all_changepoints_for_feature(feature)

        if n_removed == 0:
            show_info(f"No changepoints found for '{feature}'")
            return

        self.ds_cp_count_label.setText("")
        self.ruptures_count_label.setText("")
        show_info(f"Removed {n_removed} changepoint variable(s) for '{feature}'")

        if self.plot_container:
            self.plot_container.clear_dataset_changepoints()

        self.request_plot_update.emit()

    def _clear_all_changepoints_for_feature(self, feature: str) -> int:
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            return 0

        cp_suffixes = ["_peaks", "_troughs", "_turning_points", "_ruptures"]
        vars_to_remove = [
            f"{feature}{suffix}"
            for suffix in cp_suffixes
            if f"{feature}{suffix}" in ds.data_vars
        ]

        if not vars_to_remove:
            return 0

        self._update_trial_dataset(ds.drop_vars(vars_to_remove))
        return len(vars_to_remove)

    # =========================================================================
    # Ruptures detection (via BusyProgressDialog + ProcessPoolExecutor)
    # =========================================================================

    def _compute_ruptures_changepoints(self):
        from .dialog_busy_progress import BusyProgressDialog

        features_sel = self.app_state.features_sel
        ds_kwargs = self.app_state.get_ds_kwargs()
        if features_sel == "Audio Waveform":
            show_warning(
                "Raw audio is too large for ruptures. "
                "Select a derived feature or use Audio CPs instead."
            )
            return

        data, _ = eto.sel_valid(self.app_state.ds[features_sel], ds_kwargs)

        signal = np.asarray(data).reshape(-1, 1)
        method = self.ruptures_method_combo.currentText()
        params = self._get_ruptures_params()

        dialog = BusyProgressDialog(
            f"Detecting ruptures ({method})...", parent=self, use_process=True,
        )
        result, error = dialog.execute(
            _run_ruptures_in_process, signal, method, params,
        )

        if dialog.was_cancelled:
            self.ruptures_count_label.setText("Cancelled")
            return
        if error:
            show_warning(f"Error computing ruptures changepoints: {error}")
            return

        bkps, error_msg = result
        if error_msg:
            show_warning(f"Error computing ruptures changepoints: {error_msg}")
            return
        if bkps is None:
            return

        signal_len = len(signal)
        if bkps and bkps[-1] == signal_len:
            bkps = bkps[:-1]

        cp_array = np.zeros(signal_len, dtype=np.int8)
        for bkp in bkps:
            if 0 <= bkp < signal_len:
                cp_array[bkp] = 1

        time_coord = self.app_state.time_coord
        
        cp_var_name = f"{features_sel}_ruptures"

        new_ds = self.app_state.ds.copy()
        if cp_var_name in new_ds.data_vars:
            new_ds = new_ds.drop_vars(cp_var_name)

        model = params.get("model", "l2")
        new_ds[cp_var_name] = xr.Variable(
            dims=[time_coord.name],
            data=cp_array,
            attrs={
                "type": "changepoints",
                "target_feature": features_sel,
                "method": f"ruptures_{method}",
                "model": model,
            },
        )

        self._update_trial_dataset(new_ds)

        n_changepoints = len(bkps)
        self.ruptures_count_label.setText(f"{n_changepoints} changepoints")
        show_info(f"Added '{cp_var_name}' with {n_changepoints} changepoints")

        self._ensure_changepoints_visible()
        self._draw_dataset_changepoints_on_plot()

    # =========================================================================
    # Correction Parameters Panel (unchanged)
    # =========================================================================


    def _create_correction_params_panel(self):
        self.correction_params_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.correction_params_panel.setLayout(layout)

        self._motif_mappings = {}
        self._custom_label_thresholds = {}
        self._correction_snapshot = None

        self._automatic_correction_group = QGroupBox("Changepoint correction (automatic during labelling)")
        automatic_layout = QGridLayout()
        self._automatic_correction_group.setLayout(automatic_layout)
        layout.addWidget(self._automatic_correction_group)

        self._manual_correction_group = QGroupBox("Changepoint correction (manual) - in development")
        manual_layout = QVBoxLayout()
        self._manual_correction_group.setLayout(manual_layout)
        layout.addWidget(self._manual_correction_group)

        self.max_expansion_spin = QDoubleSpinBox()
        self.max_expansion_spin.setRange(0, 100000)
        self.max_expansion_spin.setDecimals(3)
        self.max_expansion_spin.setToolTip(
            "Max expansion of label boundaries at changepoints"
        )

        self.max_shrink_spin = QDoubleSpinBox()
        self.max_shrink_spin.setRange(0, 100000)
        self.max_shrink_spin.setDecimals(3)
        self.max_shrink_spin.setToolTip(
            "Max shrinkage of label boundaries at changepoints"
        )

        self.manual_min_label_length_spin = QDoubleSpinBox()
        self.manual_min_label_length_spin.setRange(0.001, 100000)
        self.manual_min_label_length_spin.setDecimals(3)
        self.manual_min_label_length_spin.setToolTip(
            "Minimum label length used by manual changepoint correction."
        )

        self.manual_stitch_gap_spin = QDoubleSpinBox()
        self.manual_stitch_gap_spin.setRange(0, 100000)
        self.manual_stitch_gap_spin.setDecimals(3)
        self.manual_stitch_gap_spin.setToolTip(
            "Gap threshold used by manual changepoint correction."
        )

        self.automatic_min_label_length_spin = QDoubleSpinBox()
        self.automatic_min_label_length_spin.setRange(0.001, 100000)
        self.automatic_min_label_length_spin.setDecimals(3)
        self.automatic_min_label_length_spin.setToolTip(
            "Minimum label length used in the automatic cleanup after applying a label."
        )
        self.automatic_min_label_length_spin.setValue(
            getattr(self.app_state, "automatic_min_label_length_s", 1e-3)
        )
        self.automatic_min_label_length_spin.valueChanged.connect(
            lambda value: setattr(self.app_state, "automatic_min_label_length_s", float(value))
        )

        self.automatic_stitch_gap_spin = QDoubleSpinBox()
        self.automatic_stitch_gap_spin.setRange(0, 100000)
        self.automatic_stitch_gap_spin.setDecimals(3)
        self.automatic_stitch_gap_spin.setToolTip(
            "Gap threshold used in the automatic cleanup after applying a label."
        )
        self.automatic_stitch_gap_spin.setValue(
            getattr(self.app_state, "automatic_stitch_gap_s", 0.0)
        )
        self.automatic_stitch_gap_spin.valueChanged.connect(
            lambda value: setattr(self.app_state, "automatic_stitch_gap_s", float(value))
        )

        for spin in (
            self.max_expansion_spin,
            self.max_shrink_spin,
            self.manual_min_label_length_spin,
            self.manual_stitch_gap_spin,
            self.automatic_min_label_length_spin,
            self.automatic_stitch_gap_spin,
        ):
            spin.setLocale(QLocale(QLocale.C))
            spin.setDecimals(3)
            spin.setSuffix(" s")

        automatic_layout.addWidget(QLabel("Min label length (s):"), 0, 0)
        automatic_layout.addWidget(self.automatic_min_label_length_spin, 0, 1)
        automatic_layout.addWidget(QLabel("Stitch gap (s):"), 0, 2)
        automatic_layout.addWidget(self.automatic_stitch_gap_spin, 0, 3)

        manual_grid = QGridLayout()
        manual_grid.addWidget(QLabel("Min label length (s):"), 0, 0)
        manual_grid.addWidget(self.manual_min_label_length_spin, 0, 1)
        manual_grid.addWidget(QLabel("Stitch gap (s):"), 0, 2)
        manual_grid.addWidget(self.manual_stitch_gap_spin, 0, 3)
        manual_grid.addWidget(QLabel("Max expansion (s):"), 1, 0)
        manual_grid.addWidget(self.max_expansion_spin, 1, 1)
        manual_grid.addWidget(QLabel("Max shrink (s):"), 1, 2)
        manual_grid.addWidget(self.max_shrink_spin, 1, 3)
        manual_note = QLabel(
            "Manual correction is for testing parameters of model correction."
        )
        manual_note.setWordWrap(True)
        manual_layout.addWidget(manual_note)
        manual_layout.addLayout(manual_grid)

        button_layout = QHBoxLayout()

        self.per_label_btn = QPushButton("Per-label thresholds...")
        self.per_label_btn.setToolTip("Override min label length for individual labels")
        self.per_label_btn.clicked.connect(self._open_label_thresholds_dialog)
        button_layout.addWidget(self.per_label_btn)

        self.save_params_btn = QPushButton("Save")
        self.save_params_btn.setToolTip("Save correction parameters to changepoint_settings.yaml")
        self.save_params_btn.clicked.connect(self._save_correction_params)
        button_layout.addWidget(self.save_params_btn)

        self.load_params_btn = QPushButton("Load")
        self.load_params_btn.setToolTip("Load correction parameters from changepoint_settings.yaml")
        self.load_params_btn.clicked.connect(self._load_correction_params)
        button_layout.addWidget(self.load_params_btn)

        button_layout.addStretch()
        manual_layout.addLayout(button_layout)

        correction_layout = QHBoxLayout()

        cp_label = QLabel("Apply manual correction to:")
        correction_layout.addWidget(cp_label)

        self.cp_correction_trial_btn = QPushButton("Single Trial")
        self.cp_correction_trial_btn.clicked.connect(lambda: self._cp_correction("single_trial"))
        correction_layout.addWidget(self.cp_correction_trial_btn)

        self.cp_correction_all_trials_btn = QPushButton("All Trials")
        self.cp_correction_all_trials_btn.clicked.connect(lambda: self._cp_correction("all_trials"))
        correction_layout.addWidget(self.cp_correction_all_trials_btn)

        self.cp_undo_btn = QPushButton("\u21bb")
        self.cp_undo_btn.setToolTip("Undo last manual correction")
        self.cp_undo_btn.setFixedWidth(30)
        self.cp_undo_btn.setEnabled(False)
        self.cp_undo_btn.clicked.connect(self._undo_correction)
        correction_layout.addWidget(self.cp_undo_btn)

        correction_layout.addStretch()
        manual_layout.addLayout(correction_layout)

        apply_cp = self.changepoint_correction_checkbox.isChecked()
        self._automatic_correction_group.setEnabled(apply_cp)
        self._manual_correction_group.setEnabled(apply_cp)

    def set_motif_mappings(self, mappings: dict):
        self._motif_mappings = mappings

    def _open_label_thresholds_dialog(self):
        if not self._motif_mappings:
            show_warning("No label mappings loaded yet")
            return

        dialog = LabelThresholdsDialog(
            self._motif_mappings,
            self._custom_label_thresholds,
            self.manual_min_label_length_spin.value(),
            parent=self,
        )
        if dialog.exec_():
            self._custom_label_thresholds = dialog.get_custom_thresholds()
            n_custom = len(self._custom_label_thresholds)
            if n_custom:
                self.per_label_btn.setText(f"Per-label thresholds ({n_custom})...")
            else:
                self.per_label_btn.setText("Per-label thresholds...")


    def is_changepoint_correction_enabled(self) -> bool:
        return self.changepoint_correction_checkbox.isChecked()




    def _on_changepoint_correction_changed(self, state):
        enabled = state == Qt.Checked
        self.app_state.apply_changepoint_correction = enabled
        if hasattr(self, "_automatic_correction_group"):
            self._automatic_correction_group.setEnabled(enabled)
        if hasattr(self, "_manual_correction_group"):
            self._manual_correction_group.setEnabled(enabled)

    def _save_correction_snapshot(self, mode):
        snapshot = {"mode": mode}
        if mode == "single_trial":
            trial = self.app_state.trials_sel
            snapshot["trial"] = trial
            snapshot["intervals_df"] = self.app_state.get_trial_intervals(trial).copy()
        elif mode == "all_trials":
            snapshot["trials"] = {}
            for trial in self.app_state.label_dt.trials:
                snapshot["trials"][trial] = self.app_state.get_trial_intervals(trial).copy()
            snapshot["old_cp_corrected"] = self.app_state.label_dt.attrs.get("changepoint_corrected", 0)
        self._correction_snapshot = snapshot
        self.cp_undo_btn.setEnabled(True)

    def _undo_correction(self):
        if self._correction_snapshot is None:
            return
        snapshot = self._correction_snapshot
        mode = snapshot["mode"]

        if mode == "single_trial":
            trial = snapshot["trial"]
            self.app_state.set_trial_intervals(trial, snapshot["intervals_df"])
            if trial == self.app_state.trials_sel:
                self.app_state.label_intervals = snapshot["intervals_df"]
        elif mode == "all_trials":
            for trial, df in snapshot["trials"].items():
                self.app_state.set_trial_intervals(trial, df)
            self.app_state.label_dt.attrs["changepoint_corrected"] = snapshot["old_cp_corrected"]
            self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)
            self._update_cp_status()
        
        self._correction_snapshot = None
        self.cp_undo_btn.setEnabled(False)
        if self.data_widget:
            self.data_widget.update_main_plot()
        show_info("Reverted correction")

    def _correct_trial_intervals(self, trial, ds, all_params, ds_kwargs):
        """Interval-native correction: purge -> stitch -> snap -> purge."""
        intervals_df = self.app_state.get_trial_intervals(trial)

    
        time_coord = self.app_state.time_coord

        cp_kwargs = all_params.get("cp_kwargs", ds_kwargs)
        cp_times = extract_cp_times(ds, time_coord.values, **cp_kwargs)

        all_cp_times = [cp_times]
        if "audio_cp_onsets" in ds.data_vars and "audio_cp_offsets" in ds.data_vars:
            all_cp_times.append(ds["audio_cp_onsets"].values.astype(np.float64))
            all_cp_times.append(ds["audio_cp_offsets"].values.astype(np.float64))
        if "osc_event_onsets" in ds.data_vars and "osc_event_offsets" in ds.data_vars:
            all_cp_times.append(ds["osc_event_onsets"].values.astype(np.float64))
            all_cp_times.append(ds["osc_event_offsets"].values.astype(np.float64))
        cp_times = np.unique(np.concatenate(all_cp_times)) if len(all_cp_times) > 1 else cp_times


        min_duration_s = all_params.get("min_label_length_s", 0)
        label_thresholds_raw = all_params.get("label_thresholds", {})
        stitch_gap_s = all_params.get("stitch_gap_len_s", 0)
        cp_params = all_params.get("changepoint_params", {})
        max_expansion_s = cp_params.get("max_expansion_s", np.inf)
        max_shrink_s = cp_params.get("max_shrink_s", np.inf)


        label_thresholds_s = {int(k): v for k, v in label_thresholds_raw.items()}

  
        return correct_changepoints(
            intervals_df,
            cp_times,
            min_duration_s=min_duration_s,
            stitch_gap_s=stitch_gap_s,
            max_expansion_s=max_expansion_s,
            max_shrink_s=max_shrink_s,
            label_thresholds_s=label_thresholds_s or None,
        )
        
    def cp_correction_from_labelling(self):
        if not self.app_state.apply_changepoint_correction:
            return

        min_duration_s, stitch_gap_s = self.get_apply_label_cleanup_params()

        self._save_correction_snapshot("single_trial")
        trial = self.app_state.trials_sel
        corrected_df = correct_changepoints_automatic(
            self.app_state.get_trial_intervals(trial),
            min_duration_s=min_duration_s,
            stitch_gap_s=stitch_gap_s,
        )
        self.app_state.set_trial_intervals(trial, corrected_df)
        self.app_state.label_intervals = corrected_df

        

    def _cp_correction(self, mode):
        all_params = self.get_correction_params()
        ds_kwargs = self.app_state.get_ds_kwargs()
        all_params["cp_kwargs"] = ds_kwargs

        try:
            if mode == "single_trial":
                self._save_correction_snapshot(mode)
                trial = self.app_state.trials_sel
                corrected_df = self._correct_trial_intervals(trial, self.app_state.ds, all_params, ds_kwargs)
                self.app_state.set_trial_intervals(trial, corrected_df)
                self.app_state.label_intervals = corrected_df
                self.app_state.label_dt.trial(trial).attrs['changepoint_corrected'] = np.int8(1)
                self._update_cp_status()

            if mode == "all_trials":
                if self.app_state.label_dt.attrs.get("changepoint_corrected", 0) == 1:
                    show_warning("Changepoint correction has already been applied to all trials. Don't re-apply.")
                    return

                # TODO: Mention in documentation, only Ctrl+Z functionality of the GUI.
                self._save_correction_snapshot(mode)
                for trial in self.app_state.label_dt.trials:
                    ds = self.app_state.dt.trial(trial)
                    corrected_df = self._correct_trial_intervals(trial, ds, all_params, ds_kwargs)
                    self.app_state.set_trial_intervals(trial, corrected_df)
                    self.app_state.label_dt.trial(trial).attrs['changepoint_corrected'] = np.int8(1)
                self.app_state.label_dt.attrs["changepoint_corrected"] = np.int8(1)
                self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)
                self._update_cp_status()

            if self.data_widget:
                self.data_widget.update_main_plot()
        except Exception as e:
            import traceback
            traceback.print_exc()
            show_warning(f"Changepoint correction failed: {e}")

    def _update_cp_status(self):
        default_style = ""
        corrected_style = "background-color: green; color: white;"

        if self.app_state.label_dt is None or self.app_state.trials_sel is None:
            self.cp_correction_trial_btn.setStyleSheet(default_style)
            self.cp_correction_all_trials_btn.setStyleSheet(default_style)
            return

        apply_cp = self.changepoint_correction_checkbox.isChecked()
        self.cp_correction_all_trials_btn.setEnabled(apply_cp)
        self.cp_correction_all_trials_btn.setToolTip("")

        trial_corrected = self.app_state.label_dt.trial(self.app_state.trials_sel).attrs.get('changepoint_corrected', 0)
        self.cp_correction_trial_btn.setStyleSheet(corrected_style if trial_corrected else default_style)

        global_corrected = self.app_state.label_dt.attrs.get('changepoint_corrected', 0)
        self.cp_correction_all_trials_btn.setStyleSheet(corrected_style if global_corrected else default_style)

    def get_correction_params(self) -> dict:
        return {
            "min_label_length_s": self.manual_min_label_length_spin.value(),
            "label_thresholds": {str(k): v for k, v in self._custom_label_thresholds.items()},
            "stitch_gap_len_s": self.manual_stitch_gap_spin.value(),
            "changepoint_params": {
                "max_expansion_s": self.max_expansion_spin.value(),
                "max_shrink_s": self.max_shrink_spin.value(),
            },
        }

    def get_apply_label_cleanup_params(self) -> tuple[float, float]:
        return (
            self.automatic_min_label_length_spin.value(),
            self.automatic_stitch_gap_spin.value(),
        )

    def _get_active_sr(self) -> float | None:
        ds = getattr(self.app_state, "ds", None)
        if ds is not None and "audio_cp_onsets" in ds.data_vars:
            audio_path, _ = self.app_state.get_audio_source()
            if audio_path:
                _, sr = aio.load_audio(audio_path)
                return float(sr)
        return self.app_state.get_feature_sr()

    def _save_correction_params(self):
        params = self.get_correction_params()
        sr = self._get_active_sr()
        if sr is not None:
            params["sr"] = float(sr)
        params_path = eto.get_project_root() / "configs" / "changepoint_settings.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)
        show_info(f"Saved correction parameters to {params_path.name}")

    def _load_correction_params(self):
        params_path = eto.get_project_root() / "configs" / "changepoint_settings.yaml"
        if not params_path.exists():
            show_warning(f"No settings file found at {params_path}")
            return
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        self._apply_correction_params(params)
        show_info(f"Loaded correction parameters from {params_path.name}")

    def _load_correction_params_from_file(self):
        params_path = eto.get_project_root() / "configs" / "changepoint_settings.yaml"
        if not params_path.exists():
            return
        try:
            with open(params_path, "r") as f:
                params = yaml.safe_load(f)
            if params:
                self._apply_correction_params(params)
        except (OSError, yaml.YAMLError):
            pass

    def _apply_correction_params(self, params: dict):
        self.manual_min_label_length_spin.setValue(
            params.get("min_label_length_s", self.manual_min_label_length_spin.value())
        )
        self.manual_stitch_gap_spin.setValue(
            params.get("stitch_gap_len_s", self.manual_stitch_gap_spin.value())
        )
        cp_params = params.get("changepoint_params", {})
        self.max_expansion_spin.setValue(
            cp_params.get("max_expansion_s", self.max_expansion_spin.value())
        )
        self.max_shrink_spin.setValue(
            cp_params.get("max_shrink_s", self.max_shrink_spin.value())
        )

        self._custom_label_thresholds = {
            int(k): v for k, v in params.get("label_thresholds", {}).items()
        }
        n_custom = len(self._custom_label_thresholds)
        if n_custom:
            self.per_label_btn.setText(f"Per-label thresholds ({n_custom})...")
        else:
            self.per_label_btn.setText("Per-label thresholds...")

    # =========================================================================
    # Changepoint navigation (jump forward/backward between CPs)
    # =========================================================================

    def jump_changepoint(self, direction: int):
        """Jump to the next (direction=+1) or previous (direction=-1) changepoint.

        Panel context:
        - audio/spectrogram panel last clicked → audio changepoints
        - feature/ephys/raster panel last clicked → dataset kinematic changepoints
        """
        if self.plot_container is None:
            return

        current_time = self._get_current_time()
        cp_times = self._get_jump_cp_times()
        if cp_times is None or len(cp_times) == 0:
            show_info("No changepoints available. Detect changepoints first.")
            return

        target = self._find_adjacent_cp(cp_times, current_time, direction)
        if target is None:
            return

        self._seek_to_time(target)

    def _get_current_time(self) -> float:
        video = getattr(self.app_state, 'video', None)
        if video:
            return video.frame_to_time(self.app_state.current_frame)
        return self.plot_container.time_slider.current_time

    def _get_jump_cp_times(self) -> np.ndarray | None:
        last_panel = getattr(self.plot_container, '_last_clicked_panel', 'feature')
        if last_panel in ('audio', 'spectrogram'):
            result = self._get_audio_cps_from_ds()
            if result is None:
                return None
            onsets, offsets = result
            return np.unique(np.concatenate([onsets, offsets]))
        else:
            cp_by_method, time_array = self._get_dataset_changepoint_indices()
            if cp_by_method is None:
                return None
            all_indices = np.unique(np.concatenate(list(cp_by_method.values())))
            return np.asarray(time_array)[all_indices]

    def _find_adjacent_cp(self, cp_times: np.ndarray, current_time: float, direction: int) -> float | None:
        if direction > 0:
            candidates = cp_times[cp_times > current_time + 1e-3]
            return float(candidates[0]) if len(candidates) > 0 else None
        else:
            candidates = cp_times[cp_times < current_time - 1e-3]
            return float(candidates[-1]) if len(candidates) > 0 else None

    def _seek_to_time(self, time_s: float):
        video = getattr(self.app_state, 'video', None)
        if video:
            new_frame = video.time_to_frame(time_s)
            self.app_state.current_frame = new_frame
            video.blockSignals(True)
            video.seek_to_frame(new_frame)
            video.blockSignals(False)
        else:
            self.plot_container.time_slider.set_slider_time(time_s)

        self.plot_container.update_time_marker_by_time(time_s)

        xlim = self.plot_container.get_current_xlim()
        if time_s < xlim[0] or time_s > xlim[1]:
            window_size = self.app_state.get_with_default("window_size")
            half = window_size / 2.0
            master = self.plot_container._xlink_master or self.plot_container._feature_plot
            master.vb.setXRange(time_s - half, time_s + half, padding=0)

    def closeEvent(self, event):
        super().closeEvent(event)


class LabelThresholdsDialog(QDialog):

    def __init__(self, motif_mappings: dict, custom_thresholds: dict,
                 global_min: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Per-label min length")
        self.setMinimumWidth(350)

        self._global_min = global_min
        self._custom_thresholds = dict(custom_thresholds)

        layout = QVBoxLayout(self)

        info = QLabel(f"Global min label length: {global_min} s")
        layout.addWidget(info)

        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["ID", "Name", "Min Length"])
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(24)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        self._table.setColumnWidth(0, 35)
        self._table.setColumnWidth(2, 90)

        items = [(k, v) for k, v in motif_mappings.items() if k != 0]
        self._table.setRowCount(len(items))
        self._spins: dict[int, QDoubleSpinBox] = {}

        for row_idx, (motif_id, data) in enumerate(items):
            id_item = QTableWidgetItem(str(motif_id))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row_idx, 0, id_item)

            name_item = QTableWidgetItem(data["name"])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row_idx, 1, name_item)

            spin = QDoubleSpinBox()
            spin.setLocale(QLocale(QLocale.C))
            spin.setRange(0.001, 100000)
            spin.setDecimals(3)
            spin.setSuffix(" s")
            spin.setValue(self._custom_thresholds.get(motif_id, global_min))
            self._spins[motif_id] = spin
            self._table.setCellWidget(row_idx, 2, spin)

        layout.addWidget(self._table)

        btn_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset all to global")
        reset_btn.clicked.connect(self._reset_all)
        btn_layout.addWidget(reset_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _reset_all(self):
        for spin in self._spins.values():
            spin.setValue(self._global_min)

    def get_custom_thresholds(self) -> dict[int, float]:
        result = {}
        for motif_id, spin in self._spins.items():
            val = spin.value()
            if val != self._global_min:
                result[motif_id] = val
        return result
