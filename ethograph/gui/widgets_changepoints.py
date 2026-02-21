"""Changepoints widget - dataset changepoints and audio changepoint detection."""

import numpy as np
import ruptures as rpt
import xarray as xr
import yaml
import audioio as aio

from napari.utils.notifications import show_info, show_warning
from napari.viewer import Viewer
from qtpy.QtCore import Qt, Signal
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

from ethograph import TrialTree, get_project_root
from ethograph.utils.data_utils import get_time_coord, sel_valid
from ethograph.features.changepoints import extract_cp_times, correct_changepoints
from ethograph.features.audio_changepoints import get_audio_changepoints

from .dialog_function_params import open_function_params_dialog, get_registry


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
        self._has_audio = False

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
        self._create_correction_params_panel()

        main_layout.addWidget(self.changepoints_panel)
        main_layout.addWidget(self.ruptures_panel)
        main_layout.addWidget(self.audio_cp_panel)
        main_layout.addWidget(self.correction_params_panel)

        self.changepoints_panel.hide()
        self.ruptures_panel.hide()
        self.audio_cp_panel.hide()
        self.correction_params_panel.show()
        self.correction_toggle.setText("CP Correction")

        main_layout.addStretch()

        self._restore_or_set_defaults()
        self.setEnabled(False)

    def _update_trial_dataset(self, new_ds: xr.Dataset):
        trial = self.app_state.trials_sel
        trial_node = TrialTree.trial_key(trial)
        self.app_state.dt[trial_node] = xr.DataTree(new_ds)
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
            "When disabled, uses fallback min_label_length=2 only."
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

    def _toggle_correction_params(self):
        self._show_panel("correction" if self.correction_toggle.isChecked() else "audio_cps")

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

        ref_label = QLabel(
            '<a href="https://centre-borelli.github.io/ruptures-docs" '
            'style="color: #87CEEB; text-decoration: none;">Ruptures (Truong et al., 2020)</a>'
        )
        ref_label.setOpenExternalLinks(True)
        ref_label.setToolTip("Open ruptures documentation")
        button_layout.addWidget(ref_label)

        layout.addLayout(button_layout)

    # =========================================================================
    # Configure... dialog openers
    # =========================================================================

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
            self.audio_cp_ref_label.setText(
                '<a href="https://github.com/timsainb/vocalization-segmentation" '
                'style="color: #87CEEB; text-decoration: none;">VocalSeg (Sainburg et al., 2020)</a>'
            )
            self.audio_cp_ref_label.setToolTip("Open vocalseg GitHub repository")
        else:
            self.audio_cp_ref_label.setText(
                '<a href="https://vocalpy.readthedocs.io/" '
                'style="color: #87CEEB; text-decoration: none;">VocalPy (Nicholson et al.)</a>'
            )
            self.audio_cp_ref_label.setToolTip("Open VocalPy documentation")

    # =========================================================================
    # Setters / state
    # =========================================================================

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    def set_enabled_state(self, enabled: bool):
        self._has_audio = enabled

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

                self._draw_dataset_changepoints_on_plot()
            else:
                self.plot_container.clear_audio_changepoints()
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
                time_array = get_time_coord(cp_da)

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

        features_sel = self.app_state.features_sel
        ds_kwargs = self.app_state.get_ds_kwargs()
        if features_sel == "Audio Waveform":
            audio_path, channel_idx = self.app_state.get_audio_source()
            data, sample_rate = aio.load_audio(audio_path)
            sample_rate = float(sample_rate)
            if data.ndim > 1:
                data = data[:, channel_idx]
        else:
            data, _ = sel_valid(self.app_state.ds[features_sel], ds_kwargs)
            dt = np.median(np.diff(np.asarray(self.app_state.time)))
            sample_rate = 1.0 / dt

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
                params["nperseg"] = max_nperseg
            params["noverlap"] = params["nperseg"] // 2

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

        self._store_audio_cps_to_ds(onsets, offsets, features_sel, method)
        self.audio_cp_count_label.setText(f"{len(onsets)}+{len(offsets)}")
        show_info(f"Detected {len(onsets)} onsets, {len(offsets)} offsets")

        if self.plot_container:
            self.plot_container.draw_audio_changepoints(onsets, offsets)

        self._ensure_changepoints_visible()

    # =========================================================================
    # Kinematic (dataset) changepoint detection
    # =========================================================================

    def _compute_dataset_changepoints(self):
        from ethograph import add_changepoints_to_ds
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
            return add_changepoints_to_ds(
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

        data, _ = sel_valid(self.app_state.ds[features_sel], ds_kwargs)

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

        time_coord_name = self.app_state.time.name
        cp_var_name = f"{features_sel}_ruptures"

        new_ds = self.app_state.ds.copy()
        if cp_var_name in new_ds.data_vars:
            new_ds = new_ds.drop_vars(cp_var_name)

        model = params.get("model", "l2")
        new_ds[cp_var_name] = xr.Variable(
            dims=[time_coord_name],
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

        self._correction_params_group = QGroupBox("Global parameters")
        global_layout = QGridLayout()
        self._correction_params_group.setLayout(global_layout)
        layout.addWidget(self._correction_params_group)

        self.min_label_length_spin = QSpinBox()
        self.min_label_length_spin.setRange(1, 10000)
        self.min_label_length_spin.setValue(10)
        self.min_label_length_spin.setToolTip(
            "Global minimum label length in samples.\n"
            "Labels shorter than this are removed."
        )
        self.min_label_length_spin.valueChanged.connect(self._on_min_label_length_changed)

        self.stitch_gap_spin = QSpinBox()
        self.stitch_gap_spin.setRange(0, 10000)
        self.stitch_gap_spin.setValue(3)
        self.stitch_gap_spin.setToolTip(
            "Max gap (samples) between same-label segments to stitch together"
        )

        self.max_expansion_spin = QDoubleSpinBox()
        self.max_expansion_spin.setRange(0, 1000)
        self.max_expansion_spin.setDecimals(1)
        self.max_expansion_spin.setValue(10.0)
        self.max_expansion_spin.setToolTip(
            "Max expansion of label boundaries at changepoints (samples)"
        )

        self.max_shrink_spin = QDoubleSpinBox()
        self.max_shrink_spin.setRange(0, 1000)
        self.max_shrink_spin.setDecimals(1)
        self.max_shrink_spin.setValue(10.0)
        self.max_shrink_spin.setToolTip(
            "Max shrinkage of label boundaries at changepoints (samples)"
        )

        row = 0
        global_layout.addWidget(QLabel("Min label length:"), row, 0)
        global_layout.addWidget(self.min_label_length_spin, row, 1)
        global_layout.addWidget(QLabel("Stitch gap:"), row, 2)
        global_layout.addWidget(self.stitch_gap_spin, row, 3)

        row += 1
        global_layout.addWidget(QLabel("Max expansion:"), row, 0)
        global_layout.addWidget(self.max_expansion_spin, row, 1)
        global_layout.addWidget(QLabel("Max shrink:"), row, 2)
        global_layout.addWidget(self.max_shrink_spin, row, 3)

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
        layout.addLayout(button_layout)

        correction_layout = QHBoxLayout()

        cp_label = QLabel("Apply correction to:")
        correction_layout.addWidget(cp_label)

        self.cp_correction_trial_btn = QPushButton("Single Trial")
        self.cp_correction_trial_btn.clicked.connect(lambda: self._cp_correction("single_trial"))
        correction_layout.addWidget(self.cp_correction_trial_btn)

        self.cp_correction_all_trials_btn = QPushButton("All Trials")
        self.cp_correction_all_trials_btn.clicked.connect(lambda: self._cp_correction("all_trials"))
        correction_layout.addWidget(self.cp_correction_all_trials_btn)

        self.cp_undo_btn = QPushButton("\u21bb")
        self.cp_undo_btn.setToolTip("Undo last correction")
        self.cp_undo_btn.setFixedWidth(30)
        self.cp_undo_btn.setEnabled(False)
        self.cp_undo_btn.clicked.connect(self._undo_correction)
        correction_layout.addWidget(self.cp_undo_btn)

        correction_layout.addStretch()
        layout.addLayout(correction_layout)

        apply_cp = self.changepoint_correction_checkbox.isChecked()
        self._correction_params_group.setEnabled(apply_cp)
        self.per_label_btn.setEnabled(apply_cp)
        self.save_params_btn.setEnabled(apply_cp)
        self.load_params_btn.setEnabled(apply_cp)
        self.cp_correction_trial_btn.setEnabled(apply_cp)
        self.cp_correction_all_trials_btn.setEnabled(apply_cp)

    def set_motif_mappings(self, mappings: dict):
        self._motif_mappings = mappings

    def _open_label_thresholds_dialog(self):
        if not self._motif_mappings:
            show_warning("No label mappings loaded yet")
            return

        dialog = LabelThresholdsDialog(
            self._motif_mappings,
            self._custom_label_thresholds,
            self.min_label_length_spin.value(),
            parent=self,
        )
        if dialog.exec_():
            self._custom_label_thresholds = dialog.get_custom_thresholds()
            n_custom = len(self._custom_label_thresholds)
            if n_custom:
                self.per_label_btn.setText(f"Per-label thresholds ({n_custom})...")
            else:
                self.per_label_btn.setText("Per-label thresholds...")

    def _on_min_label_length_changed(self, _value: int):
        pass

    def get_effective_min_length(self, labels: int) -> int:
        if not self.app_state.apply_changepoint_correction:
            return 2
        return self._custom_label_thresholds.get(labels, self.min_label_length_spin.value())

    def is_changepoint_correction_enabled(self) -> bool:
        return self.changepoint_correction_checkbox.isChecked()

    def _on_changepoint_correction_changed(self, state):
        enabled = state == Qt.Checked
        self.app_state.apply_changepoint_correction = enabled
        if hasattr(self, '_correction_params_group'):
            self._correction_params_group.setEnabled(enabled)
        for btn_name in ('per_label_btn', 'save_params_btn', 'load_params_btn',
                         'cp_correction_trial_btn', 'cp_correction_all_trials_btn'):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(enabled)

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
        self.app_state.labels_modified.emit()
        show_info("Reverted correction")

    def _correct_trial_intervals(self, trial, ds, all_params, ds_kwargs):
        """Interval-native correction: purge -> stitch -> snap -> purge."""
        intervals_df = self.app_state.get_trial_intervals(trial)

        feature_sel = self.app_state.features_sel
        if feature_sel and feature_sel in ds.data_vars:
            time_coord = get_time_coord(ds[feature_sel]).values
        else:
            first_var = next(iter(ds.data_vars), None)
            time_coord = get_time_coord(ds[first_var]).values if first_var else np.arange(100) * 0.01

        sr = 1.0 / np.median(np.diff(time_coord)) if len(time_coord) > 1 else 30.0

        cp_kwargs = all_params.get("cp_kwargs", ds_kwargs)
        cp_times = extract_cp_times(ds, time_coord, **cp_kwargs)

        min_label_length = all_params.get("min_label_length", 2)
        label_thresholds_samples = all_params.get("label_thresholds", {})
        stitch_gap_len = all_params.get("stitch_gap_len", 0)
        cp_params = all_params.get("changepoint_params", {})
        max_expansion = cp_params.get("max_expansion", np.inf)
        max_shrink = cp_params.get("max_shrink", np.inf)

        min_duration_s = min_label_length / sr
        stitch_gap_s = stitch_gap_len / sr
        max_expansion_s = max_expansion / sr
        max_shrink_s = max_shrink / sr
        label_thresholds_s = {
            int(k): v / sr for k, v in label_thresholds_samples.items()
        }

        return correct_changepoints(
            intervals_df,
            cp_times,
            min_duration_s=min_duration_s,
            stitch_gap_s=stitch_gap_s,
            max_expansion_s=max_expansion_s,
            max_shrink_s=max_shrink_s,
            label_thresholds_s=label_thresholds_s or None,
        )

    def _cp_correction(self, mode):
        all_params = self.get_correction_params()
        ds_kwargs = self.app_state.get_ds_kwargs()
        all_params["cp_kwargs"] = ds_kwargs

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

            self._save_correction_snapshot(mode)
            for trial in self.app_state.label_dt.trials:
                ds = self.app_state.dt.trial(trial)
                corrected_df = self._correct_trial_intervals(trial, ds, all_params, ds_kwargs)
                self.app_state.set_trial_intervals(trial, corrected_df)
                self.app_state.label_dt.trial(trial).attrs['changepoint_corrected'] = np.int8(1)
            self.app_state.label_dt.attrs["changepoint_corrected"] = np.int8(1)
            self.app_state.label_intervals = self.app_state.get_trial_intervals(self.app_state.trials_sel)
            self._update_cp_status()

        self.app_state.labels_modified.emit()

    def _update_cp_status(self):
        default_style = ""
        corrected_style = "background-color: green; color: white;"

        if self.app_state.label_dt is None or self.app_state.trials_sel is None:
            self.cp_correction_trial_btn.setStyleSheet(default_style)
            self.cp_correction_all_trials_btn.setStyleSheet(default_style)
            return

        trial_corrected = self.app_state.label_dt.trial(self.app_state.trials_sel).attrs.get('changepoint_corrected', 0)
        self.cp_correction_trial_btn.setStyleSheet(corrected_style if trial_corrected else default_style)

        global_corrected = self.app_state.label_dt.attrs.get('changepoint_corrected', 0)
        self.cp_correction_all_trials_btn.setStyleSheet(corrected_style if global_corrected else default_style)

    def get_correction_params(self) -> dict:
        if self.app_state.apply_changepoint_correction:
            label_thresholds = {
                str(k): v for k, v in self._custom_label_thresholds.items()
            }
            return {
                "min_label_length": self.min_label_length_spin.value(),
                "label_thresholds": label_thresholds,
                "stitch_gap_len": self.stitch_gap_spin.value(),
                "changepoint_params": {
                    "max_expansion": self.max_expansion_spin.value(),
                    "max_shrink": self.max_shrink_spin.value(),
                },
            }
        return {
            "min_label_length": 2,
            "label_thresholds": {},
            "stitch_gap_len": 0,
            "changepoint_params": {
                "max_expansion": 0,
                "max_shrink": 0,
            },
        }

    def _save_correction_params(self):
        params = self.get_correction_params()
        params_path = get_project_root() / "configs" / "changepoint_settings.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)
        show_info(f"Saved correction parameters to {params_path.name}")

    def _load_correction_params(self):
        params_path = get_project_root() / "configs" / "changepoint_settings.yaml"
        if not params_path.exists():
            show_warning(f"No settings file found at {params_path}")
            return
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        self._apply_correction_params(params)
        show_info(f"Loaded correction parameters from {params_path.name}")

    def _load_correction_params_from_file(self):
        params_path = get_project_root() / "configs" / "changepoint_settings.yaml"
        if not params_path.exists():
            return
        try:
            with open(params_path, "r") as f:
                params = yaml.safe_load(f)
            if params:
                self._apply_correction_params(params)
        except Exception:
            pass

    def _apply_correction_params(self, params: dict):
        self.min_label_length_spin.blockSignals(True)
        self.min_label_length_spin.setValue(params.get("min_label_length", 10))
        self.min_label_length_spin.blockSignals(False)

        self.stitch_gap_spin.setValue(params.get("stitch_gap_len", 3))

        cp_params = params.get("changepoint_params", {})
        self.max_expansion_spin.setValue(cp_params.get("max_expansion", 10.0))
        self.max_shrink_spin.setValue(cp_params.get("max_shrink", 10.0))

        self._custom_label_thresholds = {
            int(k): v for k, v in params.get("label_thresholds", {}).items()
        }
        n_custom = len(self._custom_label_thresholds)
        if n_custom:
            self.per_label_btn.setText(f"Per-label thresholds ({n_custom})...")
        else:
            self.per_label_btn.setText("Per-label thresholds...")

    def closeEvent(self, event):
        super().closeEvent(event)


class LabelThresholdsDialog(QDialog):

    def __init__(self, motif_mappings: dict, custom_thresholds: dict,
                 global_min: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Per-label min length")
        self.setMinimumWidth(350)

        self._global_min = global_min
        self._custom_thresholds = dict(custom_thresholds)

        layout = QVBoxLayout(self)

        info = QLabel(f"Global min label length: {global_min}")
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
        self._spins: dict[int, QSpinBox] = {}

        for row_idx, (motif_id, data) in enumerate(items):
            id_item = QTableWidgetItem(str(motif_id))
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row_idx, 0, id_item)

            name_item = QTableWidgetItem(data["name"])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row_idx, 1, name_item)

            spin = QSpinBox()
            spin.setRange(1, 10000)
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

    def get_custom_thresholds(self) -> dict[int, int]:
        result = {}
        for motif_id, spin in self._spins.items():
            val = spin.value()
            if val != self._global_min:
                result[motif_id] = val
        return result
