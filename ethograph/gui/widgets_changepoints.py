"""Changepoints widget - dataset changepoints and audio changepoint detection."""

import numpy as np
import xarray as xr
from qtpy.QtWidgets import (
    QGridLayout,
    QLineEdit,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
)
from qtpy.QtCore import Qt, Signal, QTimer
from napari.viewer import Viewer
from napari.utils.notifications import show_info, show_warning
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, Future
import ruptures as rpt


def _run_ruptures_in_process(
    signal: np.ndarray,
    method: str,
    model: str,
    min_size: int,
    jump: int,
    params: dict,
) -> tuple[list[int] | None, str | None]:
    """Run ruptures detection in a separate process."""
    try:
        if method == "Pelt":
            algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(signal)
            bkps = algo.predict(pen=params.get("penalty", 1.0))
        elif method == "Binseg":
            algo = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(signal)
            if params.get("penalty") is not None:
                bkps = algo.predict(pen=params["penalty"])
            elif params.get("n_bkps") is not None:
                bkps = algo.predict(n_bkps=params["n_bkps"])
            else:
                bkps = algo.predict(n_bkps=5)
        elif method == "BottomUp":
            algo = rpt.BottomUp(model=model, min_size=min_size, jump=jump).fit(signal)
            bkps = algo.predict(n_bkps=params.get("n_bkps", 5))
        elif method == "Window":
            width = params.get("width", 100)
            algo = rpt.Window(
                width=width, model=model, min_size=min_size, jump=jump
            ).fit(signal)
            bkps = algo.predict(n_bkps=params.get("n_bkps", 5))
        elif method == "Dynp":
            algo = rpt.Dynp(model=model, min_size=min_size, jump=jump).fit(signal)
            bkps = algo.predict(n_bkps=params.get("n_bkps", 5))
        else:
            return (None, f"Unknown method: {method}")

        return (bkps, None)

    except Exception as e:
        return (None, str(e))


class ChangepointsWidget(QWidget):
    """Changepoints controls - dataset changepoints and audio changepoint detection."""

    audio_changepoints_updated = Signal()
    request_plot_update = Signal()

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None
        self.meta_widget = None

        self._ruptures_executor: Optional[ProcessPoolExecutor] = None
        self._ruptures_future: Optional[Future] = None
        self._ruptures_context = None
        self._ruptures_poll_timer: Optional[QTimer] = None

        self.setAttribute(Qt.WA_AlwaysShowToolTips)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_shared_controls(main_layout)
        self._create_toggle_buttons(main_layout)
        self._create_changepoints_panel()
        self._create_audio_changepoints_panel()
        self._create_ruptures_panel()

        main_layout.addWidget(self.changepoints_panel)
        main_layout.addWidget(self.audio_changepoints_panel)
        main_layout.addWidget(self.ruptures_panel)

        self.changepoints_panel.hide()
        self.audio_changepoints_panel.show()
        self.ruptures_panel.hide()
        self.audio_toggle.setText("Audio CPs ✓")

        main_layout.addStretch()

        self._restore_or_set_defaults()
        self.setEnabled(False)

    def _create_shared_controls(self, main_layout):
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(0, 0, 0, 0)

        self.show_cp_checkbox = QCheckBox("Show changepoints")
        self.show_cp_checkbox.setToolTip("Display changepoints on plot")
        self.show_cp_checkbox.stateChanged.connect(self._on_show_changepoints_changed)
        row1_layout.addWidget(self.show_cp_checkbox)

        self.changepoint_correction_checkbox = QCheckBox(
            "Changepoint correction for labels"
        )
        self.changepoint_correction_checkbox.setChecked(True)
        self.changepoint_correction_checkbox.setToolTip(
            "Snap label boundaries to nearest changepoint when creating labels"
        )
        row1_layout.addWidget(self.changepoint_correction_checkbox)

        row1_layout.addStretch()
        main_layout.addLayout(row1_layout)

    def _create_toggle_buttons(self, main_layout):
        self.toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        self.toggle_widget.setLayout(toggle_layout)

        self.cp_toggle = QPushButton("Kinematic CPs")
        self.cp_toggle.setCheckable(True)
        self.cp_toggle.setChecked(False)
        self.cp_toggle.clicked.connect(self._toggle_changepoints)
        toggle_layout.addWidget(self.cp_toggle)

        self.ruptures_toggle = QPushButton("Ruptures CPs")
        self.ruptures_toggle.setCheckable(True)
        self.ruptures_toggle.setChecked(False)
        self.ruptures_toggle.clicked.connect(self._toggle_ruptures)
        toggle_layout.addWidget(self.ruptures_toggle)

        self.audio_toggle = QPushButton("Audio CPs")
        self.audio_toggle.setCheckable(True)
        self.audio_toggle.setChecked(True)
        self.audio_toggle.clicked.connect(self._toggle_audio_changepoints)
        toggle_layout.addWidget(self.audio_toggle)

        main_layout.addWidget(self.toggle_widget)

    def _show_panel(self, panel_name: str):
        panels = {
            "kinematic": (self.changepoints_panel, self.cp_toggle, "Kinematic CPs"),
            "ruptures": (self.ruptures_panel, self.ruptures_toggle, "Ruptures CPs"),
            "audio": (self.audio_changepoints_panel, self.audio_toggle, "Audio CPs"),
        }
        for name, (panel, toggle, label) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
                toggle.setText(f"{label} ✓")
            else:
                panel.hide()
                toggle.setChecked(False)
                toggle.setText(label)

        self._refresh_layout()

    def _toggle_changepoints(self):
        if self.cp_toggle.isChecked():
            self._show_panel("kinematic")
        else:
            self._show_panel("audio")

    def _toggle_audio_changepoints(self):
        if self.audio_toggle.isChecked():
            self._show_panel("audio")
        else:
            self._show_panel("kinematic")

    def _toggle_ruptures(self):
        if self.ruptures_toggle.isChecked():
            self._show_panel("ruptures")
        else:
            self._show_panel("audio")

    def _refresh_layout(self):
        if self.meta_widget and hasattr(self.meta_widget, "collapsible_widgets"):
            for collapsible in self.meta_widget.collapsible_widgets:
                if hasattr(collapsible, "content_widget"):
                    content = collapsible.content_widget
                    if content and self in content.findChildren(QWidget):
                        collapsible.collapse()
                        from qtpy.QtWidgets import QApplication

                        QApplication.processEvents()
                        collapsible.expand()

    def _create_changepoints_panel(self):
        self.changepoints_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.changepoints_panel.setLayout(layout)

        params_group = QGroupBox("Kinematic detection parameters")
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        self.method_combo = QComboBox()
        self.method_combo.setToolTip(
            "Peaks: local maxima\n"
            "Troughs: local minima\n"
            "Turning points: points where gradient is near zero around peaks"
        )
        self.method_combo.addItems(["peaks", "troughs", "turning_points"])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)

        row = 0
        params_layout.addWidget(QLabel("Method:"), row, 0)
        params_layout.addWidget(self.method_combo, row, 1, 1, 3)

        self.prominence_edit = QLineEdit("0.5")
        self.prominence_edit.setToolTip(
            "Minimum prominence of peaks (scipy.signal.find_peaks)"
        )

        self.distance_edit = QLineEdit("2")
        self.distance_edit.setToolTip(
            "Minimum horizontal distance between peaks in samples"
        )

        self.width_edit = QLineEdit("")
        self.width_edit.setToolTip("Minimum width of peaks in samples (optional)")

        row += 1
        params_layout.addWidget(QLabel("Prominence:"), row, 0)
        params_layout.addWidget(self.prominence_edit, row, 1)
        params_layout.addWidget(QLabel("Distance:"), row, 2)
        params_layout.addWidget(self.distance_edit, row, 3)

        self.threshold_label = QLabel("Threshold:")
        self.threshold_edit = QLineEdit("1.0")
        self.threshold_edit.setToolTip(
            "Gradient threshold for turning point detection (only for turning_points)"
        )

        self.max_value_label = QLabel("Max value:")
        self.max_value_edit = QLineEdit("")
        self.max_value_edit.setToolTip(
            "Maximum value to qualify as turning point (optional)"
        )

        row += 1
        params_layout.addWidget(QLabel("Width:"), row, 0)
        params_layout.addWidget(self.width_edit, row, 1)
        params_layout.addWidget(self.threshold_label, row, 2)
        params_layout.addWidget(self.threshold_edit, row, 3)

        row += 1
        params_layout.addWidget(self.max_value_label, row, 0)
        params_layout.addWidget(self.max_value_edit, row, 1)

        self._on_method_changed(self.method_combo.currentText())

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

    def _create_audio_changepoints_panel(self):
        self.audio_changepoints_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.audio_changepoints_panel.setLayout(layout)

        params_group = QGroupBox("Vocalseg parameters")
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        self.hop_length_edit = QLineEdit()
        self.hop_length_edit.setToolTip("Hop length in milliseconds for spectrogram")
        self.min_level_db_edit = QLineEdit()
        self.min_level_db_edit.setToolTip(
            "Minimum dB level threshold for segmentation.\n"
            "More negative = more sensitive (detects quieter sounds)."
        )
        self.min_syllable_edit = QLineEdit()
        self.min_syllable_edit.setToolTip("Minimum syllable duration in seconds")
        self.silence_threshold_edit = QLineEdit()
        self.silence_threshold_edit.setToolTip(
            "Envelope threshold for offset detection (0-1).\n"
            "Lower = offsets detected later (captures more tail).\n"
            "Higher = offsets detected earlier."
        )
        self.ref_level_db_edit = QLineEdit()
        self.ref_level_db_edit.setToolTip("Reference level dB of audio (default: 20)")

        self.spectral_range_widget = QWidget()
        spectral_layout = QHBoxLayout()
        spectral_layout.setContentsMargins(0, 0, 0, 0)
        spectral_layout.setSpacing(2)
        self.spectral_range_widget.setLayout(spectral_layout)

        self.spectral_min_spin = QDoubleSpinBox()
        self.spectral_min_spin.setRange(0, 100000)
        self.spectral_min_spin.setDecimals(0)
        self.spectral_min_spin.setSpecialValueText("None")
        self.spectral_min_spin.setToolTip("Minimum frequency in Hz (0 = no filter)")

        self.spectral_max_spin = QDoubleSpinBox()
        self.spectral_max_spin.setRange(0, 100000)
        self.spectral_max_spin.setDecimals(0)
        self.spectral_max_spin.setSpecialValueText("None")
        self.spectral_max_spin.setToolTip("Maximum frequency in Hz (0 = no filter)")

        spectral_layout.addWidget(self.spectral_min_spin)
        spectral_layout.addWidget(QLabel("–"))
        spectral_layout.addWidget(self.spectral_max_spin)

        row = 0
        params_layout.addWidget(QLabel("Hop (ms):"), row, 0)
        params_layout.addWidget(self.hop_length_edit, row, 1)
        params_layout.addWidget(QLabel("Min dB:"), row, 2)
        params_layout.addWidget(self.min_level_db_edit, row, 3)

        row += 1
        params_layout.addWidget(QLabel("Min syl (s):"), row, 0)
        params_layout.addWidget(self.min_syllable_edit, row, 1)
        params_layout.addWidget(QLabel("Silence:"), row, 2)
        params_layout.addWidget(self.silence_threshold_edit, row, 3)

        row += 1
        params_layout.addWidget(QLabel("Ref dB:"), row, 0)
        params_layout.addWidget(self.ref_level_db_edit, row, 1)
        params_layout.addWidget(QLabel("Spec range (Hz):"), row, 2)
        params_layout.addWidget(self.spectral_range_widget, row, 3)

        button_layout = QHBoxLayout()

        self.compute_cp_button = QPushButton("Detect")
        self.compute_cp_button.setToolTip(
            "Detect onset/offset candidates in current audio"
        )
        self.compute_cp_button.clicked.connect(self._compute_audio_changepoints)
        button_layout.addWidget(self.compute_cp_button)

        self.cp_count_label = QLabel("")
        button_layout.addWidget(self.cp_count_label)

        button_layout.addStretch()

        ref_label = QLabel(
            '<a href="https://github.com/timsainb/vocalization-segmentation" '
            'style="color: #87CEEB; text-decoration: none;">VocalSeg (Sainburg et al., 2020)</a>'
        )
        ref_label.setOpenExternalLinks(True)
        ref_label.setToolTip("Open vocalseg GitHub repository")
        button_layout.addWidget(ref_label)

        layout.addLayout(button_layout)

    def _create_ruptures_panel(self):
        self.ruptures_panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 0)
        self.ruptures_panel.setLayout(layout)

        params_group = QGroupBox("Ruptures detection parameters")
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

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
        self.ruptures_method_combo.currentTextChanged.connect(
            self._on_ruptures_method_changed
        )

        row = 0
        params_layout.addWidget(QLabel("Method:"), row, 0)
        params_layout.addWidget(self.ruptures_method_combo, row, 1, 1, 3)

        self.ruptures_model_combo = QComboBox()
        self.ruptures_model_combo.setToolTip(
            "Cost function for segment homogeneity:\n"
            "l1: Least absolute deviation\n"
            "l2: Least squared deviation (default)\n"
            "rbf: Radial basis function (kernel)\n"
            "linear: Linear regression\n"
            "normal: Gaussian likelihood\n"
            "ar: Autoregressive model"
        )
        self.ruptures_model_combo.addItems(["l2", "l1", "rbf", "linear", "normal", "ar"])

        row += 1
        params_layout.addWidget(QLabel("Model:"), row, 0)
        params_layout.addWidget(self.ruptures_model_combo, row, 1, 1, 3)

        self.ruptures_min_size_edit = QLineEdit("2")
        self.ruptures_min_size_edit.setToolTip("Minimum segment length (samples)")

        self.ruptures_jump_edit = QLineEdit("5")
        self.ruptures_jump_edit.setToolTip(
            "Subsampling factor (higher = faster but less precise)"
        )

        row += 1
        params_layout.addWidget(QLabel("Min size:"), row, 0)
        params_layout.addWidget(self.ruptures_min_size_edit, row, 1)
        params_layout.addWidget(QLabel("Jump:"), row, 2)
        params_layout.addWidget(self.ruptures_jump_edit, row, 3)

        self.ruptures_penalty_label = QLabel("Penalty:")
        self.ruptures_penalty_edit = QLineEdit("1.0")
        self.ruptures_penalty_edit.setToolTip(
            "Penalty value for adding a changepoint.\n"
            "Higher = fewer changepoints, Lower = more changepoints."
        )

        self.ruptures_n_bkps_label = QLabel("N breakpoints:")
        self.ruptures_n_bkps_edit = QLineEdit("5")
        self.ruptures_n_bkps_edit.setToolTip("Number of breakpoints to detect")

        row += 1
        params_layout.addWidget(self.ruptures_penalty_label, row, 0)
        params_layout.addWidget(self.ruptures_penalty_edit, row, 1)
        params_layout.addWidget(self.ruptures_n_bkps_label, row, 2)
        params_layout.addWidget(self.ruptures_n_bkps_edit, row, 3)

        self.ruptures_width_label = QLabel("Width:")
        self.ruptures_width_edit = QLineEdit("100")
        self.ruptures_width_edit.setToolTip(
            "Window width for Window method (samples)"
        )

        row += 1
        params_layout.addWidget(self.ruptures_width_label, row, 0)
        params_layout.addWidget(self.ruptures_width_edit, row, 1)

        self._on_ruptures_method_changed(self.ruptures_method_combo.currentText())

        button_layout = QHBoxLayout()

        self.compute_ruptures_button = QPushButton("Detect")
        self.compute_ruptures_button.setToolTip(
            "Detect changepoints for current feature using ruptures library"
        )
        self.compute_ruptures_button.clicked.connect(self._compute_ruptures_changepoints)
        button_layout.addWidget(self.compute_ruptures_button)

        self.cancel_ruptures_button = QPushButton("Cancel")
        self.cancel_ruptures_button.setToolTip("Cancel detection in progress")
        self.cancel_ruptures_button.clicked.connect(self._cancel_ruptures_detection)
        self.cancel_ruptures_button.hide()
        button_layout.addWidget(self.cancel_ruptures_button)

        self.clear_ruptures_button = QPushButton("Clear")
        self.clear_ruptures_button.setToolTip(
            "Remove all changepoints for current feature"
        )
        self.clear_ruptures_button.clicked.connect(self._clear_current_feature_changepoints)
        button_layout.addWidget(self.clear_ruptures_button)

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

    def _on_ruptures_method_changed(self, method: str):
        uses_penalty = method in ["Pelt", "Binseg"]
        self.ruptures_penalty_label.setVisible(uses_penalty)
        self.ruptures_penalty_edit.setVisible(uses_penalty)

        uses_n_bkps = method in ["Binseg", "BottomUp", "Window", "Dynp"]
        self.ruptures_n_bkps_label.setVisible(uses_n_bkps)
        self.ruptures_n_bkps_edit.setVisible(uses_n_bkps)

        uses_width = method == "Window"
        self.ruptures_width_label.setVisible(uses_width)
        self.ruptures_width_edit.setVisible(uses_width)

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    def set_enabled_state(self, enabled: bool):
        self.audio_changepoints_panel.setEnabled(enabled)
        self.audio_toggle.setEnabled(enabled)

        if not enabled and self.audio_toggle.isChecked():
            self.cp_toggle.setChecked(True)
            self._toggle_changepoints()

    def _restore_or_set_defaults(self):
        defaults = [
            ("audio_cp_hop_length_ms", self.hop_length_edit, 5.0),
            ("audio_cp_min_level_db", self.min_level_db_edit, -70.0),
            ("audio_cp_min_syllable_length_s", self.min_syllable_edit, 0.02),
            ("audio_cp_silence_threshold", self.silence_threshold_edit, 0.1),
            ("audio_cp_ref_level_db", self.ref_level_db_edit, 20),
        ]
        for attr, edit, default in defaults:
            value = getattr(self.app_state, attr, None)
            if value is None:
                value = default
                setattr(self.app_state, attr, value)
            edit.setText(str(value))

        spectral_range = getattr(self.app_state, "audio_cp_spectral_range", None)
        if spectral_range is not None:
            self.spectral_min_spin.setValue(spectral_range[0])
            self.spectral_max_spin.setValue(spectral_range[1])
        else:
            self.spectral_min_spin.setValue(0)
            self.spectral_max_spin.setValue(0)

        show_cp = getattr(self.app_state, "show_audio_changepoints", False)
        self.show_cp_checkbox.setChecked(show_cp)

    def _parse_float(self, text: str) -> Optional[float]:
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    def _parse_int(self, text: str) -> Optional[int]:
        try:
            return int(text)
        except (ValueError, TypeError):
            return None

    def _get_spectral_range(self) -> Optional[tuple]:
        min_val = self.spectral_min_spin.value()
        max_val = self.spectral_max_spin.value()
        if min_val == 0 and max_val == 0:
            return None
        if min_val == 0:
            min_val = None
        if max_val == 0:
            max_val = None
        if min_val is None and max_val is None:
            return None
        return (min_val, max_val)

    def _on_show_changepoints_changed(self, state):
        show = state == Qt.Checked
        self.app_state.show_changepoints = show

        if not show:
            self.changepoint_correction_checkbox.setChecked(False)

        if self.plot_container:
            if show:
                onsets = getattr(self.app_state, "audio_changepoint_onsets", None)
                offsets = getattr(self.app_state, "audio_changepoint_offsets", None)
                if onsets is not None and offsets is not None:
                    self.plot_container.draw_audio_changepoints(onsets, offsets)

                cp_by_method, time_array = self._get_dataset_changepoint_indices()
                if cp_by_method is not None:
                    self.plot_container.draw_dataset_changepoints(time_array, cp_by_method)
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

        from ethograph.utils.data_utils import get_time_coord

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

    def _compute_audio_changepoints(self):
        audio_path = getattr(self.app_state, "audio_path", None)
        if not audio_path:
            show_warning("No audio file loaded")
            return

        try:
            from ethograph.features.audio_changepoints import (
                detect_audio_changepoints,
                clear_audio_changepoint_cache,
            )

            self.compute_cp_button.setEnabled(False)
            self.compute_cp_button.setText("Computing...")

            from qtpy.QtWidgets import QApplication

            QApplication.processEvents()

            hop_length_ms = self._parse_float(self.hop_length_edit.text()) or 5.0
            min_level_db = self._parse_float(self.min_level_db_edit.text()) or -70.0
            min_syllable_s = self._parse_float(self.min_syllable_edit.text()) or 0.02
            silence_threshold = (
                self._parse_float(self.silence_threshold_edit.text()) or 0.1
            )
            ref_level_db = self._parse_float(self.ref_level_db_edit.text()) or 20
            spectral_range = self._get_spectral_range()
            n_fft = self.app_state.get_with_default("nfft")
            _, channel_idx = self.app_state.get_audio_source()

            self.app_state.audio_cp_hop_length_ms = hop_length_ms
            self.app_state.audio_cp_min_level_db = min_level_db
            self.app_state.audio_cp_min_syllable_length_s = min_syllable_s
            self.app_state.audio_cp_silence_threshold = silence_threshold
            self.app_state.audio_cp_ref_level_db = ref_level_db
            self.app_state.audio_cp_spectral_range = spectral_range

            clear_audio_changepoint_cache()

            onsets, offsets = detect_audio_changepoints(
                audio_path,
                hop_length_ms=hop_length_ms,
                min_level_db=min_level_db,
                min_syllable_length_s=min_syllable_s,
                silence_threshold=silence_threshold,
                ref_level_db=ref_level_db,
                spectral_range=spectral_range,
                n_fft=n_fft,
                channel_idx=channel_idx,
                use_cache=True,
            )

            self.app_state.audio_changepoint_onsets = onsets
            self.app_state.audio_changepoint_offsets = offsets

            self.cp_count_label.setText(f"{len(onsets)}+{len(offsets)}")
            show_info(f"Detected {len(onsets)} onsets, {len(offsets)} offsets")

            self.audio_changepoints_updated.emit()

            if self.plot_container:
                self.plot_container.draw_audio_changepoints(onsets, offsets)

            self.show_cp_checkbox.blockSignals(True)
            self.show_cp_checkbox.setChecked(True)
            self.show_cp_checkbox.blockSignals(False)
            self.app_state.show_changepoints = True
            self.request_plot_update.emit()

        except ImportError as e:
            show_warning(f"Missing dependency: {e}\nInstall with: pip install vocalseg")
        except Exception as e:
            show_warning(f"Error detecting changepoints: {e}")
        finally:
            self.compute_cp_button.setEnabled(True)
            self.compute_cp_button.setText("Detect")

    def is_changepoint_correction_enabled(self) -> bool:
        return self.changepoint_correction_checkbox.isChecked()

    def _on_method_changed(self, method: str):
        is_turning_points = method == "turning_points"
        self.threshold_label.setVisible(is_turning_points)
        self.threshold_edit.setVisible(is_turning_points)
        self.max_value_label.setVisible(is_turning_points)
        self.max_value_edit.setVisible(is_turning_points)

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

        new_ds = ds.drop_vars(vars_to_remove)

        trial = self.app_state.trials_sel
        trial_node = f"trial_{trial}"
        self.app_state.dt[trial_node] = xr.DataTree(new_ds)
        self.app_state.ds = self.app_state.dt.trial(trial)

        return len(vars_to_remove)

    def _compute_dataset_changepoints(self):
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            show_warning("No dataset loaded")
            return

        feature = getattr(self.app_state, "features_sel", None)
        if not feature:
            show_warning("No feature selected in Data Controls")
            return

        if feature not in ds.data_vars:
            show_warning(f"Feature '{feature}' not found in dataset")
            return

        from ethograph.utils.data_utils import sel_valid

        ds_kwargs = self.app_state.get_ds_kwargs()
        data, _ = sel_valid(ds[feature], ds_kwargs)

        if data.ndim > 1:
            show_warning(
                f"Changepoint detection only works on 1D variables (time,).\n"
                f"'{feature}' has shape {data.shape} after selection.\n"
                f"Please select specific values for all dimensions in Data Controls."
            )
            return

        method = self.method_combo.currentText()

        try:
            from ethograph.utils.io import add_changepoints_to_ds
            from ethograph.features.changepoints import (
                find_peaks_binary,
                find_troughs_binary,
                find_nearest_turning_points_binary,
            )

            self.compute_ds_cp_button.setEnabled(False)
            self.compute_ds_cp_button.setText("Computing...")

            from qtpy.QtWidgets import QApplication

            QApplication.processEvents()

            func_kwargs = {}

            prominence = self._parse_float(self.prominence_edit.text())
            if prominence is not None:
                func_kwargs["prominence"] = prominence

            distance = self._parse_int(self.distance_edit.text())
            if distance is not None:
                func_kwargs["distance"] = distance

            width = self._parse_int(self.width_edit.text())
            if width is not None:
                func_kwargs["width"] = width

            if method == "peaks":
                changepoint_func = find_peaks_binary
                changepoint_name = "peaks"
            elif method == "troughs":
                changepoint_func = find_troughs_binary
                changepoint_name = "troughs"
            else:
                changepoint_func = find_nearest_turning_points_binary
                changepoint_name = "turning_points"

                threshold = self._parse_float(self.threshold_edit.text())
                if threshold is not None:
                    func_kwargs["threshold"] = threshold

                max_value = self._parse_float(self.max_value_edit.text())
                if max_value is not None:
                    func_kwargs["max_value"] = max_value

            new_ds = add_changepoints_to_ds(
                ds=ds.copy(),
                target_feature=feature,
                changepoint_name=changepoint_name,
                changepoint_func=changepoint_func,
                **func_kwargs,
            )

            cp_var_name = f"{feature}_{changepoint_name}"

            trial = self.app_state.trials_sel
            trial_node = f"trial_{trial}"
            self.app_state.dt[trial_node] = xr.DataTree(new_ds)
            self.app_state.ds = self.app_state.dt.trial(trial)

            cp_data = new_ds[cp_var_name].values
            n_changepoints = np.sum(cp_data > 0)

            self.ds_cp_count_label.setText(f"{n_changepoints} changepoints")
            show_info(f"Added '{cp_var_name}' with {n_changepoints} changepoints")

            self.show_cp_checkbox.blockSignals(True)
            self.show_cp_checkbox.setChecked(True)
            self.show_cp_checkbox.blockSignals(False)
            self.app_state.show_changepoints = True

            if self.plot_container:
                cp_by_method, time_array = self._get_dataset_changepoint_indices()
                if cp_by_method is not None:
                    self.plot_container.draw_dataset_changepoints(time_array, cp_by_method)

            self.request_plot_update.emit()

        except Exception as e:
            show_warning(f"Error computing changepoints: {e}")
        finally:
            self.compute_ds_cp_button.setEnabled(True)
            self.compute_ds_cp_button.setText("Detect")

    def _compute_ruptures_changepoints(self):
        ds = getattr(self.app_state, "ds", None)
        if ds is None:
            show_warning("No dataset loaded")
            return

        feature = getattr(self.app_state, "features_sel", None)
        if not feature:
            show_warning("No feature selected in Data Controls")
            return

        if feature not in ds.data_vars:
            show_warning(f"Feature '{feature}' not found in dataset")
            return

        from ethograph.utils.data_utils import sel_valid

        ds_kwargs = self.app_state.get_ds_kwargs()
        data, _ = sel_valid(ds[feature], ds_kwargs)

        if data.ndim > 1:
            show_warning(
                f"Changepoint detection only works on 1D variables (time,).\n"
                f"'{feature}' has shape {data.shape} after selection.\n"
                f"Please select specific values for all dimensions in Data Controls."
            )
            return

        signal = np.asarray(data).reshape(-1, 1)
        method = self.ruptures_method_combo.currentText()
        model = self.ruptures_model_combo.currentText()
        min_size = self._parse_int(self.ruptures_min_size_edit.text()) or 2
        jump = self._parse_int(self.ruptures_jump_edit.text()) or 5

        params = {
            "penalty": self._parse_float(self.ruptures_penalty_edit.text()),
            "n_bkps": self._parse_int(self.ruptures_n_bkps_edit.text()),
            "width": self._parse_int(self.ruptures_width_edit.text()) or 100,
        }

        time_coord_name = self.app_state.time.name
        self._ruptures_context = (feature, time_coord_name, len(signal), method, model)

        self._set_ruptures_ui_running(True)

        self._ruptures_executor = ProcessPoolExecutor(max_workers=1)
        self._ruptures_future = self._ruptures_executor.submit(
            _run_ruptures_in_process,
            signal,
            method,
            model,
            min_size,
            jump,
            params,
        )

        self._ruptures_poll_timer = QTimer()
        self._ruptures_poll_timer.timeout.connect(self._poll_ruptures_result)
        self._ruptures_poll_timer.start(100)

    def _poll_ruptures_result(self):
        if self._ruptures_future is None:
            self._stop_polling()
            return

        if self._ruptures_future.done():
            self._stop_polling()
            try:
                result = self._ruptures_future.result()
                self._on_ruptures_finished(result)
            except Exception as e:
                self._on_ruptures_error(e)
            finally:
                self._cleanup_ruptures_executor()

    def _stop_polling(self):
        if self._ruptures_poll_timer is not None:
            self._ruptures_poll_timer.stop()
            self._ruptures_poll_timer = None

    def _cleanup_ruptures_executor(self):
        if self._ruptures_executor is not None:
            self._ruptures_executor.shutdown(wait=False, cancel_futures=True)
            self._ruptures_executor = None
        self._ruptures_future = None

    def _cancel_ruptures_detection(self):
        self._stop_polling()

        if self._ruptures_future is not None:
            self._ruptures_future.cancel()

        if self._ruptures_executor is not None:
            self._ruptures_executor.shutdown(wait=False, cancel_futures=True)
            self._ruptures_executor = None

        self._ruptures_future = None
        self._ruptures_context = None

        self._set_ruptures_ui_running(False)
        self.ruptures_count_label.setText("Cancelled")

    def _on_ruptures_error(self, exc: Exception):
        self._set_ruptures_ui_running(False)
        show_warning(f"Error computing ruptures changepoints: {exc}")
        self.ruptures_count_label.setText("")

    def _on_ruptures_finished(self, result: tuple[list[int] | None, str | None]):
        bkps, error_msg = result

        self._set_ruptures_ui_running(False)

        if error_msg:
            show_warning(f"Error computing ruptures changepoints: {error_msg}")
            self.ruptures_count_label.setText("")
            return

        if bkps is None or not self._ruptures_context:
            return

        feature, time_coord_name, signal_len, method, model = self._ruptures_context
        self._ruptures_context = None

        if bkps and bkps[-1] == signal_len:
            bkps = bkps[:-1]

        cp_array = np.zeros(signal_len, dtype=np.int8)
        for bkp in bkps:
            if 0 <= bkp < signal_len:
                cp_array[bkp] = 1

        ds = self.app_state.ds
        cp_var_name = f"{feature}_ruptures"

        new_ds = ds.copy()
        if cp_var_name in new_ds.data_vars:
            new_ds = new_ds.drop_vars(cp_var_name)

        new_ds[cp_var_name] = xr.Variable(
            dims=[time_coord_name],
            data=cp_array,
            attrs={
                "type": "changepoints",
                "target_feature": feature,
                "method": f"ruptures_{method}",
                "model": model,
            },
        )

        trial = self.app_state.trials_sel
        trial_node = f"trial_{trial}"
        self.app_state.dt[trial_node] = xr.DataTree(new_ds)
        self.app_state.ds = self.app_state.dt.trial(trial)

        n_changepoints = len(bkps)
        self.ruptures_count_label.setText(f"{n_changepoints} changepoints")
        show_info(f"Added '{cp_var_name}' with {n_changepoints} changepoints")

        self.show_cp_checkbox.blockSignals(True)
        self.show_cp_checkbox.setChecked(True)
        self.show_cp_checkbox.blockSignals(False)
        self.app_state.show_changepoints = True

        if self.plot_container:
            cp_by_method, time_array = self._get_dataset_changepoint_indices()
            if cp_by_method is not None:
                self.plot_container.draw_dataset_changepoints(time_array, cp_by_method)

        self.request_plot_update.emit()

    def _set_ruptures_ui_running(self, running: bool):
        if running:
            self.compute_ruptures_button.hide()
            self.cancel_ruptures_button.show()
            self.clear_ruptures_button.setEnabled(False)
            self.ruptures_count_label.setText("Computing...")
        else:
            self.cancel_ruptures_button.hide()
            self.compute_ruptures_button.show()
            self.compute_ruptures_button.setEnabled(True)
            self.clear_ruptures_button.setEnabled(True)

    def closeEvent(self, event):
        self._cancel_ruptures_detection()
        super().closeEvent(event)