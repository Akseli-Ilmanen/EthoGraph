"""Energy envelope controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .widgets_data import DataWidget

from .dialog_function_params import open_function_params_dialog


ENERGY_DISPLAY_NAMES = {
    "energy_lowpass": "SOS lowpass envelope",
    "energy_highpass": "SOS highpass envelope",
    "energy_band": "SOS bandpass envelope",
    "energy_meansquared": "Vocalpy meansquared (amplitude)",
    "energy_ava": "Vocalpy AVA (spectral power)",
}


def compute_energy_envelope(
    data: np.ndarray, rate: float, metric: str, app_state,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute energy envelope using registry-driven dispatch.

    Looks up the wrapper function and cached user params for the given metric,
    then returns (env_time, envelope).
    """
    import inspect

    from ethograph.features.energy import (
        bandpass_envelope,
        env_ava,
        env_meansquared,
        highpass_envelope,
        lowpass_envelope,
    )

    _METRIC_FUNCS = {
        "energy_lowpass": lowpass_envelope,
        "energy_highpass": highpass_envelope,
        "energy_band": bandpass_envelope,
        "energy_meansquared": env_meansquared,
        "energy_ava": env_ava,
    }

    func = _METRIC_FUNCS.get(metric, lowpass_envelope)
    registry_key = metric

    cache = getattr(app_state, "function_params_cache", None) or {}
    cached = cache.get(registry_key, {})

    sig = inspect.signature(func)
    valid_keys = set(sig.parameters) - {"data", "rate"}
    params = {k: v for k, v in cached.items() if k in valid_keys}

    return func(data, rate, **params)


class TransformWidget(QWidget):
    """Energy envelope controls."""

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None
        self.meta_widget = None
        self.data_widget: DataWidget | None = None

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_energy_panel(main_layout)
        self._restore_energy_selections()
        self.setEnabled(False)

    # ------------------------------------------------------------------
    # Energy envelopes panel
    # ------------------------------------------------------------------

    def _create_energy_panel(self, main_layout):
        self.energy_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.energy_panel.setLayout(layout)

        group = QGroupBox("Create energy envelope")
        grid = QGridLayout()
        group.setLayout(grid)
        layout.addWidget(group)

        grid.addWidget(QLabel("Energy metric:"), 0, 0)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(ENERGY_DISPLAY_NAMES.values())
        grid.addWidget(self.metric_combo, 0, 1, 1, 2)

        self.energy_configure_btn = QPushButton("Configure...")
        self.energy_configure_btn.setToolTip("Open parameter editor for selected energy metric")
        self.energy_configure_btn.clicked.connect(self._open_energy_params)
        grid.addWidget(self.energy_configure_btn, 1, 0, 1, 3)

        self.envelope_target_label = QLabel("Compute for:")
        self.envelope_target_combo = QComboBox()
        self.envelope_target_combo.addItems(["Audio trace", "Ephys trace"])
        self.envelope_target_combo.setToolTip(
            "Choose which trace displays the envelope:\n"
            "Audio trace — computes from audio waveform\n"
            "Ephys trace — computes from ephys waveform (single channel only)"
        )
        self.envelope_target_label.hide()
        self.envelope_target_combo.hide()
        grid.addWidget(self.envelope_target_label, 2, 0)
        grid.addWidget(self.envelope_target_combo, 2, 1, 1, 2)

        main_layout.addWidget(self.energy_panel)

    def _open_energy_params(self):
        key = self._display_to_key(self.metric_combo.currentText())
        if key:
            result = open_function_params_dialog(key, self.app_state, parent=self)
            if result is not None:
                self._on_energy_apply()

    def _restore_energy_selections(self):
        metric = self.app_state.get_with_default("energy_metric")
        display = ENERGY_DISPLAY_NAMES.get(metric, "SOS lowpass envelope")
        self.metric_combo.setCurrentText(display)

    def _display_to_key(self, display_text: str) -> str:
        for key, val in ENERGY_DISPLAY_NAMES.items():
            if val == display_text:
                return key
        return "energy_lowpass"

    def _on_energy_apply(self):
        metric_key = self._display_to_key(self.metric_combo.currentText())
        self.app_state.energy_metric = metric_key

        if not self.plot_container:
            return

        from .dialog_busy_progress import BusyProgressDialog

        pc = self.plot_container

        if (
            self.data_widget is not None
            and hasattr(self.data_widget, 'show_envelope_checkbox')
            and not self.data_widget.show_envelope_checkbox.isChecked()
        ):
            self.data_widget.show_envelope_checkbox.setChecked(True)
            return

        pc.hide_envelope_overlay()
        dialog = BusyProgressDialog("Computing energy envelope...", parent=self)
        dialog.execute_blocking(pc.show_envelope_overlay)

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def set_meta_widget(self, meta_widget):
        self.meta_widget = meta_widget

    def show_envelope_target_combo(self):
        """Show the envelope target combo (called after data load)."""
        self.envelope_target_label.show()
        self.envelope_target_combo.show()
        if not hasattr(self, '_envelope_target_connected'):
            self.envelope_target_combo.currentIndexChanged.connect(self._on_envelope_target_changed)
            self._envelope_target_connected = True
        self._on_envelope_target_changed()

    def _on_envelope_target_changed(self):
        text = self.envelope_target_combo.currentText()
        target = "audio" if text == "Audio trace" else "ephys"
        self.app_state._envelope_target = target

    def get_envelope_target(self) -> str:
        """Return 'audio' or 'ephys' based on combo selection."""
        if self.envelope_target_combo.isVisible():
            text = self.envelope_target_combo.currentText()
            return "audio" if text == "Audio trace" else "ephys"
        return "audio"

    def set_enabled_state(self, has_audio: bool = False):
        self.setEnabled(True)
