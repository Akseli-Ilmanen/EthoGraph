"""Energy envelopes and noise removal controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from .widgets_data import DataWidget

from .dialog_function_params import open_function_params_dialog
from .makepretty import styled_link


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
    """Energy envelopes and noise removal controls."""

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

        self._create_toggle_buttons(main_layout)
        self._create_energy_panel(main_layout)
        self._create_noise_panel(main_layout)

        self._restore_energy_selections()
        self._restore_noise_selections()

        self._show_panel("energy")
        self.setEnabled(False)

    # ------------------------------------------------------------------
    # Toggle buttons
    # ------------------------------------------------------------------

    def _create_toggle_buttons(self, main_layout):
        toggle_widget = QWidget()
        toggle_layout = QHBoxLayout()
        toggle_layout.setSpacing(2)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_widget.setLayout(toggle_layout)

        toggle_defs = [
            ("energy_toggle", "Energy envelopes", self._toggle_energy),
            ("noise_toggle", "Noise removal", self._toggle_noise),
        ]
        for attr, label, callback in toggle_defs:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(callback)
            toggle_layout.addWidget(btn)
            setattr(self, attr, btn)

        main_layout.addWidget(toggle_widget)

    def _show_panel(self, panel_name: str):
        panels = {
            "energy": (self.energy_panel, self.energy_toggle),
            "noise": (self.noise_panel, self.noise_toggle),
        }
        for name, (panel, toggle) in panels.items():
            if name == panel_name:
                panel.show()
                toggle.setChecked(True)
            else:
                panel.hide()
                toggle.setChecked(False)
        self._refresh_layout()

    def _toggle_energy(self):
        self._show_panel("energy" if self.energy_toggle.isChecked() else "noise")

    def _toggle_noise(self):
        self._show_panel("noise" if self.noise_toggle.isChecked() else "energy")

    def _refresh_layout(self):
        if self.meta_widget:
            self.meta_widget.refresh_widget_layout(self)

    # ------------------------------------------------------------------
    # Energy envelopes panel — simplified with Configure... button
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

        self.envelope_target_label = QLabel("Show envelope on:")
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

    # ------------------------------------------------------------------
    # Noise removal panel (noisereduce only — ephys preprocessing moved to EphysWidget)
    # ------------------------------------------------------------------

    def _create_noise_panel(self, main_layout):
        self.noise_panel = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        self.noise_panel.setLayout(layout)

        nr_group = QGroupBox("noisereduce")
        nr_layout = QGridLayout()
        nr_group.setLayout(nr_layout)
        layout.addWidget(nr_group)

        self.noise_reduce_checkbox = QCheckBox("Enable")
        self.noise_reduce_checkbox.setToolTip(
            "Apply spectral gating noise reduction to audio.\n"
            "Affects spectrogram and waveform display."
        )
        self.noise_reduce_checkbox.stateChanged.connect(self._on_noise_reduce_changed)

        self.noise_configure_btn = QPushButton("Configure...")
        self.noise_configure_btn.setToolTip("Configure noisereduce parameters")
        self.noise_configure_btn.clicked.connect(self._open_noise_params)

        ref_label = QLabel(styled_link(
            "https://github.com/timsainb/noisereduce",
            "noisereduce (Sainburg, 2020)",
        ))
        ref_label.setOpenExternalLinks(True)

        nr_layout.addWidget(self.noise_reduce_checkbox, 0, 0)
        nr_layout.addWidget(self.noise_configure_btn, 0, 1)
        nr_layout.addWidget(ref_label, 0, 2)

        main_layout.addWidget(self.noise_panel)

    def _open_noise_params(self):
        open_function_params_dialog("noise_reduction", self.app_state, parent=self)

    def _restore_noise_selections(self):
        noise_reduce = getattr(self.app_state, 'noise_reduce_enabled', None)
        if noise_reduce is None:
            noise_reduce = self.app_state.get_with_default('noise_reduce_enabled')
            self.app_state.noise_reduce_enabled = noise_reduce
        self.noise_reduce_checkbox.setChecked(noise_reduce)

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
        self.noise_reduce_checkbox.setEnabled(has_audio)
        self.noise_configure_btn.setEnabled(has_audio)

    def _on_noise_reduce_changed(self, state=None):
        self.app_state.noise_reduce_enabled = self.noise_reduce_checkbox.isChecked()

        if not self.plot_container:
            return

        from .dialog_busy_progress import BusyProgressDialog

        def _apply():
            self.plot_container.clear_audio_cache()
            self.plot_container.update_audio_panels()

        dialog = BusyProgressDialog("Applying noise reduction...", parent=self)
        dialog.execute_blocking(_apply)
