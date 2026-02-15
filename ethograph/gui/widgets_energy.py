"""Energy / envelope controls widget for configuring envelope metric and parameters."""

from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


ENERGY_METRICS = {
    "amplitude_envelope": "Amplitude envelope",
    "meansquared": "Meansquared energy",
}


class EnergyWidget(QWidget):
    """Controls for energy metric and envelope parameters.

    These are general-purpose settings shared by both the heatmap audio view
    and the envelope overlay on line plots.
    """

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None

        main_layout = QVBoxLayout()
        main_layout.setSpacing(2)
        main_layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(main_layout)

        self._create_controls(main_layout)
        self._restore_selections()
        self.setEnabled(False)

    def _create_controls(self, main_layout):
        group = QGroupBox("Energy / Envelope Controls")
        grid = QGridLayout()
        group.setLayout(grid)
        main_layout.addWidget(group)

        # Row 0: metric
        grid.addWidget(QLabel("Energy metric:"), 0, 0)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(ENERGY_METRICS.values())
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        grid.addWidget(self.metric_combo, 0, 1, 1, 3)

        # Row 1: amplitude envelope params (env_rate, cutoff)
        self.amp_label_rate = QLabel("Env rate (Hz):")
        grid.addWidget(self.amp_label_rate, 1, 0)
        self.env_rate_spin = QDoubleSpinBox()
        self.env_rate_spin.setRange(100.0, 44100.0)
        self.env_rate_spin.setSingleStep(100.0)
        self.env_rate_spin.setDecimals(0)
        self.env_rate_spin.setToolTip("Output sample rate for amplitude envelope")
        grid.addWidget(self.env_rate_spin, 1, 1)

        self.amp_label_cutoff = QLabel("Cutoff (Hz):")
        grid.addWidget(self.amp_label_cutoff, 1, 2)
        self.env_cutoff_spin = QDoubleSpinBox()
        self.env_cutoff_spin.setRange(10.0, 22050.0)
        self.env_cutoff_spin.setSingleStep(50.0)
        self.env_cutoff_spin.setDecimals(0)
        self.env_cutoff_spin.setToolTip("Lowpass cutoff for amplitude envelope")
        grid.addWidget(self.env_cutoff_spin, 1, 3)

        # Row 2: meansquared params (freq_cutoffs min/max)
        self.ms_label_fmin = QLabel("Freq min (Hz):")
        grid.addWidget(self.ms_label_fmin, 2, 0)
        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(0.0, 44100.0)
        self.freq_min_spin.setSingleStep(100.0)
        self.freq_min_spin.setDecimals(0)
        grid.addWidget(self.freq_min_spin, 2, 1)

        self.ms_label_fmax = QLabel("Freq max (Hz):")
        grid.addWidget(self.ms_label_fmax, 2, 2)
        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(0.0, 44100.0)
        self.freq_max_spin.setSingleStep(100.0)
        self.freq_max_spin.setDecimals(0)
        grid.addWidget(self.freq_max_spin, 2, 3)

        # Row 3: smooth_win + apply
        self.ms_label_smooth = QLabel("Smooth win:")
        grid.addWidget(self.ms_label_smooth, 3, 0)
        self.smooth_win_spin = QDoubleSpinBox()
        self.smooth_win_spin.setRange(0.1, 100.0)
        self.smooth_win_spin.setSingleStep(0.5)
        self.smooth_win_spin.setDecimals(1)
        self.smooth_win_spin.setToolTip("Smoothing window for meansquared energy")
        grid.addWidget(self.smooth_win_spin, 3, 1)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._on_apply)
        grid.addWidget(self.apply_button, 3, 2, 1, 2)

        # Collect conditional widgets for show/hide
        self._amp_widgets = [self.amp_label_rate, self.env_rate_spin, self.amp_label_cutoff, self.env_cutoff_spin]
        self._ms_widgets = [self.ms_label_fmin, self.freq_min_spin, self.ms_label_fmax, self.freq_max_spin,
                            self.ms_label_smooth, self.smooth_win_spin]

    def set_plot_container(self, plot_container):
        self.plot_container = plot_container

    def _restore_selections(self):
        metric = self.app_state.get_with_default("energy_metric")
        display = ENERGY_METRICS.get(metric, "Amplitude envelope")
        self.metric_combo.setCurrentText(display)

        self.env_rate_spin.setValue(self.app_state.get_with_default("env_rate"))
        self.env_cutoff_spin.setValue(self.app_state.get_with_default("env_cutoff"))
        self.freq_min_spin.setValue(self.app_state.get_with_default("freq_cutoffs_min"))
        self.freq_max_spin.setValue(self.app_state.get_with_default("freq_cutoffs_max"))
        self.smooth_win_spin.setValue(self.app_state.get_with_default("smooth_win"))

        self._update_conditional_visibility()

    def _on_metric_changed(self, _index):
        self._update_conditional_visibility()

    def _update_conditional_visibility(self):
        is_amplitude = self.metric_combo.currentText() == "Amplitude envelope"
        for w in self._amp_widgets:
            w.setVisible(is_amplitude)
        for w in self._ms_widgets:
            w.setVisible(not is_amplitude)

    def _display_to_internal(self, display_text: str) -> str:
        for key, val in ENERGY_METRICS.items():
            if val == display_text:
                return key
        return "amplitude_envelope"

    def _on_apply(self):
        self.app_state.energy_metric = self._display_to_internal(self.metric_combo.currentText())
        self.app_state.env_rate = self.env_rate_spin.value()
        self.app_state.env_cutoff = self.env_cutoff_spin.value()
        self.app_state.freq_cutoffs_min = self.freq_min_spin.value()
        self.app_state.freq_cutoffs_max = self.freq_max_spin.value()
        self.app_state.smooth_win = self.smooth_win_spin.value()

        if self.plot_container:
            heatmap = self.plot_container.heatmap_plot
            heatmap._clear_buffer()
            if self.plot_container.is_heatmap():
                heatmap.update_plot_content()

            if self.plot_container.overlay_manager.has_overlay('envelope'):
                self.plot_container._refresh_envelope_data()
