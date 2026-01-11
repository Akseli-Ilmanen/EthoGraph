"""Enhanced plot widgets with sync mode awareness."""

from qtpy.QtWidgets import (QGridLayout, QLineEdit, QWidget, QPushButton, 
                            QVBoxLayout, QLabel, QCheckBox)
from napari.viewer import Viewer
from typing import Optional
from qtpy.QtGui import QDoubleValidator

class PlotsWidget(QWidget):
    """Plot controls with sync mode awareness.
    
    Keys used in gui_settings.yaml (via app_state):
      - ymin, ymax
      - spec_ymin, spec_ymax
      - window_size
      - audio_buffer
      - spec_buffer
    """

    def __init__(self, napari_viewer: Viewer, app_state, parent=None):
        super().__init__(parent=parent)
        self.app_state = app_state
        self.viewer = napari_viewer
        self.plot_container = None
        
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
    
        
        # Reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_to_defaults)
        main_layout.addWidget(reset_button)
        
        # Grid layout for controls in 2 columns
        layout = QGridLayout()
        main_layout.addLayout(layout)

        self.ymin_edit = QLineEdit()
        self.ymax_edit = QLineEdit()
        self.spec_ymin_edit = QLineEdit()
        self.spec_ymax_edit = QLineEdit()
        
        self.percentile_ylim_edit = QLineEdit()
        validator = QDoubleValidator(95.0, 100, 2, self)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.percentile_ylim_edit.setValidator(validator)
        
        self.window_s_edit = QLineEdit()
        self.audio_buffer_edit = QLineEdit()
        self.spec_buffer_edit = QLineEdit()
        
        self.apply_button = QPushButton("Apply current settings")
        self.apply_button.setToolTip("Shortcut: Ctrl + Enter")

        self.autoscale_checkbox = QCheckBox("Autoscale (y-axis)")
        self.autoscale_checkbox.setToolTip("Shortcut: Ctrl + A")
        self.lock_axes_checkbox = QCheckBox("Lock Axes")
        self.lock_axes_checkbox.setToolTip("Shortcut: Ctrl + L")

        
        


        # Arrange controls in pairs (2 columns)
        # Row 0: Y min/max (lineplot)
        layout.addWidget(QLabel("Y min (lineplot):"), 0, 0)
        layout.addWidget(self.ymin_edit, 0, 1)
        layout.addWidget(QLabel("Y max (lineplot):"), 0, 2)
        layout.addWidget(self.ymax_edit, 0, 3)
        
        # Row 1: Y min/max (spectrogram)
        layout.addWidget(QLabel("Y min (spectrogram):"), 1, 0)
        layout.addWidget(self.spec_ymin_edit, 1, 1)
        layout.addWidget(QLabel("Y max (spectrogram):"), 1, 2)
        layout.addWidget(self.spec_ymax_edit, 1, 3)
        
        # Row 2: Percentile Y-limits
        layout.addWidget(QLabel("Percentile zoom out y-limits:"), 2, 0)
        layout.addWidget(self.percentile_ylim_edit, 2, 1)
        
        layout.addWidget(QLabel("Window size (s):"), 3, 0)
        layout.addWidget(self.window_s_edit, 3, 1)

        layout.setRowMinimumHeight(4, 10)

        layout.addWidget(self.autoscale_checkbox, 5, 0)
        layout.addWidget(self.lock_axes_checkbox, 5, 1)
        layout.addWidget(self.apply_button, 5, 2)

        
        layout.setRowMinimumHeight(6, 10)

        # Row 7: Audio buffer / Spectrogram buffer
        layout.addWidget(QLabel("Audio buffer (s):"), 7, 0)
        layout.addWidget(self.audio_buffer_edit, 7, 1)
        layout.addWidget(QLabel("Spectrogram buffer (x):"), 7, 2)
        layout.addWidget(self.spec_buffer_edit, 7, 3)


        # Connect edit signals
        self.ymin_edit.editingFinished.connect(self._on_edited)
        self.ymax_edit.editingFinished.connect(self._on_edited)
        self.spec_ymin_edit.editingFinished.connect(self._on_edited)
        self.spec_ymax_edit.editingFinished.connect(self._on_edited)
        self.percentile_ylim_edit.editingFinished.connect(self._on_edited)
        self.window_s_edit.editingFinished.connect(self._on_edited)
        self.audio_buffer_edit.editingFinished.connect(self._on_edited)
        self.spec_buffer_edit.editingFinished.connect(self._on_edited)
        
        self.apply_button.clicked.connect(self._on_edited)
        self.autoscale_checkbox.toggled.connect(self._autoscale_y_toggle)
        self.lock_axes_checkbox.toggled.connect(self._on_lock_axes_toggled)

        
        # Load initial settings
        self._restore_or_set_default_selections()


    def set_plot_container(self, plot_container):
        """Set reference to plot container widget."""
        self.plot_container = plot_container
        


    def _restore_or_set_default_selections(self):
        """Restore selections from app_state or set defaults."""
        for attr, edit in [
            ("ymin", self.ymin_edit),
            ("ymax", self.ymax_edit),
            ("spec_ymin", self.spec_ymin_edit),
            ("spec_ymax", self.spec_ymax_edit),
            ("percentile_ylim", self.percentile_ylim_edit),
            ("window_size", self.window_s_edit),
            ("audio_buffer", self.audio_buffer_edit),
            ("spec_buffer", self.spec_buffer_edit)
        ]:
            value = getattr(self.app_state, attr, None)
            if value is None:
                value = self.app_state.get_with_default(attr)
                setattr(self.app_state, attr, value)
            edit.setText("" if value is None else str(value))
        
        # Restore lock axes checkbox state
        lock_axes = self.app_state.get_with_default("lock_axes")
        self.lock_axes_checkbox.setChecked(lock_axes)

    def _parse_float(self, text: str) -> Optional[float]:
        """Parse float from text input."""
        s = (text or "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
        
    def _autoscale_y_toggle(self, checked: bool):
        """Autoscale y-axis based on current data."""
        if not self.plot_container:
            return

        if checked:
            self.plot_container.vb.enableAutoRange(x=False, y=True)
            self.lock_axes_checkbox.setChecked(False)
        else:
            self.plot_container.vb.disableAutoRange()

    def _on_lock_axes_toggled(self, checked: bool):
        """Handle lock axes checkbox toggle."""
        self.app_state.lock_axes = checked
        self.plot_container.toggle_axes_lock()
        if checked:
            self.autoscale_checkbox.setChecked(False)
        

    def _on_edited(self):
        """Handle user edits to input fields."""
        if not self.plot_container:
            return


        edits = {
            "ymin": self.ymin_edit,
            "ymax": self.ymax_edit,
            "spec_ymin": self.spec_ymin_edit,
            "spec_ymax": self.spec_ymax_edit,
            "percentile_ylim": self.percentile_ylim_edit,
            "window_size": self.window_s_edit,
            "audio_buffer": self.audio_buffer_edit,
            "spec_buffer": self.spec_buffer_edit
        }
        
        values = {}
        for attr, edit in edits.items():
            val = self._parse_float(edit.text())
            if val is None:
                val = self.app_state.get_with_default(attr)
            values[attr] = val
            if self.app_state is not None:
                setattr(self.app_state, attr, val)

        # Update spectrogram buffer settings if spec_buffer was changed
        if "spec_buffer" in values and self.plot_container.is_spectrogram():
            # Update spectrogram buffer settings when showing spectrogram
            current_plot = self.plot_container.get_current_plot()
            if hasattr(current_plot, 'update_buffer_settings'):
                current_plot.update_buffer_settings()

        # Update y-range based on current plot type
        if self.plot_container.is_spectrogram():
            self.plot_container.apply_y_range(
                values["spec_ymin"],
                values["spec_ymax"]
            )
        else:
            self.plot_container.apply_y_range(
                values["ymin"],
                values["ymax"]
            )
            
        # If percentile_ylim changed, update zoom constraints
        if "percentile_ylim" in values:
            current_plot = self.plot_container.get_current_plot()
            if hasattr(current_plot, '_apply_zoom_constraints'):
                current_plot._apply_zoom_constraints()
            
        new_xmin, new_xmax = self._calculate_new_window_size()
        self.plot_container.set_x_range(mode='preserve', curr_xlim=(new_xmin, new_xmax))



    def _calculate_new_window_size(self):
        """Calculate new window size preserving center and current x-limits."""
        if not self.plot_container:
            return

        self.current_time = self.app_state.current_frame / self.app_state.ds.fps
        window_size = self.app_state.get_with_default("window_size")
        half_window = window_size / 2

        new_xmin = self.current_time - half_window
        new_xmax = self.current_time + half_window
        return new_xmin, new_xmax


    def _reset_to_defaults(self):
        """Reset all plot values to defaults."""
        for attr, edit in [
            ("ymin", self.ymin_edit),
            ("ymax", self.ymax_edit),
            ("spec_ymin", self.spec_ymin_edit),
            ("spec_ymax", self.spec_ymax_edit),
            ("percentile_ylim", self.percentile_ylim_edit),
            ("window_size", self.window_s_edit),
            ("audio_buffer", self.audio_buffer_edit),
            ("spec_buffer", self.spec_buffer_edit)
        ]:
            value = self.app_state.get_with_default(attr)
            edit.setText("" if value is None else str(value))
            if self.app_state is not None:
                setattr(self.app_state, attr, value)
        
        # Reset lock axes checkbox
        self.lock_axes_checkbox.setChecked(False)
        self.app_state.lock_axes = False
        
        self._on_edited()

