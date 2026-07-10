"""Context-sensitive right sidebar body.

The "Data" section of the sidebar is *minimal*: instead of showing every
control at once, it shows only the controls relevant to the plot the user last
clicked.  The relevant section widgets (group boxes / panels) are borrowed from
``DataPanel``, ``PlotSettingsWidget`` and ``NavigationWidget`` and re-parented
into a single :class:`RightContextPanel`; :meth:`set_context` then shows the
subset mapped to each plot type.

Mapping (per the design brief):

============  ==================================================
plot type     sections shown
============  ==================================================
``video``     Space/Cameras, Pose (if pose data), Playback
``audio``     Energy envelope, Spectrogram settings, shared axes
``lineplot``  Xarray coords, Overlays, Line-plot axes, shared axes
``heatmap``   Xarray coords, Overlays, Heatmap, shared axes
``space``     Xarray coords, Space-plot, shared axes
============  ==================================================

The sidebar refreshes only when the clicked plot *type* changes (not on every
click), and updates are skipped entirely in zen mode or when the Labels /
Navigation section is active (handled by the caller).
"""

from __future__ import annotations

from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

# plot type -> ordered list of section keys to show
_CONTEXT_MAP: dict[str, list[str]] = {
    # The old napari-era "Space/Cameras" group (slot) is gone — cameras are
    # opened by drag-drop and there is no layers/space-plot toggle.
    "video": ["pose", "playback"],
    # Audio trace: envelope controls + shared axes. Spectrogram: only its panel.
    "audiotrace": ["energy", "shared"],
    "audio": ["energy", "shared"],  # alias used for the default context
    "spectrogram": ["spectrogram"],
    "feature": ["coords", "lineplot", "shared"],
    "lineplot": ["coords", "lineplot", "shared"],
    "heatmap": ["coords", "heatmap", "shared"],
    # Space: its own X/Y/Z + 3D + space controls (now inside spaceplot_panel);
    # the lineplot "coords" group is intentionally excluded.
    "space": ["spaceplot", "shared"],
}


class RightContextPanel(QWidget):
    """Hosts all setting sections and shows only the clicked plot's subset."""

    def __init__(self, sections: dict[str, QWidget | None], parent=None):
        super().__init__(parent)
        self._sections = {k: v for k, v in sections.items() if v is not None}
        self._current: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        self._placeholder = QLabel("Click a plot or the video to show its settings.")
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet("color: rgba(255,255,255,140);")
        layout.addWidget(self._placeholder)

        for widget in self._sections.values():
            # Re-parent each borrowed section in and start hidden.
            layout.addWidget(widget)
            widget.setVisible(False)
        layout.addStretch()

    def set_context(self, plot_type: str, has_pose: bool = True) -> bool:
        """Show only the sections mapped to *plot_type*.

        Returns ``True`` if the context changed (so the caller can refresh the
        surrounding layout), ``False`` if it was already showing *plot_type*.
        """
        if plot_type == self._current:
            return False
        self._current = plot_type
        want = set(_CONTEXT_MAP.get(plot_type, []))
        if not has_pose:
            want.discard("pose")
        self._placeholder.setVisible(not want)
        for name, widget in self._sections.items():
            widget.setVisible(name in want)
        return True

    def current_context(self) -> str | None:
        return self._current
