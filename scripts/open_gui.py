#!/usr/bin/env python
"""Launch ethograph GUI for debugging."""

import napari
from ethograph.gui.widgets_meta import MetaWidget

if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(MetaWidget(viewer), name="ethograph GUI")
    napari.run()
