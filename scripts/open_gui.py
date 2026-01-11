#!/usr/bin/env python
"""Launch MoveSeg GUI for debugging."""

import napari
from moveseg.gui.widgets_meta import MetaWidget

if __name__ == "__main__":
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(MetaWidget(viewer), name="MoveSeg GUI")
    napari.run()
