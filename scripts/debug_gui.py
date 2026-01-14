#!/usr/bin/env python
"""Debug script for napari GUI."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import napari
from ethograph.gui.widgets_meta import MetaWidget


def main():
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(MetaWidget(viewer), name="EthoGraph GUI")
    napari.run()


if __name__ == "__main__":
    main()