#!/usr/bin/env python
"""Debug script for napari GUI."""

import sys
from pathlib import Path

# Add project root to path (where ethograph/ folder is)
sys.path.insert(0, str(Path(__file__).parent.parent))

import napari
from ethograph.gui.widgets_meta import MetaWidget


def main():
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(MetaWidget(viewer), name="EthoGraph GUI")
    napari.run()


if __name__ == "__main__":
    main()