#!/usr/bin/env python
"""Debug script for napari GUI."""

import os
import sys
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import napari
import json
from moveseg.gui.widgets_meta import MetaWidget
import moveseg
import matplotlib


def main():
    # Suppress Qt geometry warnings
    import logging
    import warnings

    # Filter out Qt geometry warnings
    warnings.filterwarnings("ignore", message=".*QWindowsWindow::setGeometry.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="vispy")

    # Reduce napari and vispy debug verbosity
    logging.getLogger("napari").setLevel(logging.ERROR)
    logging.getLogger("vispy").setLevel(logging.ERROR)
    logging.getLogger("qtpy").setLevel(logging.ERROR)

    

    matplotlib.set_loglevel("WARNING")

    # Suppress Qt warnings at the OS level
    os.environ["QT_LOGGING_RULES"] = "qt.*=false"

    # Start napari viewer
    viewer = napari.Viewer()

    # Add your plugin

    widget = MetaWidget(viewer)
    viewer.window.add_dock_widget(widget, name="moveseg GUI")

    print("DEBUG: Napari started with moveseg GUI")
    print("DEBUG: Use Ctrl+C in terminal to stop")

    # Run napari
    napari.run()


if __name__ == "__main__":
    main()
