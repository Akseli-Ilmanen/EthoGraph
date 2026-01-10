#!/usr/bin/env python
"""Launch moveseg GUI with napari."""

import os
import sys
import logging
import warnings
from pathlib import Path
import napari
from moveseg.gui.widgets_meta import MetaWidget


def configure_logging(verbose: bool = False) -> None:
    """Configure logging and suppress unnecessary warnings."""
    # Set logging level based on verbosity
    log_level = logging.INFO if verbose else logging.WARNING
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress noisy libraries
    for logger_name in ['napari', 'vispy', 'qtpy', 'matplotlib']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Filter Qt warnings
    warnings.filterwarnings("ignore", message=".*QWindowsWindow::setGeometry.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="vispy")
    
    # Suppress Qt warnings at OS level
    os.environ["QT_LOGGING_RULES"] = "qt.*=false"


def launch_gui(config_path: str = None) -> None:
    """
    Launch moveseg GUI with napari viewer.
    
    Args:
        config_path: Optional path to configuration file
    """
    configure_logging()
    
    # Create viewer
    viewer = napari.Viewer(title="moveseg")
    
    # Initialize widget with optional config
    if config_path:
        widget = MetaWidget(viewer, config_path=config_path)
    else:
        widget = MetaWidget(viewer)
    
    # Add widget to viewer
    viewer.window.add_dock_widget(widget, name="moveseg GUI")
    
    # Start the application
    napari.run()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch moveseg GUI")
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file',
        default=None
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    try:
        launch_gui(config_path=args.config)
    except KeyboardInterrupt:
        print("\nShutting down moveseg GUI...")
    except Exception as e:
        logging.error(f"Failed to launch GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()