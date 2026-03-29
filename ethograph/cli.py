#!/usr/bin/env python
"""Command-line interface for ethograph."""

import logging
import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*__array__ implementation doesn't accept a copy keyword.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*The 'warn' method is deprecated.*"
)

def _ensure_qt_plugins():
    """Set QT_PLUGIN_PATH for conda-forge Qt installs (needed by menuinst shortcuts)."""
    if os.environ.get("QT_PLUGIN_PATH"):
        return
    candidates = [
        os.path.join(sys.prefix, "Library", "plugins"),        # Windows conda-forge
        os.path.join(sys.prefix, "lib", "qt5", "plugins"),     # Linux conda-forge
        os.path.join(sys.prefix, "lib", "qt", "plugins"),      # macOS conda-forge
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "platforms")):
            os.environ["QT_PLUGIN_PATH"] = path
            return


def launch():
    """Launch the ethograph GUI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("napari").setLevel(logging.WARNING)
    _ensure_qt_plugins()
    import napari
    from ethograph.gui.widgets_meta import MetaWidget

    viewer = napari.Viewer()
    viewer.window.add_dock_widget(MetaWidget(viewer), name="ethograph GUI")
    napari.run()


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: ethograph <command>")
        print("Commands:")
        print("  launch    Launch the ethograph GUI")
        print("  shortcut  Install desktop/Start Menu shortcut")
        sys.exit(1)

    command = sys.argv[1]

    if command == "launch":
        launch()
    elif command == "shortcut":
        from ethograph.shortcuts import install_shortcut
        sys.exit(install_shortcut())
    else:
        print(f"Unknown command: {command}")
        print("Available commands: launch, shortcut")
        sys.exit(1)


if __name__ == "__main__":
    main()
