#!/usr/bin/env python
"""Command-line interface for ethograph."""

import sys


def _ensure_qt_plugins():
    """Set QT_PLUGIN_PATH for conda-forge Qt installs (needed by menuinst shortcuts)."""
    import os
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
    _ensure_qt_plugins()
    print("Loading GUI...")
    print("\n")
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
        sys.exit(1)

    command = sys.argv[1]

    if command == "launch":
        launch()
    elif command == "shortcut":
        from ethograph.shortcuts import install_shortcut
        install_shortcut()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: launch, shortcut")
        sys.exit(1)


if __name__ == "__main__":
    main()