#!/usr/bin/env python
"""Command-line interface for ethograph."""

<<<<<<< HEAD
import sys


def _ensure_qt_plugins():
    """Set QT_PLUGIN_PATH for conda-forge Qt installs (needed by menuinst shortcuts)."""
    import os
=======
import logging
import os
import sys
import warnings

# Suppress noisy dependency warnings before any imports trigger them
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"logging")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"vispy\.")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"numpy\.")

# PyOpenGL info message goes through logging, not warnings
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)

def _ensure_qt_plugins():
    """Set QT_PLUGIN_PATH for conda-forge Qt installs (needed by menuinst shortcuts)."""
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
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
<<<<<<< HEAD
    _ensure_qt_plugins()
    print("Loading GUI...")
    print("\n")
=======
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("napari").setLevel(logging.WARNING)
    _ensure_qt_plugins()
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
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
<<<<<<< HEAD
=======
        print("  shortcut  Install desktop/Start Menu shortcut")
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
        sys.exit(1)

    command = sys.argv[1]

    if command == "launch":
        launch()
    elif command == "shortcut":
        from ethograph.shortcuts import install_shortcut
<<<<<<< HEAD
        install_shortcut()
=======
        sys.exit(install_shortcut())
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    else:
        print(f"Unknown command: {command}")
        print("Available commands: launch, shortcut")
        sys.exit(1)


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
