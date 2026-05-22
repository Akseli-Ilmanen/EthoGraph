#!/usr/bin/env python
"""Command-line interface for ethograph."""

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
    if os.environ.get("QT_PLUGIN_PATH"):
        return
    candidates = [
        os.path.join(sys.prefix, "Library", "plugins"),  # Windows conda-forge
        os.path.join(sys.prefix, "lib", "qt6", "plugins"),  # Linux conda-forge Qt6
        os.path.join(sys.prefix, "lib", "qt5", "plugins"),  # Linux conda-forge Qt5 (fallback)
        os.path.join(sys.prefix, "lib", "qt", "plugins"),  # macOS conda-forge
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "platforms")):
            os.environ["QT_PLUGIN_PATH"] = path
            return


def _fix_wayland_opengl():
    """Force XCB/GLX on GNOME+Wayland to avoid OpenGL crashes with napari.

    See https://github.com/Akseli-Ilmanen/ethograph/issues/1
    """
    if sys.platform != "linux":
        return
    wayland_active = bool(os.environ.get("WAYLAND_DISPLAY")) or (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    )
    if not wayland_active:
        return
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
    if "GNOME" not in desktop:
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")


def launch():
    """Launch the ethograph GUI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("napari").setLevel(logging.WARNING)
    _fix_wayland_opengl()
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
