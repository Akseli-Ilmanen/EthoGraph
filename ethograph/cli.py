#!/usr/bin/env python
"""Command-line interface for ethograph."""

import io
import os
import sys


def _get_log_path():
    log_dir = os.path.join(os.path.expanduser("~"), ".ethograph")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "ethograph.log")


def _fix_std_streams():
    """Redirect stdout/stderr to a log file when running via pythonw.exe."""
    if sys.stderr is not None and sys.stdout is not None:
        return
    log_file = open(_get_log_path(), "a")
    if sys.stdout is None:
        sys.stdout = log_file
    if sys.stderr is None:
        sys.stderr = log_file


_fix_std_streams()


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


def _show_error_dialog(title: str, message: str):
    """Show a Qt error dialog for fatal startup errors."""
    from qtpy.QtWidgets import QApplication, QMessageBox

    app = QApplication.instance() or QApplication(sys.argv)
    box = QMessageBox()
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    box.exec_()


def launch():
    """Launch the ethograph GUI."""
    try:
        from ethograph.shortcuts import ensure_shortcut_on_first_launch
        ensure_shortcut_on_first_launch()
        _ensure_qt_plugins()
        import napari
        from ethograph.gui.widgets_meta import MetaWidget

        viewer = napari.Viewer()
        viewer.window.add_dock_widget(
            MetaWidget(viewer), name="ethograph GUI"
        )
        napari.run()
    except Exception:
        import traceback

        tb = traceback.format_exc()
        try:
            sys.stderr.write(tb)
        except Exception:
            pass
        _show_error_dialog("ethograph - startup error", tb)
        sys.exit(1)


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