#!/usr/bin/env python
"""Command-line interface for ethograph."""

import sys



def launch():
    """Launch the ethograph GUI."""
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
    else:
        print(f"Unknown command: {command}")
        print("Available commands: launch")
        sys.exit(1)


if __name__ == "__main__":
    main()