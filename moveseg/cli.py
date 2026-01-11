#!/usr/bin/env python
"""Command-line interface for MoveSeg."""

import sys



def launch():
    """Launch the MoveSeg GUI."""
    print("Loading GUI...")
    print("\n")
    import napari
    from moveseg.gui.widgets_meta import MetaWidget
    
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(MetaWidget(viewer), name="MoveSeg GUI")
    napari.run()


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: moveseg <command>")
        print("Commands:")
        print("  launch    Launch the MoveSeg GUI")
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