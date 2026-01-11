# moveseg/shortcuts.py
from __future__ import annotations

import sys
import shutil
from pathlib import Path


def install_shortcut() -> int:
    print("Installing shortcut...")
    if sys.platform != "win32":
        print("Shortcuts only supported on Windows")
        return 0

    try:
        from menuinst import install
    except ImportError:
        print("Install with: conda install menuinst")
        return 1

    assets = Path(__file__).parent / "assets"
    menu_json = assets / "menu.json"
    icon = assets / "icon.ico"

    if not icon.exists() or not menu_json.exists():
        print(f"Error: Required assets not found in {assets}")
        return 1

    menu_dir = Path(sys.prefix) / "Menu"
    menu_dir.mkdir(exist_ok=True)

    target_icon = menu_dir / "icon.ico"
    target_json = menu_dir / "moveseg.json"

    try:
        if target_icon.exists():
            target_icon.unlink()
        if target_json.exists():
            target_json.unlink()

        shutil.copy(icon, target_icon)
        shutil.copy(menu_json, target_json)
    except PermissionError as e:
        print(f"Warning: Could not update files in {menu_dir}")
        print(f"Files may be in use. Error: {e}")
        print("Attempting to continue with existing files...")

    try:
        install(str(target_json))
        print("Shortcut installed to Start Menu")
        return 0
    except Exception as e:
        print(f"Error installing shortcut: {e}")
        return 1