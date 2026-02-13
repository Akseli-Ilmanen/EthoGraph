from __future__ import annotations

import os
import stat
import sys
import shutil
from pathlib import Path

ICON_EXTENSIONS = {"win32": ".ico", "linux": ".png", "darwin": ".png"}


def _platform_key() -> str:
    return {"win32": "win", "linux": "linux", "darwin": "osx"}.get(sys.platform, "")


def install_shortcut() -> int:
    print("Installing shortcut...")

    platform = _platform_key()
    if not platform:
        print(f"Unsupported platform: {sys.platform}")
        return 1

    try:
        from menuinst import install
    except ImportError:
        print("Install with: conda install menuinst")
        return 1

    assets = Path(__file__).parent / "assets"
    menu_json = assets / "menu.json"
    icon_ext = ICON_EXTENSIONS[sys.platform]
    icon = assets / f"icon{icon_ext}"

    if not menu_json.exists():
        print(f"Error: menu.json not found in {assets}")
        return 1
    if not icon.exists():
        print(f"Error: icon{icon_ext} not found in {assets}")
        return 1

    menu_dir = Path(sys.prefix) / "Menu"
    menu_dir.mkdir(exist_ok=True)

    target_json = menu_dir / "ethograph.json"
    target_icon = menu_dir / f"icon{icon_ext}"

    try:
        for src, dst in [(menu_json, target_json), (icon, target_icon)]:
            if dst.exists():
                dst.chmod(stat.S_IWRITE | stat.S_IREAD)
                dst.unlink()
            shutil.copy(src, dst)
    except PermissionError as e:
        print(f"Warning: Could not update files in {menu_dir}: {e}")
        print("Attempting to continue with existing files...")

    try:
        install(str(target_json))
        print("Shortcut installed successfully")
        return 0
    except Exception as e:
        print(f"Error installing shortcut: {e}")
        return 1
