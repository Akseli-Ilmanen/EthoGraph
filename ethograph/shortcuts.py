from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
ICON_EXTENSIONS = {"win32": ".ico", "linux": ".png", "darwin": ".png"}


def _is_conda_env() -> bool:
    return bool(os.environ.get("CONDA_PREFIX"))


def _find_conda() -> str | None:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe
    found = shutil.which("conda")
    if found:
        return found
    return None


def _ensure_menuinst() -> None:
    try:
        import menuinst  # noqa: F401
    except ImportError:
        conda = _find_conda()
        if conda is None:
            raise FileNotFoundError("conda executable not found on PATH")
        print("menuinst not found — installing into current conda environment...")
        subprocess.check_call(
            [conda, "install", "-y", "--prefix", sys.prefix, "menuinst"],
        )


def _get_menu_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    if sys.platform == "darwin":
        return Path.home() / "Applications"
    return Path.home() / ".local" / "share" / "applications"


def _prepare_menu_files() -> Path:
    menu_json = Path.home() / ".ethograph" / "menu" / "menu.json"
    menu_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSETS_DIR / "menu.json", menu_json)

    menu_dir = _get_menu_dir()
    menu_dir.mkdir(parents=True, exist_ok=True)
    ext = ICON_EXTENSIONS.get(sys.platform)
    if ext:
        icon_src = ASSETS_DIR / f"icon{ext}"
        if icon_src.exists():
            shutil.copy2(icon_src, menu_dir / icon_src.name)

    return menu_json


_SHORTCUT_MARKER = Path.home() / ".ethograph" / ".shortcut_installed"


def shortcut_exists() -> bool:
    return _SHORTCUT_MARKER.exists()


def ensure_shortcut_on_first_launch() -> None:
    if not _is_conda_env() or shortcut_exists():
        return
    try:
        install_shortcut()
    except Exception:
        pass


def install_shortcut() -> int:
    if not _is_conda_env():
        print(
            "Error: Desktop shortcuts require a conda environment.\n"
            "\n"
            "To launch ethograph:\n"
            "  1. Activate your environment:  conda activate ethograph\n"
            "  2. Run:                         ethograph launch\n"
        )
        return 1

    print("Installing shortcut...")

    try:
        _ensure_menuinst()
        import menuinst

        menu_file = _prepare_menu_files()
        menuinst.install(str(menu_file), prefix=sys.prefix)
        _SHORTCUT_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _SHORTCUT_MARKER.touch()
        print("Shortcut installed successfully.")
        return 0
    except subprocess.CalledProcessError:
        print("Failed to install menuinst. Run: conda install menuinst")
        return 1
    except Exception as e:
        print(f"Shortcut creation failed: {e}")
        return 1
