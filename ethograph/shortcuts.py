from __future__ import annotations

import os
<<<<<<< HEAD
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
=======
import shutil
import stat
import sys
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
ICON_EXTENSIONS = {"win32": ".ico", "linux": ".png", "darwin": ".png"}


def _build_bat_content() -> str:
    prefix = Path(sys.prefix)
    python = prefix / "python.exe"
    lines = [
        "@ECHO OFF",
        "TITLE ethograph",
    ]

    conda_bat = _find_conda_bat()
    venv_activate = prefix / "Scripts" / "activate.bat"

    if conda_bat:
        lines.append(f'CALL "{conda_bat}" activate "{prefix}"')
    elif venv_activate.exists():
        lines.append(f'CALL "{venv_activate}"')
    else:
        extra_dirs = [
            str(prefix),
            str(prefix / "Library" / "bin"),
            str(prefix / "Scripts"),
        ]
        lines.append(f'SET "PATH={";".join(extra_dirs)};%PATH%"')

    lines += [
        f'"{python}" -m ethograph launch',
        "ECHO.",
        "ECHO ethograph exited. Press any key to close...",
        "PAUSE >NUL",
    ]
    return "\r\n".join(lines) + "\r\n"


def _find_conda_bat() -> Path | None:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidate = Path(conda_exe).parent.parent / "condabin" / "conda.bat"
        if candidate.exists():
            return candidate
    for base in [Path(sys.prefix).parents[1], Path(os.environ.get("CONDA_PREFIX", ""))]:
        candidate = base / "condabin" / "conda.bat"
        if candidate.exists():
            return candidate
    return None


def _create_bat_launcher() -> Path:
    menu_dir = Path(sys.prefix) / "Menu"
    menu_dir.mkdir(parents=True, exist_ok=True)
    bat_path = menu_dir / "ethograph.bat"
    bat_path.write_text(_build_bat_content(), encoding="utf-8")
    return bat_path


def _create_shortcut_lnk(bat_path: Path, target_dir: Path, icon_path: Path | None) -> Path:
    import subprocess

    lnk_path = target_dir / "ethograph.lnk"
    args = f'/D /K "{bat_path}"'
    icon_line = f'$lnk.IconLocation = "{icon_path},0"' if icon_path and icon_path.exists() else ""
    ps_script = (
        "$sh = New-Object -ComObject WScript.Shell\n"
        f'$lnk = $sh.CreateShortcut("{lnk_path}")\n'
        '$lnk.TargetPath = "cmd.exe"\n'
        f"$lnk.Arguments = '{args}'\n"
        '$lnk.WorkingDirectory = "%USERPROFILE%"\n'
        "$lnk.WindowStyle = 1\n"
        f"{icon_line}\n"
        "$lnk.Save()\n"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        check=True,
        capture_output=True,
    )
    return lnk_path


def install_shortcut() -> int:
    if sys.platform != "win32":
        print("Direct shortcut creation is only supported on Windows.")
        print("On other platforms, install via: conda install menuinst && ethograph shortcut-menuinst")
        return 1

    print("Installing shortcut...")

    icon_ext = ICON_EXTENSIONS.get(sys.platform)
    icon_src = ASSETS_DIR / f"icon{icon_ext}" if icon_ext else None

    menu_dir = Path(sys.prefix) / "Menu"
    menu_dir.mkdir(parents=True, exist_ok=True)
    if icon_src and icon_src.exists():
        icon_dst = menu_dir / icon_src.name
        if icon_dst.exists():
            icon_dst.chmod(stat.S_IWRITE | stat.S_IREAD)
        shutil.copy2(icon_src, icon_dst)
        icon_path = icon_dst
    else:
        icon_path = None

    bat_path = _create_bat_launcher()
    print(f"  Launcher: {bat_path}")

    start_menu = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "ethograph"
    start_menu.mkdir(parents=True, exist_ok=True)
    _create_shortcut_lnk(bat_path, start_menu, icon_path)
    print(f"  Start Menu: {start_menu / 'ethograph.lnk'}")

    desktop = Path.home() / "Desktop"
    if desktop.exists():
        _create_shortcut_lnk(bat_path, desktop, icon_path)
        print(f"  Desktop: {desktop / 'ethograph.lnk'}")

    print("Shortcut installed successfully.")
    return 0
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
