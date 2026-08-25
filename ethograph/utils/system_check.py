"""Linux system-library preflight for the GUI.

The pip wheels of PyQt6, PyOpenGL and wgpu carry their own code but load a
handful of shared libraries from the distribution at runtime, and a fresh
Linux box — a minimal container, a WSL distro, a lab machine without a
desktop — is usually missing some of them. Each one fails somewhere
different (a PyOpenGL log line, "could not load the Qt platform plugin",
a black video canvas), so this module names them all at once, with the one
package-manager line that installs them.

Qt-free: it runs before the GUI imports and behind ``ethograph check``.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

VULKAN_ICD_DIRS = ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d")


@dataclass(frozen=True)
class SystemLib:
    """One shared library the GUI dlopens, and the package that ships it."""

    soname: str
    apt: str
    dnf: str
    purpose: str


# Loaded on every Linux launch: PyOpenGL at import, Qt for its GL widgets.
OPENGL_LIBS: tuple[SystemLib, ...] = (
    SystemLib("libGL.so.1", "libgl1", "mesa-libGL", "OpenGL (pyqtgraph, Qt)"),
    SystemLib("libOpenGL.so.0", "libopengl0", "libglvnd-opengl", "OpenGL dispatch (PyOpenGL)"),
    SystemLib("libEGL.so.1", "libegl1", "mesa-libEGL", "EGL (Qt, wgpu)"),
    SystemLib("libfontconfig.so.1", "libfontconfig1", "fontconfig", "Qt text rendering"),
    SystemLib("libdbus-1.so.3", "libdbus-1-3", "dbus-libs", "Qt"),
)

# Only the xcb platform plugin needs these; Qt on a Wayland display (WSLg,
# most modern desktops) never opens them.
XCB_LIBS: tuple[SystemLib, ...] = (
    SystemLib("libxcb-cursor.so.0", "libxcb-cursor0", "xcb-util-cursor", "Qt xcb platform plugin"),
    SystemLib("libxkbcommon-x11.so.0", "libxkbcommon-x11-0", "libxkbcommon-x11", "Qt xcb platform plugin"),
    SystemLib("libxcb-icccm.so.4", "libxcb-icccm4", "xcb-util-wm", "Qt xcb platform plugin"),
    SystemLib("libxcb-keysyms.so.1", "libxcb-keysyms1", "xcb-util-keysyms", "Qt xcb platform plugin"),
    SystemLib("libxcb-image.so.0", "libxcb-image0", "xcb-util-image", "Qt xcb platform plugin"),
    SystemLib("libxcb-render-util.so.0", "libxcb-render-util0", "xcb-util-renderutil", "Qt xcb platform plugin"),
    SystemLib("libxcb-shape.so.0", "libxcb-shape0", "libxcb", "Qt xcb platform plugin"),
    SystemLib("libxcb-xinerama.so.0", "libxcb-xinerama0", "libxcb", "Qt xcb platform plugin"),
)

# The pygfx video canvas draws through wgpu, which on Linux means Vulkan.
VULKAN_LIBS: tuple[SystemLib, ...] = (
    SystemLib("libvulkan.so.1", "libvulkan1", "vulkan-loader", "wgpu (pygfx video canvas)"),
)

# A Vulkan loader with no driver behind it: wgpu finds no adapter and pygfx
# has nothing to draw with. mesa's package carries the software fallback
# (lavapipe) alongside the hardware drivers, so it is the one answer.
VULKAN_DRIVER = SystemLib("(vulkan ICD)", "mesa-vulkan-drivers", "mesa-vulkan-drivers", "a Vulkan driver for wgpu")

# Only checked when the audio extra is installed: sounddevice raises at import
# without it.
AUDIO_LIBS: tuple[SystemLib, ...] = (
    SystemLib("libportaudio.so.2", "libportaudio2", "portaudio", "audio playback (sounddevice)"),
)

# Everything the GUI extra can ask of the distribution — the list the docs print.
GUI_LIBS: tuple[SystemLib, ...] = OPENGL_LIBS + XCB_LIBS + VULKAN_LIBS

_APT_FAMILY = {"debian", "ubuntu"}
_DNF_FAMILY = {"fedora", "rhel", "centos"}


def is_wsl(proc_version: str | None = None, environ: dict[str, str] | None = None) -> bool:
    """Whether this Linux runs under Windows Subsystem for Linux."""
    env = os.environ if environ is None else environ
    if env.get("WSL_DISTRO_NAME") or env.get("WSL_INTEROP"):
        return True
    if proc_version is None:
        try:
            proc_version = Path("/proc/version").read_text()
        except OSError:
            return False
    return "microsoft" in proc_version.lower()


def package_manager(os_release: str | None = None) -> str | None:
    """``"apt"`` / ``"dnf"`` from ``/etc/os-release``, ``None`` when unknown.

    ``ID_LIKE`` is consulted after ``ID`` so a derivative (Mint, Pop!_OS,
    Rocky) resolves to its parent's manager.
    """
    if os_release is None:
        try:
            os_release = Path("/etc/os-release").read_text()
        except OSError:
            return None
    fields: dict[str, str] = {}
    for line in os_release.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key.strip()] = value.strip().strip('"')
    ids = [fields.get("ID", "")] + fields.get("ID_LIKE", "").split()
    for name in ids:
        if name in _APT_FAMILY:
            return "apt"
        if name in _DNF_FAMILY:
            return "dnf"
    return None


def qt_platform(environ: dict[str, str] | None = None) -> str:
    """The platform plugin Qt will load: ``"wayland"`` or ``"xcb"``.

    An explicit ``QT_QPA_PLATFORM`` wins (its first entry). Otherwise Qt
    reaches for Wayland when a Wayland display is advertised and xcb when
    only ``DISPLAY`` is — the behaviour observed with the PyQt6 wheels.
    """
    env = os.environ if environ is None else environ
    explicit = env.get("QT_QPA_PLATFORM", "").split(";")[0].strip().lower()
    if explicit:
        return "wayland" if explicit.startswith("wayland") else explicit
    return "wayland" if env.get("WAYLAND_DISPLAY") else "xcb"


def library_loads(soname: str) -> bool:
    """Whether ``dlopen`` finds the library — the same test the wheels make."""
    try:
        ctypes.CDLL(soname)
    except OSError:
        return False
    return True


def vulkan_driver_present() -> bool:
    """Whether any Vulkan ICD is registered (hardware or lavapipe)."""
    return any(glob.glob(os.path.join(d, "*.json")) for d in VULKAN_ICD_DIRS)


def missing_system_libs(
    *,
    with_audio: bool | None = None,
    platform: str | None = None,
    loads=library_loads,
    driver_present=vulkan_driver_present,
) -> list[SystemLib]:
    """Every library the installed extras need on this machine and cannot load.

    ``platform`` is the Qt platform plugin (default: :func:`qt_platform`);
    the xcb libraries are wanted only when it is ``"xcb"``. ``loads`` /
    ``driver_present`` exist so the branching can be tested without a
    Linux box.
    """
    if with_audio is None:
        with_audio = find_spec("sounddevice") is not None
    if platform is None:
        platform = qt_platform()
    wanted = OPENGL_LIBS + VULKAN_LIBS
    if platform == "xcb":
        wanted += XCB_LIBS
    if with_audio:
        wanted += AUDIO_LIBS
    missing = [lib for lib in wanted if not loads(lib.soname)]
    if not driver_present():
        missing.append(VULKAN_DRIVER)
    return missing


def install_hint(missing: list[SystemLib], manager: str | None) -> str:
    """The one line that installs everything in ``missing``.

    Falls back to naming the libraries when the package manager is unknown.
    """
    if not missing:
        return ""
    if manager == "apt":
        packages = _unique(lib.apt for lib in missing)
        return "sudo apt install " + " ".join(packages)
    if manager == "dnf":
        packages = _unique(lib.dnf for lib in missing)
        return "sudo dnf install " + " ".join(packages)
    names = _unique(lib.soname for lib in missing)
    return "install with your package manager: " + ", ".join(names)


def _unique(items) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)


def report(missing: list[SystemLib], manager: str | None) -> str:
    """A human-readable summary: what is missing, why, and how to fix it."""
    if not missing:
        return "All Linux system libraries the GUI needs are present."
    lines = ["Missing Linux system libraries:"]
    width = max(len(lib.soname) for lib in missing)
    for lib in missing:
        lines.append(f"  {lib.soname:<{width}}  {lib.purpose}")
    lines.append("")
    lines.append("Install them with:")
    lines.append("    " + install_hint(missing, manager))
    return "\n".join(lines)


def display_available(environ: dict[str, str] | None = None) -> bool:
    """Whether an X11 or Wayland display is reachable."""
    env = os.environ if environ is None else environ
    return bool(env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"))


def wsl_notes(environ: dict[str, str] | None = None) -> list[str]:
    """Advice specific to WSL, empty when nothing needs saying."""
    notes: list[str] = []
    if not display_available(environ):
        notes.append(
            "No display: WSL needs WSLg (Windows 11, or `wsl --update` from PowerShell) for Linux GUI windows."
        )
    notes.append(
        "wgpu does not officially support WSL; the video canvas runs on the "
        "mesa-vulkan-drivers software/dzn driver. If it stays black, "
        "run ethograph natively on Windows instead."
    )
    return notes


def run_check(stream=None) -> int:
    """``ethograph check``: print the preflight and return an exit code."""
    out = sys.stdout if stream is None else stream
    if sys.platform != "linux":
        print(f"System-library check only applies to Linux (this is {sys.platform}).", file=out)
        return 0
    manager = package_manager()
    platform = qt_platform()
    missing = missing_system_libs(platform=platform)
    print(f"Qt platform plugin: {platform}", file=out)
    print(report(missing, manager), file=out)
    if is_wsl():
        print("", file=out)
        print("Running under WSL:", file=out)
        for note in wsl_notes():
            print(f"  - {note}", file=out)
    return 1 if missing else 0
