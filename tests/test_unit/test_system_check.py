"""The Linux preflight names what is missing and how to install it.

Qt-free and platform-free: the dlopen probe and the Vulkan ICD scan are
injected, so the branching runs on any machine.
"""

from __future__ import annotations

from ethograph.utils.system_check import (
    AUDIO_LIBS,
    GUI_LIBS,
    VULKAN_DRIVER,
    XCB_LIBS,
    SystemLib,
    install_hint,
    is_wsl,
    missing_system_libs,
    package_manager,
    qt_platform,
    report,
    wsl_notes,
)

UBUNTU = 'NAME="Ubuntu"\nID=ubuntu\nID_LIKE=debian\n'
MINT = 'ID=linuxmint\nID_LIKE="ubuntu debian"\n'
ROCKY = 'ID="rocky"\nID_LIKE="rhel centos fedora"\n'
ARCH = "ID=arch\n"


class TestPackageManager:
    def test_id_resolves(self):
        assert package_manager(UBUNTU) == "apt"
        assert package_manager("ID=fedora\n") == "dnf"

    def test_derivative_resolves_through_id_like(self):
        assert package_manager(MINT) == "apt"
        assert package_manager(ROCKY) == "dnf"

    def test_unknown_distro_is_none(self):
        assert package_manager(ARCH) is None
        assert package_manager("") is None


class TestIsWsl:
    def test_env_var_wins(self):
        assert is_wsl(proc_version="Linux version 6.8", environ={"WSL_DISTRO_NAME": "Ubuntu"})

    def test_proc_version_is_read(self):
        assert is_wsl(proc_version="Linux version 5.15.167.4-microsoft-standard-WSL2", environ={})
        assert not is_wsl(proc_version="Linux version 6.8.0-45-generic (buildd@lcy02)", environ={})


class TestQtPlatform:
    def test_explicit_setting_wins(self):
        assert qt_platform({"QT_QPA_PLATFORM": "xcb", "WAYLAND_DISPLAY": "wayland-0"}) == "xcb"
        assert qt_platform({"QT_QPA_PLATFORM": "wayland-egl;xcb"}) == "wayland"

    def test_wayland_display_means_wayland(self):
        # WSLg: both displays advertised, Qt takes Wayland.
        assert qt_platform({"DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-0"}) == "wayland"
        assert qt_platform({"DISPLAY": ":0"}) == "xcb"
        assert qt_platform({}) == "xcb"


class TestMissing:
    _ALL = dict(with_audio=True, platform="xcb", loads=lambda _: True, driver_present=lambda: True)

    def test_everything_present_is_empty(self):
        assert missing_system_libs(**self._ALL) == []

    def test_reports_only_what_fails_to_load(self):
        missing = missing_system_libs(
            with_audio=False,
            platform="xcb",
            loads=lambda so: so != "libOpenGL.so.0",
            driver_present=lambda: True,
        )
        assert [lib.soname for lib in missing] == ["libOpenGL.so.0"]

    def test_xcb_libs_wanted_only_on_xcb(self):
        # Nothing but the xcb libraries is missing — a Wayland session never opens them.
        no_xcb = lambda so: so not in {lib.soname for lib in XCB_LIBS}  # noqa: E731
        on_xcb = missing_system_libs(with_audio=False, platform="xcb", loads=no_xcb, driver_present=lambda: True)
        on_wayland = missing_system_libs(
            with_audio=False, platform="wayland", loads=no_xcb, driver_present=lambda: True
        )
        assert on_xcb == list(XCB_LIBS)
        assert on_wayland == []

    def test_audio_lib_only_when_audio_installed(self):
        absent = lambda so: not so.startswith("libportaudio")  # noqa: E731
        without = missing_system_libs(with_audio=False, platform="xcb", loads=absent, driver_present=lambda: True)
        with_ = missing_system_libs(with_audio=True, platform="xcb", loads=absent, driver_present=lambda: True)
        assert without == []
        assert with_ == list(AUDIO_LIBS)

    def test_loader_without_driver_asks_for_mesa(self):
        missing = missing_system_libs(
            with_audio=False, platform="xcb", loads=lambda _: True, driver_present=lambda: False
        )
        assert missing == [VULKAN_DRIVER]


class TestInstallHint:
    def test_apt_line_dedupes_packages(self):
        shape = next(lib for lib in GUI_LIBS if lib.soname == "libxcb-shape.so.0")
        xinerama = next(lib for lib in GUI_LIBS if lib.soname == "libxcb-xinerama.so.0")
        assert shape.dnf == xinerama.dnf
        assert install_hint([shape, xinerama], "dnf") == "sudo dnf install libxcb"
        assert install_hint([shape, xinerama], "apt") == "sudo apt install libxcb-shape0 libxcb-xinerama0"

    def test_unknown_manager_names_the_libraries(self):
        lib = SystemLib("libfoo.so.1", "foo", "foo", "testing")
        assert "libfoo.so.1" in install_hint([lib], None)
        assert "sudo" not in install_hint([lib], None)

    def test_nothing_missing_is_empty(self):
        assert install_hint([], "apt") == ""

    def test_report_carries_purpose_and_fix(self):
        text = report([GUI_LIBS[1]], "apt")
        assert "libOpenGL.so.0" in text
        assert GUI_LIBS[1].purpose in text
        assert "sudo apt install libopengl0" in text


class TestWslNotes:
    def test_no_display_is_said(self):
        notes = wsl_notes({})
        assert any("WSLg" in n for n in notes)

    def test_display_present_drops_that_note(self):
        notes = wsl_notes({"DISPLAY": ":0"})
        assert not any("WSLg" in n for n in notes)
        assert notes  # the wgpu caveat always stands
