from pathlib import Path

import napari
import pytest
from qtpy.QtWidgets import QApplication, QMessageBox

import ethograph as eto
import ethograph.utils.paths as paths_module
from ethograph.gui.app_state import ObservableAppState
from ethograph.gui.dialog_select_template import (
    TEMPLATES,
    _DOWNLOAD_BASE,
    _build_alignment_nwb,
    _resolve_template_paths,
    _template_dir,
    _template_downloaded,
)
from ethograph.gui.widgets_meta import MetaWidget
from ethograph.io.catalog import catalog_from_xarray
from ethograph.utils.download import (
    EXAMPLE_DATASETS,
    download_assets,
    ensure_default_configs,
    is_downloaded,
    write_example_configs,
)


def pytest_addoption(parser):
    parser.addoption("--show", action="store_true", default=False, help="Show napari viewer for 15s after each test")


BIRDPARK_DIR = _DOWNLOAD_BASE / "BirdPark"
BIRDPARK_NC = BIRDPARK_DIR / "copExpBP08_trim.nc"
MOLL_DIR = _DOWNLOAD_BASE / "Moll2025"
MOLL_NC = MOLL_DIR / "Trial_data.nc"
MOLL_PYNAPPLE_DIR = _DOWNLOAD_BASE / "Moll2025_pynapple"


def _get_template(key: str) -> dict:
    for template in TEMPLATES:
        if template["dataset_key"] == key:
            return template
    raise KeyError(key)


def _skip_if_not_downloaded(key: str) -> None:
    template = _get_template(key)
    if not _template_downloaded(template):
        pytest.skip(f"{key} not downloaded")


def _ensure_alignment_nwb(template: dict) -> None:
    """Build alignment.nwb only when it does not already exist."""
    nwb_path = _template_dir(template) / ".ethograph" / "alignment.nwb"
    if not nwb_path.exists():
        _build_alignment_nwb(template)


def _apply_template(meta, template_key: str, downsample: bool = False) -> None:
    template = _get_template(template_key)
    _ensure_alignment_nwb(template)
    resolved = _resolve_template_paths(template)

    io = meta.io_widget
    io._clear_all_line_edits()

    if resolved["nc_file_path"]:
        io.nc_file_path_edit.setText(resolved["nc_file_path"])
        meta.app_state.nc_file_path = resolved["nc_file_path"]
    if resolved["video_folder"]:
        io.video_folder_edit.setText(resolved["video_folder"])
        meta.app_state.video_folder = resolved["video_folder"]
    if resolved["audio_folder"]:
        io.audio_folder_edit.setText(resolved["audio_folder"])
        meta.app_state.audio_folder = resolved["audio_folder"]
    if resolved.get("pose_folder"):
        io.pose_folder_edit.setText(resolved["pose_folder"])
        meta.app_state.pose_folder = resolved["pose_folder"]
    if resolved.get("import_labels"):
        io.import_labels_checkbox.setChecked(True)

    io.downsample_checkbox.setChecked(downsample)
    if downsample:
        io.downsample_spin.setValue(100)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()


def _load_template_gui(gui, template_key: str, downsample: bool = False):
    _skip_if_not_downloaded(template_key)
    viewer, meta = gui
    _apply_template(meta, template_key, downsample=downsample)
    assert meta.app_state.ready, f"Failed to load {template_key}"
    return viewer, meta


# ---------------------------------------------------------------------------
# Dataset download — runs once per session
# ---------------------------------------------------------------------------

def _ensure_dataset(key: str, folder: Path):
    """Download example dataset if not already present."""
    info = EXAMPLE_DATASETS[key]
    if not is_downloaded(key, folder):
        folder.mkdir(parents=True, exist_ok=True)
        download_assets(
            release_tag=info["release_tag"],
            assets=info["assets_gui"],
            dest=folder,
        )
        ensure_default_configs()
        write_example_configs(key, folder)


def pytest_configure(config):
    """Ensure both required datasets exist before any test runs."""
    _ensure_dataset("birdpark", BIRDPARK_DIR)
    assert BIRDPARK_NC.exists(), f"BirdPark NC not found after download: {BIRDPARK_NC}"

    _ensure_dataset("moll2025", MOLL_DIR)
    assert MOLL_NC.exists(), f"Moll2025 NC not found after download: {MOLL_NC}"


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def moll2025_nc_path() -> str:
    return str(MOLL_NC)


@pytest.fixture
def birdpark_data_dir() -> str:
    return str(BIRDPARK_DIR)


@pytest.fixture
def moll2025_trial_tree(moll2025_nc_path):
    return eto.open(moll2025_nc_path)


@pytest.fixture
def moll2025_first_trial_ds(moll2025_trial_tree):
    return moll2025_trial_tree.itrial(0)


@pytest.fixture
def moll2025_catalog(moll2025_first_trial_ds, moll2025_trial_tree):
    return catalog_from_xarray(moll2025_first_trial_ds, moll2025_trial_tree)


@pytest.fixture
def moll2025_type_vars_dict(moll2025_catalog):
    """Backwards-compat fixture — prefer ``catalog`` in new tests."""
    return moll2025_catalog.to_type_vars_dict()


@pytest.fixture
def moll2025_label_dt(moll2025_trial_tree):
    return moll2025_trial_tree.get_label_dt()


@pytest.fixture
def app_state(qtbot, tmp_path):
    yaml_path = str(tmp_path / "test_gui_settings.yaml")
    state = ObservableAppState(yaml_path=yaml_path, auto_save_interval=999999)
    yield state
    state.stop_auto_save()


# ---------------------------------------------------------------------------
# Dialog suppression
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _suppress_dialogs(monkeypatch):
    """Suppress all GUI popups during tests via the SUPPRESS flag.

    Also patches QMessageBox statics as a safety net for any direct callers.
    """
    import ethograph.gui.notify as _notify_mod

    monkeypatch.setattr(_notify_mod, "SUPPRESS", True)

    _noop = lambda *a, **kw: QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, "critical", _noop)
    monkeypatch.setattr(QMessageBox, "warning", _noop)
    monkeypatch.setattr(QMessageBox, "information", _noop)


# ---------------------------------------------------------------------------
# GUI fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gui(request, qtbot, tmp_path, monkeypatch):
    show = request.config.getoption("--show")

    test_config_dir = tmp_path / ".ethograph"
    test_config_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(
        paths_module,
        "default_config_dir",
        lambda data_dir=None: test_config_dir,
    )

    viewer = napari.Viewer(show=show)
    qtbot.addWidget(viewer.window._qt_window)

    meta = MetaWidget(viewer)
    meta._check_unsaved_changes = lambda event: True

    yield viewer, meta
    if show:
        qtbot.wait(15_000)
    viewer.close()



@pytest.fixture
def birdpark_audio_only_gui(gui, qtbot):
    """Load birdpark dataset with audio folder only (no video) — triggers 3-panel mode."""
    viewer, meta = gui
    nc_path = str(BIRDPARK_NC)

    meta.io_widget.nc_file_path_edit.setText(nc_path)
    meta.app_state.nc_file_path = nc_path
    meta.io_widget.audio_folder_edit.setText(str(BIRDPARK_DIR))
    meta.app_state.audio_folder = str(BIRDPARK_DIR)
    # Deliberately do NOT set video_folder
    meta.app_state.video_folder = None
    meta.io_widget.downsample_checkbox.setChecked(False)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()

    return viewer, meta


@pytest.fixture
def no_video_gui(birdpark_audio_only_gui):
    """Backward-compatible alias for tests that still request no_video_gui."""
    return birdpark_audio_only_gui


@pytest.fixture
def birdpark_gui(gui, qtbot):
    return _load_template_gui(gui, "birdpark")


@pytest.fixture
def moll2025_gui(gui, qtbot):
    return _load_template_gui(gui, "moll2025")


@pytest.fixture
def lockbox_gui(gui, qtbot):
    return _load_template_gui(gui, "lockbox")


@pytest.fixture
def philodoptera_gui(gui, qtbot):
    return _load_template_gui(gui, "philodoptera")


@pytest.fixture
def canary_gui(gui, qtbot):
    _skip_if_not_downloaded("canary")
    viewer, meta = gui
    template = _get_template("canary")
    resolved = _resolve_template_paths(template)
    nc_path = resolved.get("nc_file_path")
    if not nc_path:
        audio_name = template.get("audio_file")
        if audio_name:
            candidate = _template_dir(template) / (Path(audio_name).stem + ".nc")
            if candidate.exists():
                nc_path = str(candidate)
    if not nc_path or not Path(nc_path).exists():
        pytest.skip("Canary .nc is missing; generate it via template dialog first")

    io = meta.io_widget
    io._clear_all_line_edits()
    io.nc_file_path_edit.setText(nc_path)
    meta.app_state.nc_file_path = nc_path
    if resolved.get("audio_folder"):
        io.audio_folder_edit.setText(resolved["audio_folder"])
        meta.app_state.audio_folder = resolved["audio_folder"]

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()
    assert meta.app_state.ready, "Failed to load canary"
    return viewer, meta


@pytest.fixture
def birdpark_gui_downsampled(gui, qtbot):
    return _load_template_gui(gui, "birdpark", downsample=True)


@pytest.fixture
def moll2025_pynapple_gui(gui, qtbot):
    """Load Moll2025 pynapple .npz data with alignment NWB linking to original media."""
    npz_dir = MOLL_PYNAPPLE_DIR
    speed_npz = npz_dir / "beakTip_speed.npz"
    alignment = npz_dir / ".ethograph" / "alignment.nwb"
    if not speed_npz.exists():
        pytest.skip("Moll2025_pynapple not set up")
    if not alignment.exists():
        from ethograph.utils.download import setup_moll2025_pynapple
        setup_moll2025_pynapple(npz_dir)
    assert alignment.exists(), f"alignment.nwb not found at {alignment}"

    viewer, meta = gui
    io = meta.io_widget
    io._clear_all_line_edits()

    meta.app_state._suspend_local_autoload = True
    io.nc_file_path_edit.setText(str(speed_npz))
    meta.app_state.nc_file_path = str(speed_npz)
    meta.app_state._suspend_local_autoload = False

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()

    assert meta.app_state.ready, "Failed to load Moll2025 pynapple"
    return viewer, meta
