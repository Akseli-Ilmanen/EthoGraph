import pytest
import numpy as np
from pathlib import Path
import ethograph as eto
from ethograph.gui.dialog_select_template import _DOWNLOAD_BASE
from ethograph.utils.download import (
    EXAMPLE_DATASETS,
    download_assets,
    ensure_default_configs,
    is_downloaded,
    write_example_configs,
)

BIRDPARK_DIR = _DOWNLOAD_BASE / "BirdPark"
BIRDPARK_NC = BIRDPARK_DIR / "copExpBP08_trim.nc"

# Primary test dataset — BirdPark trimmed NC
TEST_DATA_DIR = BIRDPARK_DIR
TEST_NC_PATH = BIRDPARK_NC


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


def _require_birdpark():
    _ensure_dataset("birdpark", BIRDPARK_DIR)
    assert BIRDPARK_NC.exists(), f"BirdPark data not found after download: {BIRDPARK_NC}"


@pytest.fixture
def test_nc_path():
    _require_birdpark()
    return str(TEST_NC_PATH)


@pytest.fixture
def test_data_dir():
    _require_birdpark()
    return str(TEST_DATA_DIR)


@pytest.fixture
def trial_tree(test_nc_path):
    return eto.open(test_nc_path)


@pytest.fixture
def first_trial_ds(trial_tree):
    return trial_tree.itrial(0)


@pytest.fixture
def catalog(first_trial_ds, trial_tree):
    from ethograph.io.catalog import catalog_from_xarray
    return catalog_from_xarray(first_trial_ds, trial_tree)


@pytest.fixture
def type_vars_dict(catalog):
    """Backwards-compat fixture — prefer ``catalog`` in new tests."""
    return catalog.to_type_vars_dict()


@pytest.fixture
def label_dt(trial_tree):
    return trial_tree.get_label_dt()


@pytest.fixture
def app_state(qtbot, tmp_path):
    from ethograph.gui.app_state import ObservableAppState
    yaml_path = str(tmp_path / "test_gui_settings.yaml")
    state = ObservableAppState(yaml_path=yaml_path, auto_save_interval=999999)
    yield state
    state.stop_auto_save()


@pytest.fixture(autouse=True)
def _suppress_dialogs(monkeypatch):
    """Suppress all GUI popups during tests via the SUPPRESS flag.

    Also patches QMessageBox statics as a safety net for any direct callers.
    """
    import ethograph.gui.notify as _notify_mod

    monkeypatch.setattr(_notify_mod, "SUPPRESS", True)

    from qtpy.QtWidgets import QMessageBox

    _noop = lambda *a, **kw: QMessageBox.Ok
    monkeypatch.setattr(QMessageBox, "critical", _noop)
    monkeypatch.setattr(QMessageBox, "warning", _noop)
    monkeypatch.setattr(QMessageBox, "information", _noop)


@pytest.fixture
def gui(qtbot, tmp_path, monkeypatch):
    import ethograph.utils.paths as paths_module

    test_config_dir = tmp_path / ".ethograph"
    test_config_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(
        paths_module,
        "default_config_dir",
        lambda data_dir=None: test_config_dir,
    )

    import napari
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)

    from ethograph.gui.widgets_meta import MetaWidget
    meta = MetaWidget(viewer)
    meta._check_unsaved_changes = lambda event: True

    yield viewer, meta
    viewer.close()


@pytest.fixture
def loaded_gui(gui, qtbot):
    from qtpy.QtWidgets import QApplication

    _require_birdpark()

    viewer, meta = gui
    meta.io_widget.nc_file_path_edit.setText(str(TEST_NC_PATH))
    meta.app_state.nc_file_path = str(TEST_NC_PATH)
    meta.io_widget.downsample_checkbox.setChecked(False)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()

    return viewer, meta


@pytest.fixture
def no_video_gui(gui, qtbot):
    """Load birdpark dataset with audio folder only (no video) — triggers 3-panel mode."""
    from qtpy.QtWidgets import QApplication

    _require_birdpark()

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
def loaded_gui_downsampled(gui, qtbot):
    from qtpy.QtWidgets import QApplication

    _require_birdpark()

    viewer, meta = gui
    meta.io_widget.nc_file_path_edit.setText(str(TEST_NC_PATH))
    meta.app_state.nc_file_path = str(TEST_NC_PATH)
    meta.io_widget.downsample_checkbox.setChecked(True)
    meta.io_widget.downsample_spin.setValue(100)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()

    return viewer, meta
