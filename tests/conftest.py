import pytest
import numpy as np
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_NC_PATH = TEST_DATA_DIR / "copExpBP08_trim.nc"


@pytest.fixture
def test_nc_path():
    assert TEST_NC_PATH.exists(), f"Test data not found: {TEST_NC_PATH}"
    return str(TEST_NC_PATH)


@pytest.fixture
def test_data_dir():
    return str(TEST_DATA_DIR)


@pytest.fixture
def trial_tree(test_nc_path):
    from ethograph import TrialTree
    return TrialTree.open(test_nc_path)


@pytest.fixture
def first_trial_ds(trial_tree):
    return trial_tree.itrial(0)


@pytest.fixture
def type_vars_dict(first_trial_ds, trial_tree):
    from ethograph.utils.validation import extract_type_vars
    return extract_type_vars(first_trial_ds, trial_tree)


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


@pytest.fixture
def gui(qtbot, tmp_path, monkeypatch):
    import ethograph.utils.paths as paths_module

    yaml_path = tmp_path / "test_gui_settings.yaml"
    monkeypatch.setattr(
        paths_module,
        "gui_default_settings_path",
        lambda: yaml_path,
    )

    import napari
    viewer = napari.Viewer(show=False)
    qtbot.addWidget(viewer.window._qt_window)

    from ethograph.gui.widgets_meta import MetaWidget
    meta = MetaWidget(viewer)

    yield viewer, meta
    viewer.close()


@pytest.fixture
def loaded_gui(gui, qtbot):
    from qtpy.QtWidgets import QApplication

    viewer, meta = gui
    meta.io_widget.nc_file_path_edit.setText(str(TEST_NC_PATH))
    meta.app_state.nc_file_path = str(TEST_NC_PATH)
    meta.io_widget.downsample_checkbox.setChecked(False)

    meta.data_widget.on_load_clicked()
    QApplication.processEvents()

    return viewer, meta
