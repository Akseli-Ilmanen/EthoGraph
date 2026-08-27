"""``infer.postprocess.gui_settings`` takes the GUI's correction numbers.

The GUI's *CP Correction* section and ``infer.postprocess`` are the same
steps under different names. ``GUI_POSTPROCESS_KEYS`` is the translation and
must agree with the GUI's own spec; ``gui_settings: true`` reads the GUI's
file every load, explicit keys win over it, and a saved run config carries
the resolved values so a run does not follow the GUI afterwards.
See ``docs/adr/0006-postprocess-from-gui-settings.md``.
"""

from pathlib import Path

import pytest
import yaml

from ethograph.segment.config import (
    GUI_POSTPROCESS_KEYS,
    GUI_POSTPROCESS_STEPS,
    GUI_SETTINGS_FILENAME,
    config_from_dict,
    config_to_dict,
    load_config,
    read_gui_postprocess,
)

GUI_VALUES = {
    "cp_min_label_length_s": 0.08,
    "cp_label_thresholds": {"3": 0.2},
    "cp_stitch_gap_len_s": 0.02,
    "cp_max_expansion_s": 0.3,
    "cp_max_shrink_s": 0.4,
    "cp_step_purge": True,
    "cp_step_stitch": True,
    "cp_step_snap": True,
    "cp_step_purge_after": True,
}


def _minimal(postprocess: dict) -> dict:
    return {
        "sessions": [{"source": "s.nc", "labels_path": "s_labels.tsv"}],
        "features": {"columns": {"speed": {}}, "labels": {"mapping": "mapping.txt"}},
        "infer": {"postprocess": postprocess},
    }


def _gui_file(directory: Path, **overrides) -> Path:
    path = directory / GUI_SETTINGS_FILENAME
    path.write_text(yaml.safe_dump({**GUI_VALUES, **overrides}), encoding="utf-8")
    return path


class TestTranslation:
    def test_keys_and_defaults_are_the_guis_own(self):
        """Contract guard: the pipeline's copy of the GUI's key names and defaults."""
        pytest.importorskip("qtpy")
        from ethograph.gui.app_state import AppStateSpec

        for field_name, (key, default) in {**GUI_POSTPROCESS_KEYS, **GUI_POSTPROCESS_STEPS}.items():
            assert key in AppStateSpec.VARS, f"{field_name}: gui_settings.yaml has no key {key!r}"
            assert AppStateSpec.VARS[key][1] == default, f"{key}: default differs from the GUI's"

    def test_values_are_translated(self, tmp_path: Path):
        values = read_gui_postprocess(_gui_file(tmp_path))
        assert values == {
            "min_duration_s": 0.08,
            "label_thresholds": {"3": 0.2},
            "stitch_gap_s": 0.02,
            "max_expansion_s": 0.3,
            "max_shrink_s": 0.4,
            "changepoint_correction": True,
        }

    def test_an_unsaved_key_takes_the_guis_default(self, tmp_path: Path):
        path = tmp_path / GUI_SETTINGS_FILENAME
        path.write_text(yaml.safe_dump({"cp_min_label_length_s": 0.5}), encoding="utf-8")
        values = read_gui_postprocess(path)
        assert values["min_duration_s"] == 0.5
        assert values["max_shrink_s"] == GUI_POSTPROCESS_KEYS["max_shrink_s"][1]
        assert values["label_thresholds"] == {}

    @pytest.mark.parametrize(
        "off, zeroed",
        [
            ({"cp_step_purge": False, "cp_step_purge_after": False}, {"min_duration_s": 0.0, "label_thresholds": {}}),
            ({"cp_step_purge": False}, {"min_duration_s": 0.08}),  # the other purge box keeps it on
            ({"cp_step_stitch": False}, {"stitch_gap_s": 0.0}),
            ({"cp_step_snap": False}, {"changepoint_correction": False}),
        ],
    )
    def test_an_unticked_step_reads_as_its_value_zeroed(self, tmp_path: Path, off, zeroed):
        values = read_gui_postprocess(_gui_file(tmp_path, **off))
        for key, expected in zeroed.items():
            assert values[key] == expected

    def test_a_missing_file_is_an_error(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="gui_settings"):
            read_gui_postprocess(tmp_path / "nowhere.yaml")


class TestConfig:
    def test_true_reads_the_ethograph_home(self, tmp_path: Path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        _gui_file(home)
        monkeypatch.setenv("ETHOGRAPH_HOME", str(home))

        cfg = config_from_dict(_minimal({"gui_settings": True}), tmp_path)
        pp = cfg.infer.postprocess
        assert pp.min_duration_s == 0.08
        assert pp.label_thresholds == {3: 0.2}
        assert pp.stitch_gap_s == 0.02
        assert (pp.max_expansion_s, pp.max_shrink_s) == (0.3, 0.4)
        assert pp.changepoint_correction is True
        assert pp.gui_settings == str(home / GUI_SETTINGS_FILENAME)

    def test_a_path_is_relative_to_the_config(self, tmp_path: Path):
        _gui_file(tmp_path)
        cfg = config_from_dict(_minimal({"gui_settings": GUI_SETTINGS_FILENAME}), tmp_path)
        assert cfg.infer.postprocess.max_shrink_s == 0.4

    def test_an_explicit_key_wins_over_the_gui(self, tmp_path: Path):
        _gui_file(tmp_path)
        cfg = config_from_dict(_minimal({"gui_settings": GUI_SETTINGS_FILENAME, "max_shrink_s": 0.01}), tmp_path)
        assert cfg.infer.postprocess.max_shrink_s == 0.01
        assert cfg.infer.postprocess.max_expansion_s == 0.3

    def test_a_dotlist_override_composes_with_it(self, tmp_path: Path):
        _gui_file(tmp_path)
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(_minimal({"gui_settings": GUI_SETTINGS_FILENAME})), encoding="utf-8"
        )
        cfg = load_config(tmp_path / "config.yaml", ["infer.postprocess.stitch_gap_s=0.5"])
        assert cfg.infer.postprocess.stitch_gap_s == 0.5
        assert cfg.infer.postprocess.min_duration_s == 0.08

    def test_the_changepoint_selection_stays_the_configs(self, tmp_path: Path):
        _gui_file(tmp_path)
        cfg = config_from_dict(
            _minimal({"gui_settings": GUI_SETTINGS_FILENAME, "changepoints": {"keypoint": "beak"}}), tmp_path
        )
        assert cfg.infer.postprocess.changepoints == {"keypoint": "beak"}

    def test_a_saved_run_config_does_not_follow_the_gui(self, tmp_path: Path):
        """The dump carries the resolved values, so re-reading it after the GUI changed gives the same run."""
        gui = _gui_file(tmp_path)
        cfg = config_from_dict(_minimal({"gui_settings": GUI_SETTINGS_FILENAME}), tmp_path)
        dumped = config_to_dict(cfg)
        assert dumped["infer"]["postprocess"]["gui_settings"] == str(gui)
        assert dumped["infer"]["postprocess"]["max_shrink_s"] == 0.4

        _gui_file(tmp_path, cp_max_shrink_s=9.0)
        again = config_from_dict(dumped, tmp_path)
        assert again.infer.postprocess.max_shrink_s == 0.4

    def test_without_it_nothing_is_read(self, tmp_path: Path):
        cfg = config_from_dict(_minimal({"max_shrink_s": 0.2}), tmp_path)
        assert cfg.infer.postprocess.gui_settings is None
        assert cfg.infer.postprocess.max_shrink_s == 0.2
