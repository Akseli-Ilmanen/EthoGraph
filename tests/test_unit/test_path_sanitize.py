"""Settings paths that name nothing on this machine never reach the state.

A ``local_settings.yaml`` travels with its dataset and a ``gui_settings.yaml``
follows the user, so both routinely carry folders from another pc or an
unplugged drive. Restoring one made every trial report a missing
video/pose/audio file, blaming the data instead of the stale setting.
"""

from __future__ import annotations

from ethograph.gui.app_state import AppStateSpec, ObservableAppState
from ethograph.utils.paths import (
    is_throwaway_path,
    path_exists,
    sanitize_path_state,
    tmp_alignment_base,
)


def test_path_vars_are_real_state_keys():
    assert set(AppStateSpec.PATH_VARS) <= set(AppStateSpec.VARS)


def test_path_exists_distinguishes_file_and_dir(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_text("x")
    assert path_exists(str(f), "file")
    assert not path_exists(str(f), "dir")
    assert path_exists(str(tmp_path), "dir")
    assert not path_exists(str(tmp_path), "file")
    assert path_exists(str(tmp_path), "any")
    assert not path_exists("", "any")


def test_missing_paths_are_dropped_others_kept(tmp_path):
    state = {
        "video_folder": str(tmp_path / "gone"),
        "audio_folder": str(tmp_path),
        "nfft": 256,
    }
    cleaned = sanitize_path_state(state, AppStateSpec.PATH_VARS)
    assert "video_folder" not in cleaned
    assert cleaned["audio_folder"] == str(tmp_path)
    assert cleaned["nfft"] == 256
    # The caller's dict is untouched.
    assert "video_folder" in state


def test_list_paths_are_filtered_element_wise(tmp_path):
    img = tmp_path / "arena.png"
    img.write_text("x")
    cleaned = sanitize_path_state({"image_paths": [str(img), str(tmp_path / "gone.png")]}, AppStateSpec.PATH_VARS)
    assert cleaned["image_paths"] == [str(img)]

    cleaned = sanitize_path_state({"image_paths": [str(tmp_path / "gone.png")]}, AppStateSpec.PATH_VARS)
    assert "image_paths" not in cleaned


def test_load_from_dict_skips_missing_folder(tmp_path, qapp):
    state = ObservableAppState()
    state.load_from_dict({"video_folder": str(tmp_path / "gone"), "audio_folder": str(tmp_path)})
    assert state.video_folder is None
    assert state.audio_folder == str(tmp_path)


def test_unavailable_path_survives_a_save(tmp_path, qapp):
    """An unplugged drive must not permanently erase the folder from YAML."""
    missing = str(tmp_path / "gone")
    state = ObservableAppState()
    state.load_from_dict({"video_folder": missing})
    saved = state.get_saveable_state_dict(scope=AppStateSpec.SCOPE_LOCAL)
    assert saved["video_folder"] == missing

    # ...but a folder the user picks since replaces it.
    state.video_folder = str(tmp_path)
    saved = state.get_saveable_state_dict(scope=AppStateSpec.SCOPE_LOCAL)
    assert saved["video_folder"] == str(tmp_path)


def test_is_throwaway_path_only_matches_the_drop_dir(tmp_path):
    base = tmp_alignment_base()
    assert is_throwaway_path(str(base / "6422c617" / "alignment.tmp.nwb"))
    assert not is_throwaway_path(str(tmp_path / "alignment.nwb"))
    assert not is_throwaway_path(None)


def test_a_throwaway_alignment_is_never_saved(qapp):
    """A per-drop alignment lives until the next drop wipes its directory.

    Persisting it into the dataset's local_settings.yaml outlives the file:
    the path either dangles or resolves to another session's alignment, which
    silently shadows the real ``.ethograph/alignment.nwb``.
    """
    throwaway = str(tmp_alignment_base() / "6422c617" / "alignment-5853f903.tmp.nwb")
    state = ObservableAppState()
    state.nwb_file_path = throwaway
    saved = state.get_saveable_state_dict(scope=AppStateSpec.SCOPE_LOCAL)
    assert "nwb_file_path" not in saved
    # The live value is untouched — only persistence is refused.
    assert state.nwb_file_path == throwaway


def test_a_real_alignment_is_still_saved(tmp_path, qapp):
    real = tmp_path / "alignment.nwb"
    real.write_bytes(b"")
    state = ObservableAppState()
    state.nwb_file_path = str(real)
    saved = state.get_saveable_state_dict(scope=AppStateSpec.SCOPE_LOCAL)
    assert saved["nwb_file_path"] == str(real)
