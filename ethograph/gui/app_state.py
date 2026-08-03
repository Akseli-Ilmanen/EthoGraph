"""Settings that the user can modify and are saved in gui_settings.yaml"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from qtpy.QtCore import QObject, QTimer, Signal

import ethograph as eto
from ethograph.gui.app_constants import DEFAULT_LABEL_OVERLAY_MODES
from ethograph.gui.notify import notify
from ethograph.io.metadata_table import (
    load_metadata_df,
    load_metadata_tsv,
    trials_ep_from_metadata_df,
    validate_metadata_timing,
)
from ethograph.io.time_model import (
    RestrictionWindow,
    TimeRange,
    TrialVideoBounds,
)
from ethograph.labels.tsv_store import (
    get_trial_from_tsv,
    get_trial_meta,
    labels_tsv_path,
    save_labels_tsv,
    set_trial_in_tsv,
    set_trial_meta_attr,
)
from ethograph.utils.paths import auto_git_commit, ethograph_home
from ethograph.utils.qt import find_combo_index

logger = logging.getLogger(__name__)

SIMPLE_SIGNAL_TYPES = (int, float, str, bool)


def get_signal_type(type_hint):
    """Derive Qt Signal-compatible type from a type hint."""
    if type_hint in SIMPLE_SIGNAL_TYPES:
        return type_hint
    return object


def check_type(value, type_hint) -> bool:
    """Check if value matches type_hint. Returns True if valid."""
    if value is None:
        origin = get_origin(type_hint)
        if origin is type(int | str):  # UnionType
            return type(None) in get_args(type_hint)
        return type_hint is type(None)

    origin = get_origin(type_hint)

    if origin is type(int | str):  # UnionType (e.g., str | None)
        return any(check_type(value, arg) for arg in get_args(type_hint))

    if origin is list:
        if not isinstance(value, list):
            return False
        args = get_args(type_hint)
        if args:
            return all(isinstance(item, args[0]) for item in value)
        return True

    if origin is dict:
        if not isinstance(value, dict):
            return False
        args = get_args(type_hint)
        if len(args) == 2:
            key_type, val_type = args
            return all(isinstance(k, key_type) for k in value.keys())
        return True

    if isinstance(type_hint, type):
        return isinstance(value, type_hint)

    return True


class AppStateSpec:
    SCOPE_GLOBAL = "global"
    SCOPE_LOCAL = "local"

    # Variable name: (type, default, save_to_yaml)
    VARS = {
        # Video
        "current_frame": (int, 0, False),
        "changes_saved": (bool, True, False),
        "video": (object | None, None, False),
        "num_frames": (int, 0, False),
        "_info_data": (dict[str, Any], {}, False),
        "sync_state": (str | None, None, False),
        "before_s_trial": (float, 0.0, True),
        "after_s_trial": (float, 0.0, True),
        "before_s_label": (float, 1.0, True),
        "after_s_label": (float, 1.0, True),
        "before_s_sequence": (float, 1.0, True),
        "after_s_sequence": (float, 1.0, True),
        # How the plot x-limits are derived: "interval" (follows slider scope:
        # trial/label/sequence extent + before/after padding) or "fixed"
        # (fixed-size window from t=0). User preference, not tied to how the
        # dataset was loaded — SCOPE_GLOBAL (default) so it persists across
        # datasets instead of being guessed per load path.
        "xlim_mode": (str, "interval", True),
        "fixed_window_s": (float, 10.0, True, SCOPE_LOCAL),
        "pose_markers_visible": (bool, True, True, SCOPE_LOCAL),
        "labels_visible": (bool, True, True, SCOPE_LOCAL),
        # Per-plot-type label rendering: "full" | "bottom" | "none"
        "label_overlay_modes": (dict[str, str], dict(DEFAULT_LABEL_OVERLAY_MODES), True),
        "feature_view_mode": (str, "LinePlot", True, SCOPE_LOCAL),
        # Panel layout (UnifiedPanelContainer.layout_state()): per-dataset,
        # auto-saved to .ethograph/local_settings.yaml like other local vars.
        "panel_layout": (dict | None, None, True, SCOPE_LOCAL),
        # Outer window state (geometry base64 blob): app-wide, auto-saved to
        # gui_settings.yaml. No JSON layout files exist.
        "window_state": (dict | None, None, True),
        # Data
        "data_loader": (object | None, None, False),
        "source_collection": (object | None, None, False),
        "ds": (xr.Dataset | None, None, False),
        "ds_temp": (xr.Dataset | None, None, False),
        "dt": (xr.DataTree | None, None, False),
        "labels_confidence_ds": (xr.Dataset | None, None, False),
        "pred_labels_df": (pd.DataFrame | None, None, False),
        "pred_store": (object, None, False),
        "pred_confidence_threshold": (float, 0.75, True),
        "pred_segment_confidence_threshold": (float, 0.6, True),
        "trial_conditions": (list | None, None, False),
        "keypoints": (list[str], [], False),
        "import_labels_nc_data": (bool, False, True),
        # Playback speed as a % of the original recording speed (100 = native
        # speed). Drives both the video frame rate and the audio pitch/rate
        # together — there is no separate FPS or audio-speed control.
        "playback_speed_pct": (float, 100.0, True),
        # "auto" | "synced" | "smooth" | "skip" — global playback preference.
        "playback_mode": (str, "auto", True),
        "hide_label_text": (bool, False, True),
        # Segment playback (V key / "Play segment"): when True, the red marker
        # ends on the label's exact (sub-frame) offset time; when False (default)
        # it ends on the nearest video frame's time. See docs/advanced/playback.md.
        "segment_end_continuous_time": (bool, False, True),
        "filter_warnings": (bool, True, True),
        "center_playback": (bool, False, True),
        "time_jump_s": (float, 0.1, True),
        "time": (
            xr.DataArray | None,
            None,
            False,
        ),  # for feature variables (e.g. 'time' or 'time_aux')
        "label_intervals": (pd.DataFrame | None, None, False),
        "metadata_df": (pd.DataFrame | None, None, False),
        "metadata_path": (str | None, None, True, SCOPE_LOCAL),
        "trial_alignment": (TrialVideoBounds | None, None, False),
        "ephys_offset": (float, 0.0, True, SCOPE_LOCAL),
        "navigate_mode": (str, "trial", True, SCOPE_LOCAL),
        "slider_scope": (str, "trial", True, SCOPE_LOCAL),
        "restrict_window": (RestrictionWindow | None, None, False),
        "label_instance_idx": (int, 0, False),
        "sequence_pattern": (str, "", True),
        "sequence_match_idx": (int, 0, False),
        "trials": (list[int | str], [], False),
        "downsample_enabled": (bool, False, True),
        "downsample_factor": (int, 100, True),
        # Boolean
        "has_audio": (bool, False, False),
        "has_neo": (bool, False, False),
        "has_neurons": (bool, False, False),
        "files_aligned_to_trials": (bool, True, True, SCOPE_LOCAL),
        # Paths
        "nc_file_path": (str | None, None, False),
        "_labels_file_path": (
            str | None,
            None,
            False,
        ),  # Tracks active labels file (canonical or predictions)
        "nwb_file_path": (str | None, None, True, SCOPE_LOCAL),
        "video_folder": (str | None, None, True, SCOPE_LOCAL),
        "audio_folder": (str | None, None, True, SCOPE_LOCAL),
        "pose_folder": (str | None, None, True, SCOPE_LOCAL),
        "ephys_path": (str | None, None, True, SCOPE_LOCAL),
        "neurons_path": (str | None, None, True, SCOPE_LOCAL),
        "video_path": (str | None, None, False),
        # Playback quality: "full" decodes the source video; "proxy" decodes a
        # cached low-res/short-GOP copy for smooth navigation. Global viewing
        # pref (not per-dataset). Only affects which file the DECODER reads;
        # all alignment/frame math stays on the source.
        "video_quality_mode": (str, "full", True),
        "audio_path": (str | None, None, False),
        # audio_source_map key driving audio PLAYBACK (last-clicked audio panel);
        # None follows the global mic combo. Distinct from what each panel draws.
        "playback_mic_key": (str | None, None, False),
        "pose_path": (str | None, None, False),
        "source_software": (str | None, None, True, SCOPE_LOCAL),
        "image_paths": (list[str], [], True, SCOPE_LOCAL),
        "nwb_pose_keys": (list[str], [], True, SCOPE_LOCAL),
        "pose_hide_threshold": (float, 0.9, True),
        "pose_show_skeleton": (bool, False, True),
        "pose_points_use_base": (bool, False, True),
        "pose_points_base_color": (str | None, "#FF3333", True),
        "skeleton_use_base": (bool, True, True),
        "skeleton_base_color": (str | None, "#00CC66", True),
        "skeleton_config_override": (dict | None, None, True),
        # Keypoint labelling: the schema being labelled and the chosen fill
        # backend. The labelled coordinates themselves are project data and go
        # to a sidecar next to the video, never here.
        "labelling_keypoints": (list[str], [], True, SCOPE_LOCAL),
        "labelling_individuals": (list[str], [], True, SCOPE_LOCAL),
        "labelling_backend": (str, "spline", True),
        # Labelling marker diameter, in SCREEN pixels (zoom-independent).
        "labelling_point_size": (float, 16.0, True),
        # Plotting
        "ymin": (float | None, None, True),
        "ymax": (float | None, None, True),
        "spec_ymin": (float | None, None, True),
        "spec_ymax": (float | None, None, True),
        "ready": (bool, False, False),
        "downsample_factor_used": (int | None, None, False),
        "nfft": (int, 256, True),
        "hop_frac": (float, 0.5, True),
        "vmin_db": (float, -120.0, True),
        "vmax_db": (float, -20.0, True),
        "buffer_multiplier": (float, 5.0, True),
        "percentile_ylim": (float, 99.5, True),
        "space_plot_type": (str, "Layers", True, SCOPE_LOCAL),
        "space_feature": (str | None, None, True),
        "space_dim": (str | None, None, True),
        "space_color": (str | None, None, True),
        "space_x_axis": (str | None, None, True),
        "space_y_axis": (str | None, None, True),
        "space_z_axis": (str | None, None, True),
        "space_3d": (bool, False, True),
        "space_percentile_xyzlim": (float, 100.0, True),
        "space_marker_visible": (bool, True, True),
        "space_confidence_filter": (bool, False, True),
        "space_confidence_threshold": (float, 0.6, True),
        "space_limit_to_window": (
            bool,
            False,
            False,
        ),  # May confuse user, better not keep saved.
        "space_lock_axes": (bool, False, False),
        "space_hide_zeros": (bool, False, True),
        "space_show_references": (bool, True, True),
        "space_library_geometry": (str | None, None, True, SCOPE_LOCAL),
        "primary_camera": (str | None, None, True),
        "primary_camera_previous": (str | None, None, False),
        "extra_cameras": (list[str], [], True),
        "lock_axes": (bool, False, False),
        "zen_mode": (bool, False, False),
        "spec_colormap": (str, "CET-R4", True),
        "spec_levels_mode": (str, "auto", True),
        # All checkbox states for dimension combos (e.g., {"keypoint": True, "space": False})
        "all_checkbox_states": (dict[str, bool], {}, True),
        # Audio processing
        "audio_cp_hop_length_ms": (float, 5.0, True),
        "audio_cp_min_level_db": (float, -70.0, True),
        "audio_cp_min_syllable_length_s": (float, 0.02, True),
        "audio_cp_silence_threshold": (float, 0.1, True),
        "show_changepoints": (bool, True, True),
        "plot_has_changepoints": (bool, False, False),
        "apply_changepoint_correction": (bool, True, True),
        "cp_step_purge": (bool, True, True),
        "cp_step_stitch": (bool, True, True),
        "cp_step_snap": (bool, True, True),
        "cp_step_purge_after": (bool, True, True),
        "automatic_min_label_length_s": (float, 1e-3, True),
        "automatic_stitch_gap_s": (float, 0.0, True),
        "remote_backup_enabled": (bool, False, True),
        "remote_backup_path": (str | None, None, True),
        "remote_backup_mode": (str, "timestamp", True),
        "remote_path_depth": (int, 0, True),
        # Envelope / energy (general, used by both heatmap and overlay)
        "energy_metric": (str, "energy_lowpass", True),
        "env_rate": (float, 2000.0, True),
        "env_cutoff": (float, 500.0, True),
        "freq_cutoffs_min": (float, 500.0, True),
        "freq_cutoffs_max": (float, 10000.0, True),
        "smooth_win": (float, 2.0, True),
        "band_env_min": (float, 300.0, True),
        "band_env_max": (float, 6000.0, True),
        "band_env_rate": (float, 1000.0, True),
        "ava_min_freq": (float, 30000.0, True),
        "ava_max_freq": (float, 110000.0, True),
        "ava_smoothing_timescale": (float, 0.007, True),
        "ava_use_softmax_amp": (bool, True, True),
        # Heatmap-specific display
        "heatmap_exclusion_percentile": (float, 98.0, True),
        "heatmap_colormap": (str, "RdBu_r", True),
        "heatmap_normalization": (str, "per_channel", True),
        # Firing rate
        "fr_bin_size": (float, 0.01, True),
        "fr_sigma": (float, 2.0, True),
        # Changepoint correction
        "cp_min_label_length_s": (float, 0.05, True),
        "cp_stitch_gap_len_s": (float, 0.015, True),
        "cp_max_expansion_s": (float, 0.05, True),
        "cp_max_shrink_s": (float, 0.05, True),
        "cp_label_thresholds": (dict, {}, True),
        # Function params cache (dialog_function_params.py)
        "function_params_cache": (dict, {}, True),
    }

    @classmethod
    def get_meta(cls, key):
        if key not in cls.VARS:
            raise KeyError(f"No metadata for key: {key}")
        value = cls.VARS[key]
        if len(value) == 3:
            type_hint, default, save = value
            scope = cls.SCOPE_GLOBAL
            return type_hint, default, save, scope
        type_hint, default, save, scope = value
        return type_hint, default, save, scope

    @classmethod
    def get_default(cls, key):
        return cls.get_meta(key)[1]

    @classmethod
    def get_type(cls, key):
        return cls.get_meta(key)[0]

    @classmethod
    def saveable_attributes(cls, scope: str | None = None) -> set[str]:
        attrs = set()
        for key in cls.VARS:
            _, _, save, key_scope = cls.get_meta(key)
            if not save:
                continue
            if scope is None or scope == key_scope:
                attrs.add(key)
        return attrs


class ObservableAppState(QObject):
    """State container with change notifications and computed properties."""

    # Signals for state changes (auto-derive signal type from type hint)
    for var in AppStateSpec.VARS:
        type_hint, _, _, _ = AppStateSpec.get_meta(var)
        locals()[f"{var}_changed"] = Signal(get_signal_type(type_hint))

    trial_changed = Signal()
    GLOBAL_SETTINGS_FILENAME = "gui_settings.yaml"
    LOCAL_SETTINGS_FILENAME = "local_settings.yaml"
    SETTINGS_DIRNAME = ".ethograph"
    _TIME_REFRESH_KEYS = {"ds", "dt", "video", "video_path", "audio_path"}

    def __init__(self, yaml_path: str | None = None, auto_save_interval: int = 10000):
        super().__init__()
        object.__setattr__(self, "_values", {})
        for var in AppStateSpec.VARS:
            _, default, _, _ = AppStateSpec.get_meta(var)
            self._values[var] = default

        self.audio_source_map: dict[str, tuple[str, int]] = {}
        # mic device label -> ordered audio_source_map keys (one per channel)
        self.audio_mic_channels: dict[str, list[str]] = {}
        self.ephys_source_map: dict[
            str, tuple[str, str, int]
        ] = {}  # filepath, neo_stream_id, channel_idx, e.g.("/data/session.rhd", "1", 0)).
        self.ephys_stream_sel: str | None = None
        self._suspend_local_autoload = False
        self._all_labels_df: pd.DataFrame | None = None
        self._metadata_df: pd.DataFrame | None = None
        self._label_mappings: dict | None = None
        # Label branches have a fixed position mapping: branch 0 always draws
        # "full" (the entire plot), branch 1 always draws "top1", branch 2
        # always draws "top2" — there are never more than 3 branches. Only
        # one branch is "active" (editable by clicking/labeling) at a time;
        # each branch's overlay visibility is independent of whether it's active.
        self._active_branch: int = 0
        self._branch_shown: dict[int, bool] = {0: True}
        self._show_predictions_overlay: bool = False

        from ethograph.io.nwb_alignment import EmpytAlignment

        self.nwb_alignment = EmpytAlignment()

        self._yaml_path = yaml_path or "gui_settings.yaml"
        self._auto_save_timer = QTimer()
        self._auto_save_timer.timeout.connect(self.save_to_yaml)
        self._auto_save_timer.start(auto_save_interval)

    @property
    def video_fps(self) -> float | None:
        camera = self.primary_camera
        return self.nwb_alignment.get_stream_rate("video", camera)

    @property
    def sel_attrs(self) -> dict:
        """
        Return all attributes ending with _sel as a dict.
        """
        result = {}
        for attr in dir(self):
            if attr.endswith("_sel"):
                value = getattr(self, attr, None)
                if not callable(value):
                    result[attr] = value
        return result

    @property
    def active_label_ids(self) -> set[int] | None:
        """Return label IDs belonging to any branch currently shown as an overlay.

        Returns None when no mappings are loaded (meaning all IDs allowed).
        This gates which existing labels can be clicked/selected/displayed —
        it is independent of which branch is *editable* (see
        :attr:`editable_label_ids`).
        """
        mappings = self._label_mappings
        if not mappings:
            return None
        shown = self._shown_branches
        return {lid for lid, data in mappings.items() if isinstance(lid, int) and data.get("branch", 0) in shown}

    @property
    def _shown_branches(self) -> set[int]:
        """Set of branch indices whose visibility checkbox is currently on."""
        return {b for b, shown in self._branch_shown.items() if shown}

    @property
    def editable_label_ids(self) -> set[int] | None:
        """Label IDs belonging to the currently active (editable) branch.

        New labels are only ever drawn into the active branch; labels of
        every other branch must never be trimmed/overwritten by it,
        regardless of whether those other branches are currently shown.
        """
        mappings = self._label_mappings
        if not mappings:
            return None
        return {
            lid
            for lid, data in mappings.items()
            if isinstance(lid, int) and data.get("branch", 0) == self._active_branch
        }

    @property
    def trial_bounds(self) -> TimeRange | None:
        """Time range for the current trial, sourced from TrialVideoBounds.trial_range."""
        alignment = getattr(self, "trial_alignment", None)
        if alignment is not None:
            return alignment.trial_range
        return None

    @property
    def before_s(self) -> float:
        mode = getattr(self, "navigate_mode", "trial")
        return self._values.get(f"before_s_{mode}", 0.0)

    @property
    def after_s(self) -> float:
        mode = getattr(self, "navigate_mode", "trial")
        return self._values.get(f"after_s_{mode}", 0.0)

    @property
    def view_span(self) -> float:
        if self.get_with_default("xlim_mode") == "fixed":
            return self.get_with_default("fixed_window_s")
        return self.before_s + self.after_s

    @property
    def window_bounds(self) -> TimeRange | None:
        """Core data range — the actual trial/label/sequence extent without padding.

        Plots use this for x-axis limits and zoom constraints.
        The padded ``restrict_window.time_range`` is for slider/scroll limits.
        In fixed x-limits mode the core window is just a viewport that slides
        over the full scope extent, so the extent is the data range.
        """
        rw = getattr(self, "restrict_window", None)
        if rw is not None:
            return rw.time_range if rw.mode == "fixed" else rw.core_range
        return self.trial_bounds

    @property
    def padded_bounds(self) -> TimeRange | None:
        """Padded display range including before/after context.

        Use for scroll/slider limits where the user should be able to pan
        beyond the core trial range.
        """
        rw = getattr(self, "restrict_window", None)
        if rw is not None:
            return rw.time_range
        return self.trial_bounds

    @property
    def time_coord(self) -> xr.DataArray | None:
        """Get the time coordinate for the currently selected features."""
        ds = getattr(self, "ds", None)
        features_sel = getattr(self, "features_sel", None)
        if ds is not None and features_sel in ds.data_vars:
            return eto.get_time_coord(ds[features_sel])
        return None

    def get_with_default(self, key):
        """Return value from app state, or default from AppStateSpec if None."""
        value = getattr(self, key, None)
        if value is None:
            value = AppStateSpec.get_default(key)
        return value

    def get_ephys_source(self) -> tuple[str | None, str, int]:
        """Get ephys file path, stream_id, and channel index from current ephys_stream_sel.

        Returns (ephys_path, stream_id, channel_idx) tuple. Uses ephys_source_map
        to resolve the display name.
        """
        import os

        stream_sel = getattr(self, "ephys_stream_sel", None)
        if not stream_sel or not self.ephys_source_map:
            return None, "0", 0

        entry = self.ephys_source_map.get(stream_sel)
        if entry is None:
            return None, "0", 0

        filename, stream_id, channel_idx = entry

        if not filename:
            return None, stream_id, channel_idx

        if os.path.isabs(filename):
            ephys_path = os.path.normpath(filename)
        else:
            base_ephys_path = getattr(self, "ephys_path", None)
            if not base_ephys_path:
                return None, stream_id, channel_idx
            ephys_path = os.path.normpath(os.path.join(os.path.dirname(base_ephys_path), filename))

        return ephys_path, stream_id, channel_idx

    def playback_mic_selection(self) -> str | None:
        """audio_source_map key that drives playback: the last-clicked panel's
        pin, else the global mic. Only returns a key valid in the current
        dataset (a stale key from a prior dataset is ignored)."""
        for key in (self.playback_mic_key, getattr(self, "mics_sel", None)):
            if key and key in self.audio_source_map:
                return key
        return None

    def playback_audio_label(self) -> str | None:
        """Compact indicator label ``ChN: first-10-chars…`` (full name in tooltip)."""
        key = self.playback_mic_selection()
        if not key:
            return None
        mic_file, ch = self.audio_source_map.get(key, (key, 0))
        name = str(mic_file)
        short = name[:10] + ("…" if len(name) > 10 else "")
        return f"Ch{ch + 1}: {short}"

    def playback_audio_tooltip(self) -> str | None:
        """Full channel description for the indicator's hover tooltip."""
        key = self.playback_mic_selection()
        if not key:
            return None
        mic_file, ch = self.audio_source_map.get(key, (key, 0))
        return f"Playback channel {ch + 1} — {mic_file}"

    def has_playback_audio(self) -> bool:
        """Whether an audio channel is available to play back."""
        return bool(getattr(self, "has_audio", False) or self.audio_path or self.playback_mic_selection())

    def effective_playback_mode(self) -> str:
        """Resolve ``playback_mode`` to a concrete mode for the current data.

        ``auto`` follows audio presence; an explicit ``synced`` with no audio
        degrades to ``smooth`` (there is nothing to synchronise to).
        """
        from .app_constants import PLAYBACK_MODE_AUTO, PLAYBACK_MODE_SMOOTH, PLAYBACK_MODE_SYNCED

        mode = self.playback_mode
        has_audio = self.has_playback_audio()
        if mode == PLAYBACK_MODE_AUTO:
            return PLAYBACK_MODE_SYNCED if has_audio else PLAYBACK_MODE_SMOOTH
        if mode == PLAYBACK_MODE_SYNCED and not has_audio:
            return PLAYBACK_MODE_SMOOTH
        return mode

    def get_audio_source(self, mic_name: str | None = None) -> tuple[str | None, int]:
        """Get audio file path and channel index for a mic selection.

        *mic_name* overrides the global ``mics_sel`` (used by audio panels
        pinned to one mic/channel). Returns (audio_path, channel_idx) tuple.
        Uses audio_source_map to resolve the display name to
        (mic_file, channel_idx).
        """
        mics_sel = mic_name or getattr(self, "mics_sel", None)
        if not mics_sel or not self.audio_source_map:
            return None, 0

        mic_file, channel_idx = self.audio_source_map.get(mics_sel, (mics_sel, 0))
        if not mic_file:
            return None, channel_idx

        audio_folder = getattr(self, "audio_folder", None)

        # Try resolve via nwb_alignment (ImageSeries path → fallback folder)
        for mic_dev in self.nwb_alignment.mics:
            trial = getattr(self, "trials_sel", None)
            if trial is None:
                break
            media = self.nwb_alignment.get_media(trial, "audio", mic_dev)
            if media and (media == mic_file or Path(media).name == mic_file):
                resolved = self.nwb_alignment.resolve_media_path(
                    trial,
                    "audio",
                    device=mic_dev,
                    fallback_folder=audio_folder,
                )
                if resolved:
                    return resolved, channel_idx
            if not media:
                # Stream-based alignments (drag & drop) have no trials-table
                # filename columns — match the ImageSeries file directly.
                resolved = self.nwb_alignment.resolve_media_path(
                    trial,
                    "audio",
                    device=mic_dev,
                    fallback_folder=audio_folder,
                )
                if resolved and (mic_file == str(mic_dev) or Path(resolved).name == mic_file):
                    return resolved, channel_idx

        # Direct fallback
        if audio_folder:
            import os

            path = os.path.normpath(os.path.join(audio_folder, mic_file))
            return path, channel_idx

        return None, channel_idx

    def __getattr__(self, name):
        # Check for class attributes/properties first
        cls = type(self)
        if hasattr(cls, name):
            attr = getattr(cls, name)
            # If it's a property, use its getter
            if hasattr(attr, "__get__"):
                return attr.__get__(self)
            return attr
        if name in AppStateSpec.VARS:
            return self._values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in (
            "time",
            "_values",
            "settings",
            "_yaml_path",
            "_auto_save_timer",
            "navigation_widget",
            "lineplot",
            "audio_source_map",
            "audio_mic_channels",
            "ephys_source_map",
            "ephys_stream_sel",
            "_suspend_local_autoload",
            "_layout_snapshot_provider",
            "_all_labels_df",
            "_metadata_df",
            "_label_mappings",
            "_active_branch",
            "_branch_shown",
            "_show_predictions_overlay",
        ):
            super().__setattr__(name, value)
            return

        if name in AppStateSpec.VARS:
            type_hint = AppStateSpec.get_type(name)
            if not check_type(value, type_hint):
                raise TypeError(f"{name}: expected {type_hint}, got {type(value).__name__} = {value!r}")

            old_value = self._values.get(name)
            self._values[name] = value

            signal = getattr(self, f"{name}_changed", None)
            if signal:
                try:
                    changed = bool(old_value != value)
                except (ValueError, TypeError):
                    changed = old_value is not value
                if changed:
                    signal.emit(value)

            if name == "nc_file_path" and not self._suspend_local_autoload:
                self.load_local_settings()

            # Auto-sync nwb_file_path → nwb_alignment
            # Skip if alignment was already set by the data loader (e.g. remote NWB)
            if name == "nwb_file_path":
                existing = getattr(self, "nwb_alignment", None)
                if existing is None or getattr(existing, "_path", None) is not None:
                    from ethograph.io.nwb_alignment import make_nwb_alignment

                    self.nwb_alignment = make_nwb_alignment(value)

            if name == "metadata_path":
                if value:
                    metadata_df, resolved_path = load_metadata_df(
                        source_path=self.nc_file_path,
                        metadata_path=value,
                        nwb_alignment=self.nwb_alignment,
                        trials_ep=self.nwb_alignment.trials_ep,
                        trial_ids=getattr(self, "trials", None) or None,
                    )
                    self._values[name] = resolved_path or value
                    self.metadata_df = metadata_df

                    # Read raw file for timing (load_metadata_df may strip
                    # timing columns depending on which fallback path it took).
                    raw_path = Path(resolved_path or value)
                    if raw_path.suffix.lower() in {".tsv", ".csv", ".xlsx", ".xls"} and raw_path.exists():
                        raw_df = load_metadata_tsv(raw_path)
                        if "start_time" in raw_df.columns and "stop_time" in raw_df.columns:
                            validate_metadata_timing(raw_df, raw_path)
                            new_ep = trials_ep_from_metadata_df(raw_df)
                            if new_ep is not None:
                                self._rebuild_trials_from_ep(new_ep)
                else:
                    self.metadata_df = None

            return

        super().__setattr__(name, value)

    def _rebuild_trials_from_ep(self, trials_ep) -> None:
        """Propagate new trial boundaries to source_collection.

        The data_loader is stateless w.r.t. trials — callers pass t0/t1
        directly to ``select()``, so no loader update is needed here.
        """
        trial_ids = list(range(1, len(trials_ep) + 1))
        self.trials = trial_ids

        # Rebuild source_collection trial bookmarks
        sc = getattr(self, "source_collection", None)
        if sc is not None:
            sc.set_trials(
                ids=trial_ids,
                starts=[float(s) for s in trials_ep.start],
                stops=[float(e) for e in trials_ep.end],
            )

        # Update alignment so .trials_ep reflects the new epochs
        alignment = getattr(self, "nwb_alignment", None)
        if alignment is not None and hasattr(alignment, "_trials_ep_cache"):
            alignment._trials_ep_cache = trials_ep

        logger.info("Rebuilt %d trials from metadata timing columns", len(trial_ids))

    # --- Dynamic _sel variables ---
    def get_ds_kwargs(self):
        ds_kwargs = {}

        for dim in self.ds.dims:
            if "time" in dim:
                continue
            attr_name = f"{dim}_sel"
            if not hasattr(self, attr_name):
                continue

            output = getattr(self, attr_name)
            if output is None or output in ["", "None"]:
                continue

            # Check if dim has coords and determine appropriate type
            if dim in self.ds.coords:
                coord_dtype = self.ds.coords[dim].dtype
                if coord_dtype.kind in ("i", "u"):
                    ds_kwargs[dim] = int(output)
                else:
                    ds_kwargs[dim] = str(output)
            else:
                # Dim without coord - assume integer index
                ds_kwargs[dim] = int(output)

        return ds_kwargs

    def get_selections(self) -> dict[str, str]:
        """Backend-agnostic selection dict from combo *_sel attributes.

        Uses ``data_loader.dims`` when available (pynapple path),
        falls back to ``get_ds_kwargs()`` for pure xarray.
        """
        store = getattr(self, "data_loader", None)
        if store is None:
            if self.ds is None:
                return {}
            return self.get_ds_kwargs()

        selections: dict[str, str] = {}
        for dim_name in store.dims:
            attr_name = f"{dim_name}_sel"
            if not hasattr(self, attr_name):
                continue
            val = getattr(self, attr_name)
            if val is not None and val not in ("", "None"):
                selections[dim_name] = str(val)
        return selections

    def key_sel_exists(self, type_key: str) -> bool:
        """Check if a key selection exists for a given type."""
        return hasattr(self, f"{type_key}_sel")

    def get_key_sel(self, type_key: str):
        """Get current value for a given info key."""
        attr_name = f"{type_key}_sel"
        return getattr(self, attr_name, None)

    def _coerce_to_list_type(self, value, reference_list: list):
        """Coerce value to match the type of items in reference_list."""
        if not reference_list:
            return value
        sample = reference_list[0]
        if isinstance(sample, int) and not isinstance(value, int):
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        return value

    def set_key_sel(self, type_key, currentValue):
        """Set current value for a given info key.

        When currentValue is None, the dimension will not be filtered in
        get_ds_kwargs(), effectively showing all values for that dimension.
        """
        if type_key == "trials" and hasattr(self, "trials") and self.trials:
            currentValue = self._coerce_to_list_type(currentValue, self.trials)

        attr_name = f"{type_key}_sel"
        prev_attr_name = f"{type_key}_sel_previous"

        current_stored_value = getattr(self, attr_name, None)
        if current_stored_value != currentValue and current_stored_value is not None:
            setattr(self, prev_attr_name, current_stored_value)

        setattr(self, attr_name, currentValue)

    def toggle_key_sel(self, type_key, data_widget):
        """Toggle between current and previous value for a given key.

        If a previous value exists, swap current and previous.
        Otherwise, cycle to the next item in the combo box.

        Special case: type_key="Audio Waveform" toggles the features
        selection to/from Audio Waveform.
        """

        attr_name = f"{type_key}_sel"
        prev_attr_name = f"{type_key}_sel_previous"

        current_value = getattr(self, attr_name, None)
        previous_value = getattr(self, prev_attr_name, None)

        if previous_value is not None:
            setattr(self, attr_name, previous_value)
            setattr(self, prev_attr_name, current_value)
            if data_widget is not None:
                self._update_combo_box(type_key, previous_value, data_widget)
        elif data_widget is not None:
            self._cycle_combo_box(type_key, data_widget)

    def cycle_key_sel(self, type_key, data_widget):
        """Cycle to the next item in the combo box for a given key."""
        if data_widget is not None:
            self._cycle_combo_box(type_key, data_widget)

    def _update_combo_box(self, type_key, new_value, data_widget):
        """Update the corresponding combo box in the UI and trigger its change signal."""
        try:
            combo = data_widget.io_widget.combos.get(type_key) or data_widget.combos.get(type_key)

            if combo is not None:
                index = find_combo_index(combo, str(new_value))
                if index < 0 and type_key == "mics":
                    for i in range(combo.count()):
                        if combo.itemText(i).startswith(str(new_value)):
                            index = i
                            break
                if index >= 0:
                    combo.setCurrentIndex(index)
        except (AttributeError, TypeError) as e:
            logger.error("Error updating combo box for %s: %s", type_key, e)

    def _cycle_combo_box(self, type_key, data_widget):
        """Cycle the combo box to the next item when no previous selection exists."""
        try:
            combo = data_widget.io_widget.combos.get(type_key) or data_widget.combos.get(type_key)
            if combo is not None and combo.count() > 1:
                next_index = (combo.currentIndex() + 1) % combo.count()
                combo.setCurrentIndex(next_index)
        except (AttributeError, TypeError) as e:
            logger.error("Error cycling combo box for %s: %s", type_key, e)

    # --- Save/Load methods ---
    PATH_SUFFIXES = ("_path", "_folder")

    def _global_settings_path(self) -> Path:
        return ethograph_home() / self.GLOBAL_SETTINGS_FILENAME

    def _local_settings_path(self) -> Path | None:
        nc_file_path = getattr(self, "nc_file_path", None)
        if not nc_file_path:
            return None
        try:
            nc_path = Path(nc_file_path)
        except (TypeError, ValueError):
            return None
        return nc_path.parent / self.SETTINGS_DIRNAME / self.LOCAL_SETTINGS_FILENAME

    def _yaml_read(self, path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _yaml_write(self, path: Path, state_dict: dict) -> None:
        # Atomic replace: a crash mid-write must never truncate the settings
        # file the next launch will load.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.dump(self._to_native(state_dict), f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, path)

    def _to_native(self, value):
        """Recursively convert numpy types to native Python types for YAML serialization."""
        if isinstance(value, dict):
            return {key: self._to_native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_native(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        return value

    def get_saveable_state_dict(self, scope: str | None = None) -> dict:
        state_dict = {}
        for attr in AppStateSpec.saveable_attributes(scope=scope):
            value = self._values.get(attr)
            if value is not None and isinstance(value, (str, float, int, bool)):
                state_dict[attr] = self._to_native(value)
            elif isinstance(value, dict) and value:
                state_dict[attr] = value

        if scope in (None, AppStateSpec.SCOPE_LOCAL):
            for attr in dir(self):
                if attr.endswith("_sel") or attr.endswith("_sel_previous"):
                    try:
                        value = getattr(self, attr)
                        if not callable(value) and value is not None:
                            if isinstance(value, (str, float, int, bool)):
                                state_dict[attr] = self._to_native(value)
                    except (AttributeError, TypeError) as exc:
                        logger.error("Error accessing %s: %s", attr, exc)
        return state_dict

    def _sort_state_dict(self, state_dict: dict) -> dict:
        """Sort state dict by category: paths, bools, _sel, strings, numbers, nested dicts."""

        def _category_key(item):
            key, value = item
            is_nested = isinstance(value, dict)
            is_path = any(key.endswith(s) for s in self.PATH_SUFFIXES)
            is_sel = key.endswith("_sel") or key.endswith("_sel_previous")
            is_bool = isinstance(value, bool)
            is_str = isinstance(value, str)

            if is_nested:
                order = 5
            elif is_path:
                order = 0
            elif is_bool:
                order = 2
            elif is_sel:
                order = 1
            elif is_str:
                order = 3
            else:
                order = 4
            return (order, key)

        return dict(sorted(state_dict.items(), key=_category_key))

    def print_state(self) -> None:
        """Print all simple-typed app state vars, grouped by category."""
        _PRINTABLE = (str, int, float, bool, list, dict, type(None))
        _CATEGORY_LABELS = {
            0: "Paths",
            1: "Selections",
            2: "Booleans",
            3: "Strings",
            4: "Numbers",
            5: "Lists/Dicts",
            6: "None",
        }

        def _category_key(item):
            key, value = item
            if isinstance(value, (dict, list)):
                return 5
            if value is None:
                return 6
            if any(key.endswith(s) for s in self.PATH_SUFFIXES):
                return 0
            if key.endswith("_sel") or key.endswith("_sel_previous"):
                return 1
            if isinstance(value, bool):
                return 2
            if isinstance(value, str):
                return 3
            return 4

        state = {}
        for attr in self._values:
            value = self._values[attr]
            if isinstance(value, _PRINTABLE):
                state[attr] = self._to_native(value) if isinstance(value, (str, int, float, bool)) else value
        # Also include dynamic _sel attributes
        for attr in dir(self):
            if attr.endswith("_sel") or attr.endswith("_sel_previous"):
                try:
                    value = getattr(self, attr)
                    if not callable(value) and isinstance(value, _PRINTABLE):
                        state[attr] = self._to_native(value) if isinstance(value, (str, int, float, bool)) else value
                except (AttributeError, TypeError):
                    pass

        current_cat = None
        for key, value in sorted(state.items(), key=lambda item: (_category_key(item), item[0])):
            cat = _category_key((key, value))
            if cat != current_cat:
                print(f"\n{'=' * 50}")
                print(f"  {_CATEGORY_LABELS[cat]}")
                print(f"{'=' * 50}")
                current_cat = cat

            if isinstance(value, list) and len(value) > 10:
                value = f"{value[:10]}... (total {len(value)} items)"

            print(f"  {key}: {value}")

    def load_from_dict(self, state_dict: dict):
        # Migrate legacy keys
        if "restrict_mode" in state_dict:
            val = state_dict.pop("restrict_mode")
            if val == "video":
                val = "trial"
            state_dict.setdefault("navigate_mode", val)

        # Migrate old before_s/after_s or restrict_extra_t0/t1 → per-category
        old_before = state_dict.pop("before_s", state_dict.pop("restrict_extra_t0", None))
        old_after = state_dict.pop("after_s", state_dict.pop("restrict_extra_t1", None))
        if old_before is not None:
            for suffix in ("trial", "label", "sequence"):
                state_dict.setdefault(f"before_s_{suffix}", old_before)
        if old_after is not None:
            for suffix in ("trial", "label", "sequence"):
                state_dict.setdefault(f"after_s_{suffix}", old_after)

        state_dict.pop("window_size", None)

        self._suspend_local_autoload = True
        try:
            for key, value in state_dict.items():
                if value is None:
                    continue
                if key in AppStateSpec.VARS or key.endswith("_sel") or key.endswith("_sel_previous"):
                    setattr(self, key, value)
        finally:
            self._suspend_local_autoload = False

    def load_local_settings(self) -> bool:
        try:
            local_path = self._local_settings_path()
            if local_path is None:
                return False
            state_dict = self._yaml_read(local_path)
            if not state_dict:
                return False
            self.load_from_dict(state_dict)
            logger.info("Local state loaded from %s", local_path)
            return True
        except (OSError, yaml.YAMLError) as e:
            logger.error("Error loading local state from YAML: %s", e)
            return False

    def save_to_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            if yaml_path is not None:
                # Backward-compatible single-file save.
                path = Path(yaml_path)
                state_dict = self._sort_state_dict(self.get_saveable_state_dict())
                self._yaml_write(path, state_dict)
                return True

        except (OSError, yaml.YAMLError):
            logger.exception("Error saving state to %s", yaml_path)
            return False

        # Refresh panel/window layout snapshots (set by MetaWidget) so the
        # periodic auto-save always persists the live arrangement. A snapshot
        # failure must not block saving the rest of the state (this runs in
        # the auto-save QTimer slot, where an uncaught exception would
        # silently kill every save).
        provider = getattr(self, "_layout_snapshot_provider", None)
        if provider is not None:
            try:
                provider()
            except Exception:
                logger.exception("Layout snapshot failed; saving state without a layout refresh")

        ok = True
        try:
            global_path = self._global_settings_path()
            global_state = self._sort_state_dict(self.get_saveable_state_dict(scope=AppStateSpec.SCOPE_GLOBAL))
            self._yaml_write(global_path, global_state)
        except (OSError, yaml.YAMLError):
            logger.exception("Error saving global state to %s", self._global_settings_path())
            ok = False

        try:
            local_path = self._local_settings_path()
            if local_path is not None:
                local_state = self._sort_state_dict(self.get_saveable_state_dict(scope=AppStateSpec.SCOPE_LOCAL))
                self._yaml_write(local_path, local_state)
        except (OSError, yaml.YAMLError):
            logger.exception("Error saving local state to %s", self._local_settings_path())
            ok = False

        return ok

    def load_from_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            if yaml_path is not None:
                path = Path(yaml_path)
                if not path.exists():
                    logger.warning("YAML file %s not found, using defaults", path)
                    return False
                state_dict = self._yaml_read(path)
                self.load_from_dict(state_dict)
                logger.info("State loaded from %s", path)
                return True

            loaded_any = False

            global_path = self._global_settings_path()
            global_state = self._yaml_read(global_path)
            # Drop per-dataset keys left in the global file by older versions
            # (e.g. navigate_mode/slider_scope) — they must never become a
            # sticky default that follows the user into the next dataset.
            local_keys = AppStateSpec.saveable_attributes(scope=AppStateSpec.SCOPE_LOCAL)
            global_state = {k: v for k, v in global_state.items() if k not in local_keys}
            if global_state:
                self.load_from_dict(global_state)
                logger.info("Global state loaded from %s", global_path)
                loaded_any = True

            if self.load_local_settings():
                loaded_any = True

            if not loaded_any:
                logger.warning("No settings YAML found, using defaults")
            return loaded_any
        except (OSError, yaml.YAMLError) as e:
            logger.error("Error loading state from YAML: %s", e)
            return False

    def delete_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            if yaml_path is not None:
                p = Path(yaml_path)
                if p.exists():
                    p.unlink()
                    logger.info("Deleted YAML file %s", yaml_path)
                    return True
                logger.warning("YAML file %s does not exist", yaml_path)
                return False

            deleted_any = False
            global_path = self._global_settings_path()
            if global_path.exists():
                global_path.unlink()
                logger.info("Deleted YAML file %s", global_path)
                deleted_any = True

            local_path = self._local_settings_path()
            if local_path is not None and local_path.exists():
                local_path.unlink()
                logger.info("Deleted YAML file %s", local_path)
                deleted_any = True

            if not deleted_any:
                logger.warning("No YAML settings files found to delete")
            return deleted_any
        except OSError as e:
            logger.error("Error deleting YAML file: %s", e)
            return False

    def stop_auto_save(self):
        if self._auto_save_timer.isActive():
            self._auto_save_timer.stop()
            self.save_to_yaml()

    # --- Interval label helpers ---
    def get_trial_intervals(self, trial) -> pd.DataFrame:
        return get_trial_from_tsv(self._all_labels_df, trial)

    def set_trial_intervals(self, trial, df: pd.DataFrame) -> None:
        self._all_labels_df = set_trial_in_tsv(self._all_labels_df, trial, df)
        nav = getattr(self, "navigation_widget", None)
        if nav is not None and hasattr(nav, "on_labels_changed"):
            nav.on_labels_changed()

    def get_trial_meta(self, trial) -> dict:
        return get_trial_meta(self._all_labels_df, trial)

    def set_trial_meta_attr(self, trial, key: str, value) -> None:
        self._all_labels_df = set_trial_meta_attr(self._all_labels_df, trial, key, value)

    def get_global_meta_attr(self, key: str, default=0):
        """Check if ALL trials with labels have a meta attr set to truthy."""
        if self._all_labels_df is None or self._all_labels_df.empty:
            return default
        if not self.trials:
            return default
        trials_with_labels = set(self._all_labels_df["trial"].unique())
        if not trials_with_labels:
            return default
        for trial in self.trials:
            if trial not in trials_with_labels:
                continue  # no labels → nothing to correct, skip
            meta = get_trial_meta(self._all_labels_df, trial)
            if not meta.get(key, 0):
                return default
        return 1

    def set_global_meta_attr(self, key: str, value) -> None:
        """Set a meta attr on ALL trials."""
        for trial in self.trials:
            self._all_labels_df = set_trial_meta_attr(self._all_labels_df, trial, key, value)

    def _get_downsampled_suffix(self) -> str:
        if self.downsample_factor_used:
            return f"_downsampled_{self.downsample_factor_used}x"
        return ""

    def save_labels(self, remote_path: str | None = None, remote_mode: str | None = None) -> None:
        """Save labels to active file (canonical or predictions TSV) + local backup + optional remote backup.

        Parameters
        ----------
        remote_path : str, optional
            Folder path for remote backup. Falls back to ``self.remote_backup_path``.
        remote_mode : str, optional
            "timestamp", "overwrite", or "git". Falls back to ``self.remote_backup_mode``.
        """
        if self._all_labels_df is None:
            return

        effective_remote_path = remote_path or self.remote_backup_path or None
        effective_remote_mode = remote_mode if remote_mode is not None else self.remote_backup_mode

        nc_path = Path(self.nc_file_path)
        suffix = self._get_downsampled_suffix()
        stem = f"{nc_path.stem}{suffix}"

        # Enrich with computed columns (duration, sequence, global timing, trial attrs)
        from ethograph.labels.export import enrich_labels_df

        keep_attrs = self.trial_conditions if self.trial_conditions else []
        enriched = enrich_labels_df(
            self._all_labels_df,
            nwb_alignment=self.nwb_alignment,
            keep_attrs=keep_attrs,
            dt=self.dt,
            metadata_df=self.metadata_df,
        )
        save_df = enriched if not enriched.empty else self._all_labels_df

        # 1. Primary file: use _labels_file_path if set (predictions/custom), otherwise canonical
        if self._labels_file_path and Path(self._labels_file_path).exists():
            primary_tsv = Path(self._labels_file_path)
        else:
            primary_tsv = labels_tsv_path(nc_path, suffix)
        save_labels_tsv(primary_tsv, save_df)

        # 2. Local backup with timestamp
        backup_dir = nc_path.parent / "labels" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_labels_tsv(backup_dir / f"{stem}_labels_{timestamp}.tsv", save_df)

        # 3. Remote backup (optional)
        # remote_path_depth controls how many parent folders to mirror inside remote_root:
        #   0 = flat (Trial_data_labels.tsv)
        #   1 = behav/Trial_data_labels.tsv
        #   2 = ses-000/behav/Trial_data_labels.tsv  (etc.)
        if self.remote_backup_enabled and effective_remote_path:
            remote_root = Path(effective_remote_path)
            depth = self.remote_path_depth
            if depth > 0:
                parent_parts = nc_path.parent.parts[1:]  # strip drive / leading '/'
                mirror_parts = parent_parts[max(0, len(parent_parts) - depth) :]
                remote_dir = remote_root.joinpath(*mirror_parts)
            else:
                remote_dir = remote_root
            remote_dir.mkdir(parents=True, exist_ok=True)
            remote_file = remote_dir / f"{stem}_labels.tsv"
            if effective_remote_mode in ("overwrite", "git"):
                save_labels_tsv(remote_file, save_df)
                if effective_remote_mode == "git":
                    auto_git_commit(remote_file)
            else:
                save_labels_tsv(remote_dir / f"{stem}_labels_{timestamp}.tsv", save_df)

        notify(f"Saved labels: {primary_tsv.name}")
        self.changes_saved = True
