"""Settings that the user can modify and are saved in gui_settings.yaml"""

import gc
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from napari.settings import get_settings
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, QTimer, Signal

import ethograph as eto
from ethograph.gui.plots_timeseriessource import TrialAlignment, TimeRange

from .makepretty import find_combo_index
from ethograph.utils.label_intervals import (
    empty_intervals,
    intervals_to_xr,
    xr_to_intervals,
)


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
        "window_size": (float, 5.0, True),
        "audiotrace_visible": (bool, True, True, SCOPE_LOCAL),
        "spectrogram_visible": (bool, True, True, SCOPE_LOCAL),
        "neo_visible": (bool, True, True, SCOPE_LOCAL),
        "ephys_visible": (bool, True, True, SCOPE_LOCAL),
        "featureplot_visible": (bool, True, True, SCOPE_LOCAL),
        "video_viewer_visible": (bool, True, True, SCOPE_LOCAL),
        "pose_markers_visible": (bool, True, True, SCOPE_LOCAL),
        "labels_visible": (bool, True, True, SCOPE_LOCAL),
        "feature_view_mode": (str, "LinePlot", True, SCOPE_LOCAL),

        # Data
        "ds": (xr.Dataset | None, None, False),
        "ds_temp": (xr.Dataset | None, None, False),
        "dt": (xr.DataTree | None, None, False),
        "label_ds": (xr.Dataset | None, None, False),
        "label_dt": (xr.DataTree | None, None, False),
        "pred_ds": (xr.Dataset | None, None, False),
        "pred_dt": (xr.DataTree | None, None, False),
        "trial_conditions": (list | None, None, False),
        "import_labels_nc_data": (bool, False, True),
        "fps_playback": (float, 30.0, True),
        "audio_playback_speed": (float, 1.0, True),
        "av_speed_coupled": (bool, True, True),
        "skip_frames": (bool, False, True),
        "filter_warnings": (bool, False, True),
        "center_playback": (bool, False, True),
        "time_jump_ms": (float, 100.0, True),
        "time": (xr.DataArray | None, None, False), # for feature variables (e.g. 'time' or 'time_aux')
        "label_intervals": (pd.DataFrame | None, None, False),
        "trial_alignment": (TrialAlignment | None, None, False),
        "trials": (list[int | str], [], False),
        "downsample_enabled": (bool, False, True),
        "downsample_factor": (int, 100, True),
        
        # Boolean
        "has_video": (bool, False, False),
        "has_pose": (bool, False, False),
        "has_audio": (bool, False, False),
        "has_neo": (bool, False, False),
        "has_kilosort": (bool, False, False),
        

        # Paths 
        "nc_file_path": (str | None, None, False),
        "video_folder": (str | None, None, True, SCOPE_LOCAL),
        "remote_video": (bool, False, True),
        "audio_folder": (str | None, None, True, SCOPE_LOCAL),
        "pose_folder": (str | None, None, True, SCOPE_LOCAL),
        "ephys_path": (str | None, None, True, SCOPE_LOCAL),
        "kilosort_folder": (str | None, None, True, SCOPE_LOCAL),
        

        "video_path": (str | None, None, False),
        "audio_path": (str | None, None, False),
        "pose_path": (str | None, None, False), 
        
        
        
        
        "pose_hide_threshold": (float, 0.9, True),

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
        "space_plot_type": (str, "Layers", True),
        "slot2_sel": (str | None, None, True),
        "slot3_sel": (str | None, None, True),
        "lock_axes": (bool, False, False),
        "spec_colormap": (str, "CET-R4", True),
        "spec_levels_mode": (str, "auto", True),

        # All checkbox states for dimension combos (e.g., {"keypoints": True, "space": False})
        "all_checkbox_states": (dict[str, bool], {}, True),

        # Audio processing
        "audio_cp_hop_length_ms": (float, 5.0, True),
        "audio_cp_min_level_db": (float, -70.0, True),
        "audio_cp_min_syllable_length_s": (float, 0.02, True),
        "audio_cp_silence_threshold": (float, 0.1, True),
        "show_changepoints": (bool, True, True),
        "apply_changepoint_correction": (bool, True, True),
        "automatic_min_label_length_s": (float, 1e-3, True),
        "automatic_stitch_gap_s": (float, 0.0, True),
        "save_tsv_enabled": (bool, True, True),

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
    def get_scope(cls, key):
        return cls.get_meta(key)[3]

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


    # Removed labels_modified and verification_changed signals for simplification
    trial_changed = Signal()
    GLOBAL_SETTINGS_FILENAME = "gui_settings.yaml"
    LOCAL_SETTINGS_FILENAME = "local_settings.yaml"
    SETTINGS_DIRNAME = ".ethograph"
    _TIME_REFRESH_KEYS = {"ds", "dt", "video", "video_path", "audio_path", "window_size"}


    def __init__(self, yaml_path: str | None = None, auto_save_interval: int = 30000):
        super().__init__()
        object.__setattr__(self, "_values", {})
        for var in AppStateSpec.VARS:
            _, default, _, _ = AppStateSpec.get_meta(var)
            self._values[var] = default

        self.audio_source_map: dict[str, tuple[str, int]] = {}
        self.ephys_source_map: dict[str, tuple[str, str, int]] = {}
        self.ephys_stream_sel: str | None = None
        self._suspend_local_autoload = False

        self.settings = get_settings()
        self._yaml_path = yaml_path or "gui_settings.yaml"
        self._auto_save_timer = QTimer()
        self._auto_save_timer.timeout.connect(self.save_to_yaml)
        self._auto_save_timer.start(auto_save_interval)

    @property
    def video_fps(self) -> float:
        video = getattr(self, 'video', None)
        if video is None:
            return 1
        else:
            return video.fps

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
    def trial_bounds(self) -> TimeRange | None:
        """Time range for the current trial, sourced from TrialAlignment.trial_range."""
        alignment = getattr(self, 'trial_alignment', None)
        if alignment is not None:
            return alignment.trial_range
        return None

    @property
    def time_coord(self) -> xr.DataArray | None:
        """Get the time coordinate for the currently selected features."""
        ds = getattr(self, 'ds', None)
        features_sel = getattr(self, 'features_sel', None)
        if ds is not None and features_sel in ds.data_vars:
            return eto.get_time_coord(ds[features_sel])
        return None
        

    def get_with_default(self, key):
        """Return value from app state, or default from AppStateSpec if None."""
        value = getattr(self, key, None)
        if value is None:
            value = AppStateSpec.get_default(key)
        return value

    

    def get_feature_sr(self, position: bool = False) -> float | None:
        ds = getattr(self, "ds", None)
        feature_sel = getattr(self, "features_sel", None)
        if ds is None:
            return None
        if position:
            tc = eto.get_time_coord(ds["position"])
        elif feature_sel and feature_sel in ds.data_vars:
            tc = eto.get_time_coord(ds[feature_sel])
        if tc is None or len(tc) < 2:
            return None
        return float(1.0 / np.median(np.diff(tc)))



    def get_ephys_source(self) -> tuple[str | None, str, int]:
        """Get ephys file path, stream_id, and channel index from current ephys_stream_sel.

        Returns (ephys_path, stream_id, channel_idx) tuple. Uses ephys_source_map
        to resolve the display name.
        """
        import os

        stream_sel = getattr(self, 'ephys_stream_sel', None)
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
            base_ephys_path = getattr(self, 'ephys_path', None)
            if not base_ephys_path:
                return None, stream_id, channel_idx
            ephys_path = os.path.normpath(
                os.path.join(os.path.dirname(base_ephys_path), filename)
            )

        return ephys_path, stream_id, channel_idx



    def get_audio_source(self) -> tuple[str | None, int]:
        """Get audio file path and channel index from current mics_sel.

        Returns (audio_path, channel_idx) tuple. Uses audio_source_map to resolve
        the display name to (mic_file, channel_idx).
        """
        import os

        mics_sel = getattr(self, 'mics_sel', None)
        if not mics_sel or not self.audio_source_map:
            return None, 0

        mic_file, channel_idx = self.audio_source_map.get(mics_sel, (mics_sel, 0))

        audio_folder = getattr(self, 'audio_folder', None)
        if not audio_folder or not mic_file:
            return None, channel_idx

        audio_path = os.path.normpath(os.path.join(audio_folder, mic_file))
        return audio_path, channel_idx



    def __getattr__(self, name):
        # Check for class attributes/properties first
        cls = type(self)
        if hasattr(cls, name):
            attr = getattr(cls, name)
            # If it's a property, use its getter
            if hasattr(attr, '__get__'):
                return attr.__get__(self)
            return attr
        if name in AppStateSpec.VARS:
            return self._values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ("time", "_values", "settings", "_yaml_path", "_auto_save_timer", "navigation_widget", "lineplot", "audio_source_map", "ephys_source_map", "ephys_stream_sel", "_suspend_local_autoload"):
            super().__setattr__(name, value)
            return

        if name in AppStateSpec.VARS:
            type_hint = AppStateSpec.get_type(name)
            if not check_type(value, type_hint):
                raise TypeError(f"{name}: expected {type_hint}, got {type(value).__name__} = {value!r}")

            old_value = self._values.get(name)
            self._values[name] = value

            signal = getattr(self, f"{name}_changed", None)
            if signal and old_value is not value:
                signal.emit(value)

            if name == "nc_file_path" and not self._suspend_local_autoload:
                self.load_local_settings()

            return

        super().__setattr__(name, value)


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
                if coord_dtype.kind in ('i', 'u'):
                    ds_kwargs[dim] = int(output)
                else:
                    ds_kwargs[dim] = str(output)
            else:
                # Dim without coord - assume integer index
                ds_kwargs[dim] = int(output)

        return ds_kwargs
            


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
            print(f"Error updating combo box for {type_key}: {e}")

    def _cycle_combo_box(self, type_key, data_widget):
        """Cycle the combo box to the next item when no previous selection exists."""
        try:
            combo = data_widget.io_widget.combos.get(type_key) or data_widget.combos.get(type_key)
            if combo is not None and combo.count() > 1:
                next_index = (combo.currentIndex() + 1) % combo.count()
                combo.setCurrentIndex(next_index)
        except (AttributeError, TypeError) as e:
            print(f"Error cycling combo box for {type_key}: {e}")

    # --- Save/Load methods ---
    PATH_SUFFIXES = ("_path", "_folder")

    def _global_settings_path(self) -> Path:
        return Path.home() / self.SETTINGS_DIRNAME / self.GLOBAL_SETTINGS_FILENAME

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
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(state_dict, f, default_flow_style=False, sort_keys=False)

    def _to_native(self, value):
        """Convert numpy types to native Python types for YAML serialization."""
        if hasattr(value, 'item'):
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
                        print(f"Error accessing {attr}: {exc}")
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
        """Print all yaml-persisted app state vars, grouped by category."""
        _CATEGORY_LABELS = {0: "Paths", 1: "Selections", 2: "Booleans", 3: "Strings", 4: "Numbers", 5: "Dicts"}

        def _category_key(item):
            key, value = item
            if isinstance(value, dict):
                return 5
            if any(key.endswith(s) for s in self.PATH_SUFFIXES):
                return 0
            if key.endswith("_sel") or key.endswith("_sel_previous"):
                return 1
            if isinstance(value, bool):
                return 2
            if isinstance(value, str):
                return 3
            return 4

        state = self.get_saveable_state_dict()
        current_cat = None
        for key, value in sorted(state.items(), key=lambda item: (_category_key(item), item[0])):
            cat = _category_key((key, value))
            if cat != current_cat:
                print(f"\n{'='*50}")
                print(f"  {_CATEGORY_LABELS[cat]}")
                print(f"{'='*50}")
                current_cat = cat
            print(f"  {key}: {value}")

    def load_from_dict(self, state_dict: dict):
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
            print(f"Local state loaded from {local_path}")
            return True
        except (OSError, yaml.YAMLError) as e:
            print(f"Error loading local state from YAML: {e}")
            return False

    def save_to_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            if yaml_path is not None:
                # Backward-compatible single-file save.
                path = Path(yaml_path)
                state_dict = self._sort_state_dict(self.get_saveable_state_dict())
                self._yaml_write(path, state_dict)
                return True

            global_path = self._global_settings_path()
            global_state = self._sort_state_dict(self.get_saveable_state_dict(scope=AppStateSpec.SCOPE_GLOBAL))
            self._yaml_write(global_path, global_state)

            local_path = self._local_settings_path()
            if local_path is not None:
                local_state = self._sort_state_dict(self.get_saveable_state_dict(scope=AppStateSpec.SCOPE_LOCAL))
                self._yaml_write(local_path, local_state)

            return True
        except (OSError, yaml.YAMLError) as e:
            print(f"Error saving state to YAML: {e}")
            return False

    def load_from_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            if yaml_path is not None:
                path = Path(yaml_path)
                if not path.exists():
                    print(f"YAML file {path} not found, using defaults\n")
                    return False
                state_dict = self._yaml_read(path)
                self.load_from_dict(state_dict)
                print(f"State loaded from {path}\n")
                return True

            loaded_any = False

            global_path = self._global_settings_path()
            global_state = self._yaml_read(global_path)
            if global_state:
                self.load_from_dict(global_state)
                print(f"Global state loaded from {global_path}")
                loaded_any = True

            if self.load_local_settings():
                loaded_any = True

            if not loaded_any:
                print("No settings YAML found, using defaults\n")
            return loaded_any
        except (OSError, yaml.YAMLError) as e:
            print(f"Error loading state from YAML: {e}")
            return False
        
    def delete_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            if yaml_path is not None:
                p = Path(yaml_path)
                if p.exists():
                    p.unlink()
                    print(f"Deleted YAML file {yaml_path}")
                    return True
                print(f"YAML file {yaml_path} does not exist")
                return False

            deleted_any = False
            global_path = self._global_settings_path()
            if global_path.exists():
                global_path.unlink()
                print(f"Deleted YAML file {global_path}")
                deleted_any = True

            local_path = self._local_settings_path()
            if local_path is not None and local_path.exists():
                local_path.unlink()
                print(f"Deleted YAML file {local_path}")
                deleted_any = True

            if not deleted_any:
                print("No YAML settings files found to delete")
            return deleted_any
        except OSError as e:
            print(f"Error deleting YAML file: {e}")
            return False
    
    def stop_auto_save(self):
        if self._auto_save_timer.isActive():
            self._auto_save_timer.stop()
            self.save_to_yaml()

    # --- Interval label helpers ---
    def get_trial_intervals(self, trial) -> pd.DataFrame:
        if self.label_dt is None:
            return empty_intervals()
        trial_ds = self.label_dt.trial(trial)
        if "onset_s" in trial_ds.data_vars:
            return xr_to_intervals(trial_ds)
        return empty_intervals()

    def set_trial_intervals(self, trial, df: pd.DataFrame) -> None:
        if self.label_dt is None:
            return
        interval_ds = intervals_to_xr(df)
        old_ds = self.label_dt.trial(trial)
        interval_ds.attrs = old_ds.attrs.copy()
        if "labels_confidence" in old_ds.data_vars:
            interval_ds["labels_confidence"] = old_ds["labels_confidence"]
        self.label_dt.update_trial(trial, lambda _: interval_ds)

    def _get_downsampled_suffix(self) -> str:
        """Get suffix for downsampled files."""
        if self.downsample_factor_used:
            return f"_downsampled_{self.downsample_factor_used}x"
        return ""
    
    def _save_labels_tsv(self, nc_path, suffix):        
        tsv_path = nc_path.parent / f"{nc_path.stem}{suffix}_labels.tsv"        
        keep_attrs = self.trial_conditions if self.trial_conditions is not None else []
        df = eto.trees_to_df(self.dt, keep_attrs)
        df.to_csv(tsv_path, index=False, sep='\t', encoding='utf-8-sig')
                    

    def save_labels(self):
        """Save only updated labels to preserve data integrity of other variables."""

        nc_path = Path(self.nc_file_path)
        suffix = self._get_downsampled_suffix()

        # Save label seperately as backup
        labels_dir = nc_path.parent / "labels"
        labels_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_filename = f"{nc_path.stem}{suffix}_labels_{timestamp}{nc_path.suffix}"
        versioned_path = labels_dir / versioned_filename

        self.label_dt.save(versioned_path)
        show_info(f"✅ Saved: {Path(versioned_path).name}")

        self.changes_saved = True


    def save_file(self) -> None:
        if getattr(self.dt, "attrs", {}).get("nwb_source_path", "").startswith(("http://", "https://", "s3://")):
            show_info(
                "Full save is unavailable for remote NWB files. "
                "Use 'Save labels' instead."
            )
            return

        nc_path = Path(self.nc_file_path)
        suffix = self._get_downsampled_suffix()
        if self.save_tsv_enabled:
            self._save_labels_tsv(nc_path, suffix)

        if suffix:
            save_path = nc_path.parent / f"{nc_path.stem}{suffix}{nc_path.suffix}"
            updated_dt = self.dt.overwrite_with_labels(self.label_dt)

            
            updated_dt.save(save_path)
            updated_dt.close()
            show_info(f"✅ Saved downsampled: {save_path.name}")
        else:
            updated_dt = self.dt.overwrite_with_labels(self.label_dt)
            updated_dt.load()
            self.dt.close()

            updated_dt.save(nc_path)

            self.dt = eto.open(nc_path)
            show_info(f"✅ Saved: {nc_path.name}")