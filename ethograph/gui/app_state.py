"""Settings that the user can modify and are saved in gui_settings.yaml"""

import gc
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from napari.settings import get_settings
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, QTimer, Signal

from ethograph import TrialTree
from ethograph.utils.data_utils import get_time_coord, trees_to_df
from ethograph.utils.label_intervals import (
    empty_intervals,
    intervals_to_xr,
    xr_to_intervals,
)
from .plots_spectrogram import SharedAudioCache


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
    # Variable name: (type, default, save_to_yaml)
    VARS = {
        # Video
        "current_frame": (int, 0, False),
        "changes_saved": (bool, True, False),
        "num_frames": (int, 0, False),
        "_info_data": (dict[str, Any], {}, False),
        "sync_state": (str | None, None, False),
        "window_size": (float, 3.0, True),

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
        "skip_frames": (bool, False, True),
        "time": (xr.DataArray | None, None, False), # for feature variables (e.g. 'time' or 'time_aux')
        "label_intervals": (pd.DataFrame | None, None, False),
        "trials": (list[int | str], [], False),
        "downsample_enabled": (bool, False, True),
        "downsample_factor": (int, 100, True),

        # Paths 
        "nc_file_path": (str | None, None, True),
        "video_folder": (str | None, None, True),
        "audio_folder": (str | None, None, True),
        "pose_folder": (str | None, None, True),
        "video_path": (str | None, None, True),
        "audio_path": (str | None, None, True),
        "pose_path": (str | None, None, True),
        "pose_hide_threshold": (float, 0.9, True),
        "ephys_folder": (str | None, None, True),
        "ephys_path": (str | None, None, True),

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
        "space_plot_type": (str, "Layer controls", True),
        "lock_axes": (bool, False, False),
        "spec_colormap": (str, "CET-R4", True),
        "spec_levels_mode": (str, "auto", True),

        # All checkbox states for dimension combos (e.g., {"keypoints": True, "space": False})
        "all_checkbox_states": (dict[str, bool], {}, True),

        # Audio processing
        "noise_reduce_enabled": (bool, False, True),
        "noise_reduce_prop_decrease": (float, 1.0, True),
        "audio_cp_hop_length_ms": (float, 5.0, True),
        "audio_cp_min_level_db": (float, -70.0, True),
        "audio_cp_min_syllable_length_s": (float, 0.02, True),
        "audio_cp_silence_threshold": (float, 0.1, True),
        "show_changepoints": (bool, True, True),
        "apply_changepoint_correction": (bool, True, True),
        "save_tsv_enabled": (bool, True, True),

        # Envelope / energy (general, used by both heatmap and overlay)
        "energy_metric": (str, "amplitude_envelope", True),
        "env_rate": (float, 2000.0, True),
        "env_cutoff": (float, 500.0, True),
        "freq_cutoffs_min": (float, 500.0, True),
        "freq_cutoffs_max": (float, 10000.0, True),
        "smooth_win": (float, 2.0, True),
        "band_env_min": (float, 300.0, True),
        "band_env_max": (float, 6000.0, True),
        "band_env_rate": (float, 1000.0, True),

        # Heatmap-specific display
        "heatmap_exclusion_percentile": (float, 98.0, True),
        "heatmap_colormap": (str, "RdBu_r", True),
        "heatmap_normalization": (str, "per_channel", True),
    }

    @classmethod
    def get_default(cls, key):
        if key in cls.VARS:
            return cls.VARS[key][1]
        raise KeyError(f"No default for key: {key}")

    @classmethod
    def get_type(cls, key):
        if key in cls.VARS:
            return cls.VARS[key][0]
        raise KeyError(f"No type for key: {key}")

    @classmethod
    def saveable_attributes(cls) -> set[str]:
        return {k for k, (_, _, save) in cls.VARS.items() if save}


class ObservableAppState(QObject):
    """State container with change notifications and computed properties."""

    # Signals for state changes (auto-derive signal type from type hint)
    for var, (type_hint, _, _) in AppStateSpec.VARS.items():
        locals()[f"{var}_changed"] = Signal(get_signal_type(type_hint))


    labels_modified = Signal()
    verification_changed = Signal()
    trial_changed = Signal()


    def __init__(self, yaml_path: str | None = None, auto_save_interval: int = 30000):
        super().__init__()
        object.__setattr__(self, "_values", {})
        for var, (_, default, _) in AppStateSpec.VARS.items():
            self._values[var] = default

        self.audio_source_map: dict[str, tuple[str, int]] = {}
        self.ephys_source_map: dict[str, tuple[str, str, int]] = {}

        self.settings = get_settings()
        self._yaml_path = yaml_path or "gui_settings.yaml"
        self._auto_save_timer = QTimer()
        self._auto_save_timer.timeout.connect(self.save_to_yaml)
        self._auto_save_timer.start(auto_save_interval)



    @property
    def sel_attrs(self) -> dict:
        """
        Return all attributes ending with _sel or _sel_previous as a dict.
        """
        result = {}
        for attr in dir(self):
            if attr.endswith("_sel") or attr.endswith("_sel_previous"):
                value = getattr(self, attr, None)
                if not callable(value):
                    result[attr] = value
        return result
    

    def get_with_default(self, key):
        """Return value from app state, or default from AppStateSpec if None."""
        value = getattr(self, key, None)
        if value is None:
            value = AppStateSpec.get_default(key)
        return value


    def set_time(self, feature_sel=None):
        """Set app_state.time to an audio-rate time coordinate for Audio Waveform."""
                
        if feature_sel is None:
            feature_sel = self.features_sel

        if feature_sel == "Audio Waveform":
            audio_path, _ = self.get_audio_source()
            loader = SharedAudioCache.get_loader(audio_path)
            trial_onset = float(self.ds.attrs.get('trial_onset', 0.0))
            trial_duration = float(self.ds.time.values[-1]) + 1.0 / self.ds.attrs['fps']

            start_sample = int(trial_onset * loader.rate)
            end_sample = int((trial_onset + trial_duration) * loader.rate)
            n_samples = min(end_sample, len(loader)) - max(0, start_sample)

            time_values = np.arange(n_samples) / loader.rate
            self.time = xr.DataArray(
                time_values, dims=["time_audio"],
                coords={"time_audio": time_values}, name="time_audio",
            )

        elif feature_sel in self.ephys_source_map:
            from .plots_ephystrace import SharedEphysCache

            ephys_path, stream_id, _ = self.get_ephys_source()
            loader = SharedEphysCache.get_loader(ephys_path, stream_id)
            if loader is not None:
                trial_onset = float(self.ds.attrs.get('trial_onset', 0.0))
                trial_duration = float(self.ds.time.values[-1]) + 1.0 / self.ds.attrs['fps']

                start_sample = int(trial_onset * loader.rate)
                end_sample = int((trial_onset + trial_duration) * loader.rate)
                n_samples = min(end_sample, len(loader)) - max(0, start_sample)

                time_values = np.arange(n_samples) / loader.rate
                self.time = xr.DataArray(
                    time_values, dims=["time_ephys"],
                    coords={"time_ephys": time_values}, name="time_ephys",
                )

        else:
            self.time = get_time_coord(self.ds[feature_sel])  
        


    def get_ephys_source(self) -> tuple[str | None, str, int]:
        """Get ephys file path, stream_id, and channel index from current features_sel.

        Returns (ephys_path, stream_id, channel_idx) tuple. Uses ephys_source_map
        to resolve the display name.
        """
        import os

        feature_sel = getattr(self, 'features_sel', None)
        if not feature_sel or not self.ephys_source_map:
            return None, "0", 0

        entry = self.ephys_source_map.get(feature_sel)
        if entry is None:
            return None, "0", 0

        filename, stream_id, channel_idx = entry

        ephys_folder = getattr(self, 'ephys_folder', None)
        if not ephys_folder or not filename:
            return None, stream_id, channel_idx

        ephys_path = os.path.normpath(os.path.join(ephys_folder, filename))
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
        if name in AppStateSpec.VARS:
            return self._values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ("time", "_values", "settings", "_yaml_path", "_auto_save_timer", "navigation_widget", "lineplot", "audio_source_map", "ephys_source_map"):
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
        old_value = getattr(self, attr_name, None)

        if old_value is not None and old_value != currentValue and type_key in ["features", "keypoints", "individuals", "cameras", "mics"]:
            setattr(self, prev_attr_name, old_value)
        
        if type_key == "features" and self.ds:
            self.set_time(feature_sel=currentValue)
            


        setattr(self, attr_name, currentValue)


            
    def set_key_sel_previous(self, type_key, previousValue):
        """Set previous selection for a given key."""
        prev_attr_name = f"{type_key}_sel_previous"
        setattr(self, prev_attr_name, previousValue)

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
            
        if type_key == "features":
            self.set_time()
            
   
    
    def _update_combo_box(self, type_key, new_value, data_widget):
        """Update the corresponding combo box in the UI and trigger its change signal."""
        try:
            combo = data_widget.io_widget.combos.get(type_key) or data_widget.combos.get(type_key)

            if combo is not None:
                index = combo.findText(str(new_value))
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
    def _to_native(self, value):
        """Convert numpy types to native Python types for YAML serialization."""
        if hasattr(value, 'item'):
            return value.item()
        return value

    def get_saveable_state_dict(self) -> dict:
        state_dict = {}
        for attr in AppStateSpec.saveable_attributes():
            value = self._values.get(attr)
            if value is not None and isinstance(value, (str, float, int, bool)):
                state_dict[attr] = self._to_native(value)
            elif isinstance(value, dict) and value:
                state_dict[attr] = value

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


    def load_from_dict(self, state_dict: dict):
        for key, value in state_dict.items():
            if value is None:
                continue
            if key in AppStateSpec.VARS or key.endswith("_sel") or key.endswith("_sel_previous"):
                setattr(self, key, value)

    def save_to_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            path = yaml_path or self._yaml_path
            state_dict = self.get_saveable_state_dict()
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(state_dict, f, default_flow_style=False, sort_keys=False)
            return True
        except (OSError, yaml.YAMLError) as e:
            print(f"Error saving state to YAML: {e}")
            return False

    def load_from_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            path = yaml_path or self._yaml_path
            if not Path(path).exists():
                print(f"YAML file {path} not found, using defaults\n")
                return False
            with open(path, encoding="utf-8") as f:
                state_dict = yaml.safe_load(f) or {}
            self.load_from_dict(state_dict)
            print(f"State loaded from {path}\n")
            return True
        except (OSError, yaml.YAMLError) as e:
            print(f"Error loading state from YAML: {e}")
            return False
        
    def delete_yaml(self, yaml_path: str | None = None) -> bool:
        try:
            path = yaml_path or self._yaml_path
            p = Path(path)
            if p.exists():
                p.unlink()
                print(f"Deleted YAML file {path}")
                return True
            else:
                print(f"YAML file {path} does not exist")
                return False
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
        trial_key = TrialTree.trial_key(trial)
        interval_ds = intervals_to_xr(df)
        old_ds = self.label_dt.trial(trial)
        interval_ds.attrs = old_ds.attrs.copy()
        if "labels_confidence" in old_ds.data_vars:
            interval_ds["labels_confidence"] = old_ds["labels_confidence"]
        self.label_dt[trial_key] = xr.DataTree(interval_ds)

    def _get_downsampled_suffix(self) -> str:
        """Get suffix for downsampled files."""
        if self.downsample_factor_used:
            return f"_downsampled_{self.downsample_factor_used}x"
        return ""
    
    def _save_labels_tsv(self, nc_path, suffix):        
        tsv_path = nc_path.parent / f"{nc_path.stem}{suffix}_labels.tsv"        
        keep_attrs = self.trial_conditions if self.trial_conditions is not None else []
        df = trees_to_df(self.dt, keep_attrs)
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

        self.label_dt.to_netcdf(versioned_path)
        show_info(f"✅ Saved: {Path(versioned_path).name}")

        self.changes_saved = True


    def save_file(self) -> None:


        nc_path = Path(self.nc_file_path)
        suffix = self._get_downsampled_suffix()
        if self.save_tsv_enabled:
            self._save_labels_tsv(nc_path, suffix)

        if suffix:
            save_path = nc_path.parent / f"{nc_path.stem}{suffix}{nc_path.suffix}"
            updated_dt = self.dt.overwrite_with_labels(self.label_dt)

            
            updated_dt.to_netcdf(save_path, mode='w')
            updated_dt.close()
            show_info(f"✅ Saved downsampled: {save_path.name}")
        else:
            updated_dt = self.dt.overwrite_with_labels(self.label_dt)
            updated_dt.load()
            self.dt.close()
            self.label_dt.close()

            updated_dt.save(nc_path)

            self.dt = TrialTree.open(nc_path)
            self.label_dt = self.dt.get_label_dt()
            show_info(f"✅ Saved: {nc_path.name}")