"""Refactored observable application state with napari video sync support."""

from pathlib import Path
from typing import Any
import numbers
import xarray as xr
import yaml
from napari.settings import get_settings
from qtpy.QtCore import QObject, QTimer, Signal
from datetime import datetime
from napari.utils.notifications import show_info
import time
from qtpy.QtWidgets import QApplication
from moveseg.utils.io import TrialTree
import numpy as np

class AppStateSpec:
    @classmethod
    def get_default(cls, key):
        """Return the default value for a given key from VARS."""
        if key in cls.VARS:
            return cls.VARS[key][1]
        raise KeyError(f"No default for key: {key}")

    # Variable name: (type, default, save_to_yaml, signal_type)
    VARS = {
        # Video 
        "current_frame": (int, 0, False, int),
        "changes_saved": (bool, True, False, bool),
        "num_frames": (int, 0, False, int),
        "_info_data": (dict[str, Any], {}, False, object),
        "sync_state": (str | None, None, False, object),        
        "window_size": (float, 5.0, True, object),
        "audio_buffer": (float | None, None, True, float),

        # Data
        # Seperation between raw data and label data (mutable) for data integrity
        "ds": (xr.Dataset | None, None, False, object),
        "ds_temp": (xr.Dataset | None, None, False, object), # for exploratory data manipulations
        "dt": (xr.DataTree | None, None, False, object),
        "label_ds": (xr.Dataset | None, None, False, object),
        "label_dt": (xr.DataTree | None, None, False, object),
        "pred_ds": (xr.Dataset | None, None, False, object),
        "pred_dt": (xr.DataTree | None, None, False, object),
        "import_labels_nc_data": (bool, False, True, bool),
        "fps_playback": (float, 30.0, True, float),
        "trials": (list[int], [], False, object),
        "trials_sel_condition_value": (str | None, None, True, object),
        "plot_spectrogram": (bool, False, True, bool),
        "load_s3d": (bool, False, False, bool),
        
        
        # Paths
        "nc_file_path": (str | None, None, True, str),
        "video_folder": (str | None, None, True, str),
        "audio_folder": (str | None, None, True, str),
        "tracking_folder": (str | None, None, True, str),
        "video_path": (str | None, None, True, str),
        "audio_path": (str | None, None, True, str),
        "tracking_path": (str | None, None, True, str),
         
        # Plotting
        "ymin": (float | None, None, True, object),
        "ymax": (float | None, None, True, object),
        "spec_ymin": (float | None, None, True, object),
        "spec_ymax": (float | None, None, True, object),
        "spec_buffer": (float | None, None, True, float),
        "video_buffer_size": (int, 300, True, int),
        "ready": (bool, False, False, bool),
        "nfft": (int, 1024, True, int),
        "hop_frac": (float, 0.5, True, float),
        "vmin_db": (float, -120.0, True, float),
        "vmax_db": (float, 20.0, True, float),
        "buffer_multiplier": (float, 5.0, True, float),
        "recompute_threshold": (float, 0.5, True, float),
        "cmap": (str, "magma", True, str),
        "percentile_ylim": (float, 99.5, True, float),
        "space_plot_type": (str, "Layer controls", True, str),
        
        "lock_axes": (bool, False, False, bool), # always started unlocked

    }


class AppState:
    def __init__(self):
        for var, (var_type, default, _, _) in AppStateSpec.VARS.items():
            setattr(self, var, default)

    def saveable_attributes(self) -> set[str]:
        return {k for k, (_, _, save, _) in AppStateSpec.VARS.items() if save}


class ObservableAppState(QObject):

    """State container with change notifications and computed properties."""

    # Signals for state changes
    for var, (_, _, _, signal_type) in AppStateSpec.VARS.items():
        locals()[f"{var}_changed"] = Signal(signal_type)

    data_updated = Signal()


    def __init__(self, yaml_path: str | None = None, auto_save_interval: int = 30000):
        super().__init__()
        object.__setattr__(self, "_state", AppState())

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

    def __getattr__(self, name):
        if name in AppStateSpec.VARS:
            return getattr(self._state, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        # Handle special attributes that don't go through state
        if name in (
            "_state",
            "settings",
            "_yaml_path",
            "_auto_save_timer",
            "navigation_widget",
            "lineplot",
        ):
            super().__setattr__(name, value)
            return

        # Handle state variables
        if name in AppStateSpec.VARS:
            old_value = getattr(self._state, name, None)
            setattr(self._state, name, value)

            # Emit signal if value changed
            signal = getattr(self, f"{name}_changed", None)
            if signal and old_value != value:
                signal.emit(value)

            return

        # Handle other attributes
        super().__setattr__(name, value)

    # --- Dynamic _sel variables ---
    def get_ds_kwargs(self):
        ds_kwargs = {}

        for dim in self.ds.dims:
            attr_name = f"{dim}_sel"
            if hasattr(self, attr_name):
                output = getattr(self, attr_name)
                if output is not None and output not in ["", "All", "None"]:
                    ds_kwargs[dim] = output

        return ds_kwargs
            


    def key_sel_exists(self, type_key: str) -> bool:
        """Check if a key selection exists for a given type."""
        return hasattr(self, f"{type_key}_sel")

    def get_key_sel(self, type_key: str):
        """Get current value for a given info key."""
        attr_name = f"{type_key}_sel"
        return getattr(self, attr_name, None)



    def set_key_sel(self, type_key, currentValue):
        """Set current value for a given info key."""
        if currentValue is None:
            return

        
        if type_key == "trials":
            currentValue = _normalize_to_native_type(currentValue)
            trial_type = type(self.trials[0])
            
            # Ensure it matches the trials list type
            if not isinstance(currentValue, trial_type):
                try:
                    currentValue = trial_type(currentValue)
                except (ValueError, TypeError) as e:
                    print(f"Failed to coerce trial selection '{currentValue}' to {trial_type}: {e}")
                    return


        attr_name = f"{type_key}_sel"
        prev_attr_name = f"{type_key}_sel_previous"
        old_value = getattr(self, attr_name, None)
        
        if old_value is not None and old_value != currentValue and type_key in ["features", "keypoints", "individuals", "cameras", "mics"]:
            setattr(self, prev_attr_name, old_value)
        
        setattr(self, attr_name, currentValue)

        if old_value != currentValue:
            self.data_updated.emit()
            
    def set_key_sel_previous(self, type_key, previousValue):
        """Set previous selection for a given key."""
        prev_attr_name = f"{type_key}_sel_previous"
        setattr(self, prev_attr_name, previousValue)

    def toggle_key_sel(self, type_key, data_widget=None):
        """Toggle between current and previous value for a given key."""
        attr_name = f"{type_key}_sel"
        prev_attr_name = f"{type_key}_sel_previous"
        
        current_value = getattr(self, attr_name, None)
        previous_value = getattr(self, prev_attr_name, None)
        
        if previous_value is not None:
            # Swap current and previous
            setattr(self, attr_name, previous_value)
            setattr(self, prev_attr_name, current_value)
            
            # Update UI combo box if data_widget is provided
            if data_widget is not None:
                self._update_combo_box(type_key, previous_value, data_widget)
            
            self.data_updated.emit()
    
    def _update_combo_box(self, type_key, new_value, data_widget):
        """Update the corresponding combo box in the UI and trigger its change signal."""
        try:
            # Check IOWidget combos first, then DataWidget combos
            combo = data_widget.io_widget.combos.get(type_key) or data_widget.combos.get(type_key)
            
            if combo is not None:
                combo.setCurrentText(str(new_value))
                combo.currentTextChanged.emit(combo.currentText())
        except (AttributeError, TypeError) as e:
            print(f"Error updating combo box for {type_key}: {e}")

    # --- Save/Load methods ---
    def get_saveable_state_dict(self) -> dict:
        state_dict = {}
        for attr in self._state.saveable_attributes():
            value = getattr(self._state, attr)
            if value is not None:
                # Only save if value is str, float, int, or bool
                if isinstance(value, (str, float, int, bool)):
                    state_dict[attr] = value
        
        # Save dynamic _sel attributes
        for attr in dir(self):
            if attr.endswith("_sel") or attr.endswith("_sel_previous"):
                try:
                    value = getattr(self, attr)
                    if not callable(value) and value is not None:
                        if isinstance(value, (str, float, numbers.Integral, bool)) and not (isinstance(value, str) and value.startswith("<xarray.DataArray")):
                            state_dict[attr] = value
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
                print(f"YAML file {path} not found, using defaults")
                return False
            with open(path, encoding="utf-8") as f:
                state_dict = yaml.safe_load(f) or {}
            self.load_from_dict(state_dict)
            print(f"State loaded from {path}")
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



    def save_labels(self):
        """Save only updated labels to preserve data integrity of other variables."""

        nc_path = Path(self.nc_file_path)
        
        # Save label seperately as backup
        labels_dir = nc_path.parent / "labels"
        labels_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_filename = f"{nc_path.stem}_labels_{timestamp}{nc_path.suffix}"
        versioned_path = labels_dir / versioned_filename

        self.label_dt.to_netcdf(versioned_path)
        show_info(f"✅ Saved: {Path(versioned_path).name}")
        
        self.changes_saved = True
        
        
    def save_file(self) -> None:
        nc_path = Path(self.nc_file_path)
        
        # Create temporary copy for saving
        temp_path = nc_path.with_suffix('.tmp.nc')
        
        updated_dt = self.dt.overwrite_with_labels(self.label_dt)
        updated_dt.to_netcdf(temp_path, mode='w')
        updated_dt.close()
        

        self.dt.close()
        temp_path.replace(nc_path)
        

        self.dt = TrialTree.load(nc_path)
        show_info(f"✅ Saved: {nc_path.name}")


def _normalize_to_native_type(value: Any) -> int | str:
    """Normalize value to native Python int or str based on content."""
    import numpy as np
    
    # Convert numpy types to native Python types first
    if isinstance(value, (np.generic, np.ndarray)):
        value = value.item() if isinstance(value, np.generic) else value.tolist()
    
    # Already a native int or str
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # Try to parse as int if it looks like one
        if value.strip().lstrip('-').isdigit():
            return int(value)
        return value
    
    # For other types, try int first, fall back to str
    try:
        return int(value)
    except (ValueError, TypeError):
        return str(value)