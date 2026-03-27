# Data Format Requirements

This page documents the xarray Dataset structure that EthoGraph expects.
Every trial lives inside a [TrialTree](trialtree.md) (an `xr.DataTree` subclass).
The requirements below apply to **each trial's `xr.Dataset`**.

---

## Required attributes (all scenarios)

Every dataset **must** have these attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `attrs["trial"]` | `int` | Trial identifier (1, 2, 3, ...). Must be unique across trials. |
| `attrs["fps"]` | `float` | Frame rate of the primary video. Required unless audio-only (see below). |
| `coords["individuals"]` | `str` array | Names of tracked subjects (e.g. `["mouse1", "mouse2"]`). |

```python
import xarray as xr

ds = xr.Dataset(
    coords={"individuals": ["bird1", "bird2"]},
)
ds.attrs["trial"] = 1
```

Validation is performed by `validate_required_attrs()` and `validate_dataset()` in
`ethograph/utils/validation.py`.

---

## Marking features for the GUI

The GUI populates its **Feature** dropdown from variables whose `attrs["type"]`
equals `"features"`. Any `xr.DataArray` you want to appear in the plot must be
tagged:

```python
ds["speed"] = xr.DataArray(
    speed_values,                       # shape: (time, keypoints, individuals)
    dims=["time", "keypoints", "individuals"],
    attrs={"type": "features"},         # <-- required
)
```

Every feature variable **must** have at least one dimension whose name contains
`"time"` (e.g. `time`, `time_accelerometer`, `time_video`). Different features may use
different time coordinates with different sampling rates -- the GUI handles this via `get_time_coord()` in `ethograph/utils/xr_utils.py`.

If you build a dataset from scratch with `dataset_to_basic_trialtree()` (`ethograph/utils/io.py`),
all data variables (except label columns) are auto-tagged as
`type="features"`.

---

## Specifying individuals

Individuals are stored as a **coordinate**, not an attribute. With multi-animal data, this allows the GUI to store separate labels and feature data for different individuals.

```python
ds = xr.Dataset(
    data_vars={
        "speed": xr.DataArray(
            speed_array,                # shape: (time, individuals)
            dims=["time", "individuals"],
            attrs={"type": "features"},
        ),
    },
    coords={
        "time": time_values,
        "individuals": ["mouse1", "mouse2", "mouse3"],
    },
)
```

When labelling, the selected individual filters which labels are shown and
created.

---

## Media files, session table, and timing

Media filenames, trial timing, and stream offsets are stored at the session
level on the TrialTree, not inside individual trial datasets. Filenames are
stored as **filename-only strings** (never full paths) so that datasets remain
valid when folders are moved or accessed from a different machine.

For full documentation of `dt.set_media()`, `dt.set_stream_offset()`,
`dt.start_time()`, and related methods, see
[TrialTree — Media files](trialtree.md#media-files) and
[TrialTree — Session table and timing](trialtree.md#session-table-and-timing).

---

## Scenario-specific requirements

### Video + Pose

The standard scenario. Provide `fps`, camera filenames, and optionally pose
filenames. Feature variables (position, velocity, speed, ...) should be tagged
with `type="features"`.

```python
import ethograph as eto

ds = xr.Dataset(
    data_vars={
        "position": xr.DataArray(pos, dims=["time", "space", "keypoints", "individuals"],
                                 attrs={"type": "features"}),
        "speed":    xr.DataArray(spd, dims=["time", "keypoints", "individuals"],
                                 attrs={"type": "features"}),
    },
    coords={
        "time": time_s,
        "space": ["x", "y"],
        "keypoints": ["nose", "tail"],
        "individuals": ["mouse1"],
    },
)
ds.attrs["trial"] = 1
ds.attrs["fps"] = 30.0

dt = eto.from_datasets([ds])
dt.set_media("video", [["trial001.mp4"]])
dt.set_media("pose", [["trial001.h5"]])
```

### Audio only (no video)

When cameras are absent and mics are present, the GUI enters **no-video
mode**: a time slider replaces the napari video player, and playback uses
`sounddevice`. Features are optional -- an audio-only file with just
microphone filenames is valid. `fps` is not required.

```python
ds = xr.Dataset(coords={"individuals": ["bird1"]})
ds.attrs["trial"] = 1

dt = eto.from_datasets([ds])
dt.set_media("audio", [["song_trial001.wav"]])
```

### Ephys with video/audio alignment

Ephys is a session-wide stream (one file covering all trials). The raw recording file is selected directly in the GUI — you do not need to call `dt.set_media("ephys", ...)` in your dataset creation script. If the ephys clock differs from the behavioural reference, record the offset so the trace aligns correctly:

```python
dt.set_stream_offset("ephys", 0.0)   # seconds; adjust to match your setup
```

See [Ephys data](ephys-data.md) for supported file formats, Kilosort folder setup, and channel mapping details. See [TrialTree — Stream offsets](trialtree.md#stream-offsets) for the full offset API.

---

## Color variables (optional)

Color data (e.g. angle-based RGB from pose) must have an `RGB` dimension of
size 3 and `attrs["type"] = "colors"`:

```python
ds["angle_rgb"] = xr.DataArray(
    rgb_values,                          # shape: (time, keypoints, individuals, 3)
    dims=["time", "keypoints", "individuals", "RGB"],
    attrs={"type": "colors"},
)
```

Values must be in `[0, 1]` (float) or `[0, 255]` (int). Validation is
performed by `validate_colors()` in `validation.py`.

To compute angle-based RGB automatically from pose data, use
`add_angle_rgb_to_ds()` from `ethograph/utils/io.py`:

```python
from ethograph.utils.io import add_angle_rgb_to_ds

ds = add_angle_rgb_to_ds(ds, smoothing_params={"sigma": 3})
# Creates ds["angles"] (type="features") and ds["angle_rgb"] (type="colors")
```

This function computes frame-to-frame angle changes in the xy plane via
`xr.apply_ufunc` across all individuals and keypoints, with optional Gaussian
smoothing.

---

## Changepoint variables (optional)

Changepoint arrays are binary (`0` or `1`) integer arrays that share the same
time dimension as their target feature. They require:

- `attrs["type"] = "changepoints"`
- `attrs["target_feature"]` -- name of the feature variable they annotate

```python
ds["speed_troughs"] = xr.DataArray(
    cp_binary,                           # shape: (time, keypoints, individuals), values 0 or 1
    dims=["time", "keypoints", "individuals"],
    attrs={
        "type": "changepoints",
        "target_feature": "speed",
    },
)
```

Validation (`validate_changepoints()` in `validation.py`) checks that values
are integer-only and in `[0, 1]`, and that `target_feature` references an
existing variable.

To compute changepoints programmatically, use `add_changepoints_to_ds()` from
`ethograph/utils/io.py`:

```python
from ethograph.utils.io import add_changepoints_to_ds
from ethograph.features.changepoitns import find_troughs_binary


ds = add_changepoints_to_ds(
    ds,
    target_feature="speed",
    changepoint_name="troughs",
    changepoint_func=find_troughs_binary,
)
# Creates ds["speed_troughs"] with type="changepoints", target_feature="speed"
```

`add_changepoints_to_ds()` uses `xr.apply_ufunc` with `vectorize=True`, so
your detection function only needs to handle a 1-D signal. It is applied
automatically across all other dimensions (individuals, keypoints, etc.).

---

## Audio changepoints (optional)

**NOTE**: Audio changepoints format is subject to change, still in development.

Audio changepoints use a different storage format because dense binary arrays
at audio sample rates (44 kHz) would be prohibitively large. Instead, they
are stored as onset/offset time pairs in seconds:

```python
ds["audio_cp_onsets"]  = xr.DataArray(onset_times_s,  dims=["audio_cp"],
                                       attrs={"type": "audio_changepoints",
                                              "target_feature": "audio"})
ds["audio_cp_offsets"] = xr.DataArray(offset_times_s, dims=["audio_cp"],
                                       attrs={"type": "audio_changepoints",
                                              "target_feature": "audio"})
```

These are typically computed by the GUI's audio changepoint detection
(VocalPy/VocalSeg).


---

## Custom dimensions (optional)

Any dimension that co-occurs with a time dimension in at least one feature
variable is automatically discovered by `find_temporal_dims()` in
`validation.py` and gets a selection [combo box](https://www.pythonguis.com/docs/qcombobox/) in the GUI. For example, a
`channels` dimension:

```python
ds["emg"] = xr.DataArray(
    emg_data,                            # shape: (time, channels)
    dims=["time", "channels"],
    coords={"channels": ["biceps", "triceps"]},
    attrs={"type": "features"},
)
```

If the dimension has coordinates (string or numeric), those labels appear in
the combo box. If it has no coordinates, integer indices (0, 1, 2, ...) are
shown instead.

Dimensions **do not need to match across features**. For example, `speed` may
have `(time, keypoints, individuals)` while `emg` has `(time, channels)`. The
GUI creates combo boxes for the union of all discovered dimensions. When a
feature doesn't have a selected dimension, that selection is silently ignored.
This is handled by `sel_valid()` in `ethograph/utils/xr_utils.py`, which
filters selection kwargs to only those dimensions present on the current
DataArray:

```python
import ethograph as eto

ds = xr.Dataset(
    data_vars={
        "speed": xr.DataArray(
            np.random.randn(1000, 4, 2),
            dims=["time", "keypoints", "individuals"],
            attrs={"type": "features"},
        ),
        "emg": xr.DataArray(
            np.random.randn(1000, 2, 2),
            dims=["time", "channels", "individuals"],
            attrs={"type": "features"},
        ),
    },
    coords={
        "time": np.arange(1000) / 30.0,
        "keypoints": ["nose", "left_ear", "right_ear", "tail"],
        "individuals": ["mouse1", "mouse2"],
        "channels": ["biceps", "triceps"],
    },
)

# The GUI creates combos for: keypoints, individuals, channels
# When "speed" is selected, the "channels" combo is ignored.
# When "emg" is selected, the "keypoints" combo is ignored.
# "individuals" applies to both since both features have that dimension.
# This is handled internally by sel_valid():
data, used_kwargs = eto.sel_valid(
    ds["emg"],
    {"keypoints": "nose", "channels": "biceps", "individuals": "mouse1"},
)
# "channels" and "individuals" are applied; "keypoints" is silently skipped.
```

---

## Trial condition attributes (Optional)

Any dataset attribute that is **not** in the set `{trial, pose, cameras, mics}`
and is **not** a common attribute across all trials (e.g. `fps`) is treated as
a **trial condition**. In the following example, you played tone A in the first
10 trials, and tone B in the second 10 trials. Adding trial conditions in the trial
attributes is helpful for two reasons. In 'Navigation/Help', you can filter to trials
that match a certain trial condition. And in [Export Labels](export-labels.md), the
exported `.tsv` file will include these trial conditions as metadata per label
in that trial.


```python
for trial_id in dt.trials[:10]:
    ds = dt.trial(trial_id)
    ds.attrs["stimulus"] = "tone_A"
for trial_id in dt.trials[10:20]:
    ds = dt.trial(trial_id)
    ds.attrs["stimulus"] = "tone_B"
```


---

## Full example: multi-trial dataset

```python
import numpy as np
import pandas as pd
import xarray as xr
import ethograph as eto

datasets = []
for trial_id in range(1, 11):
    n_time = 9000  # 5 minutes at 30 fps
    time_s = np.arange(n_time) / 30.0

    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                np.random.randn(n_time, 2, 4, 2),
                dims=["time", "space", "keypoints", "individuals"],
                attrs={"type": "features"},
            ),
            "speed": xr.DataArray(
                np.abs(np.random.randn(n_time, 4, 2)),
                dims=["time", "keypoints", "individuals"],
                attrs={"type": "features"},
            ),
        },
        coords={
            "time": time_s,
            "space": ["x", "y"],
            "keypoints": ["nose", "left_ear", "right_ear", "tail"],
            "individuals": ["mouse1", "mouse2"],
        },
    )
    ds.attrs["trial"] = trial_id
    ds.attrs["fps"] = 30.0
    ds.attrs["stimulus"] = "tone_A" if trial_id % 2 else "tone_B"

    datasets.append(ds)

# Session table with timing
session_table = pd.DataFrame({
    "trial": list(range(1, 11)),
    "start_time": [i * 300.0 for i in range(10)],
    "stop_time": [(i + 1) * 300.0 - 0.5 for i in range(10)],
})

dt = eto.from_datasets(datasets, session_table=session_table)

# Store media per stream
dt.set_media("video",
    [[f"cam1_trial{tid:03d}.mp4", f"cam2_trial{tid:03d}.mp4"] for tid in range(1, 11)],
    device_labels=["left", "right"],
)
dt.set_media("pose",
    [[f"dlc_cam1_trial{tid:03d}.h5", f"dlc_cam2_trial{tid:03d}.h5"] for tid in range(1, 11)],
    device_labels=["left", "right"],
)
# Ephys file is selected in the GUI, not stored here.
# If the ephys clock differs from the reference, set the offset:
dt.set_stream_offset("ephys", 0.0)

dt.save("trials.nc")
```

---

## Summary of `type` attribute values

| `attrs["type"]` | Purpose | Validation | Helper function |
|-----------------|---------|------------|-----------------|
| `"features"` | Shown in Feature dropdown, plotted as line/heatmap | Must be an array with a time dimension | `dataset_to_basic_trialtree()` auto-tags |
| `"colors"` | RGB overlay on plots | Must have `RGB` dim, values in `[0,1]` or `[0,255]` | `add_angle_rgb_to_ds()` |
| `"changepoints"` | Binary markers drawn on plots | Integer values in `{0, 1}`, needs `target_feature` | `add_changepoints_to_ds()` |
| `"audio_changepoints"` | Onset/offset pairs for audio events | Float seconds, paired `_onsets`/`_offsets` variables | TO BE ADDED |

---

## Key functions reference

| Function | Module | Purpose |
|----------|--------|---------|
| `TrialTree.from_datasets(datasets, session_table)` | `ethograph/utils/trialtree.py` | Build a TrialTree from a list of Datasets |
| `dt.set_media(stream, files, device_labels, per_trial)` | `ethograph/utils/trialtree.py` | Store media filenames for a stream |
| `dt.get_media(trial, stream, device)` | `ethograph/utils/trialtree.py` | Retrieve a media filename |
| `dt.devices(stream)` | `ethograph/utils/trialtree.py` | List device labels for a stream |
| `dt.set_session_table(table)` | `ethograph/utils/trialtree.py` | Set session timing table |
| `dt.start_time(trial)` | `ethograph/utils/trialtree.py` | Session-absolute start time of a trial |
| `dt.stop_time(trial)` | `ethograph/utils/trialtree.py` | Session-absolute stop time (or `None`) |
| `dt.trial_epoch(trial)` | `ethograph/utils/trialtree.py` | Pynapple `IntervalSet` for a trial |
| `dt.restrict(obj, trial)` | `ethograph/utils/trialtree.py` | Restrict a pynapple object to a trial window |
| `dt.source_start_time(trial, stream)` | `ethograph/utils/trialtree.py` | Trial-relative time of sample 0 for a stream |
| `dt.set_stream_offset(stream, offset)` | `ethograph/utils/trialtree.py` | Set session-absolute offset for a stream |
| `dataset_to_basic_trialtree(ds, video_path)` | `ethograph/utils/io.py` | Convert a single dataset into a TrialTree |
| `add_changepoints_to_ds(ds, ...)` | `ethograph/utils/io.py` | Compute and store changepoints for a feature |
| `add_angle_rgb_to_ds(ds, smoothing_params)` | `ethograph/utils/io.py` | Compute angle-based RGB colours from pose |
| `validate_dataset(ds, type_vars_dict)` | `ethograph/utils/validation.py` | Validate dataset structure and content |
| `validate_datatree(dt)` | `ethograph/utils/validation.py` | Validate an entire TrialTree |
| `sel_valid(da, sel_kwargs)` | `ethograph/utils/xr_utils.py` | Select data using only valid dims |
| `get_time_coord(da)` | `ethograph/utils/xr_utils.py` | Get whichever time coordinate a DataArray uses |