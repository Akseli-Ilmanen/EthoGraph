# TrialTree

`TrialTree` ([source code](https://github.com/Akseli-Ilmanen/EthoGraph/blob/main/ethograph/utils/trialtree.py)) is a thin wrapper around [xarray.DataTree](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#datatree) (`dt`) that makes it easier to work with multi-trial behavioural datasets. Each trial is stored as a child node containing an `xr.Dataset`, and the tree provides convenience methods for accessing, iterating, and modifying trials.

The dataset format builds on [Movement](https://movement.neuroinformatics.dev/latest/user_guide/movement_dataset.html) conventions for representing pose estimation and behavioural time series.

```
TrialTree (root)
├── "1"  →  xr.Dataset  (trial 1: features, coords, attrs)
├── "2"  →  xr.Dataset  (trial 2)
├── "3"  →  xr.Dataset  (trial 3)
└── ...
```

---

## Opening and creating

```python
import ethograph as eto

# Open an existing .nc file
dt = eto.open("path/to/trials.nc") # dt = datatree

# Access trial list
dt.trials  # [1, 2, 3, ...]
```

To create a TrialTree from scratch (single-trial, e.g. from a Movement dataset or numpy array):

```python
import ethograph as eto

ds = ...  # your xr.Dataset with a time dimension

# Wraps a single dataset into a TrialTree, adds empty label placeholders,
# and marks all non-label variables as features
dt = eto.dataset_to_basic_trialtree(ds)

# For multi-trial data, set attrs and build from a list of datasets
datasets = [ds1, ds2]
for i, ds in enumerate(datasets):
    ds.attrs["trial"] = i + 1
dt = eto.from_datasets(datasets)
```

To attach media file paths (video, audio, pose) after creating the tree:

```python
dt.set_media_files(
    video=[["camera1_trial001.mp4"]],
    audio=[["mic1_trial001.wav"]],
    pose=[["dlc_trial001.h5"]],
)
```

See the [tutorials](https://github.com/Akseli-Ilmanen/EthoGraph/tree/main/tutorials) for full worked examples of creating `.nc` files from different data sources.

---

## Media files

Use one of these three alignment modes.

### Mode 1: Per-trial files

Each trial has its own media files. This is the standard case when acquisition already exports trial-split files.

```python
import xarray as xr

session = xr.Dataset(
    coords={
        "trial": [1, 2, 3],
        "cameras": ["cam-1", "cam-2"],
    }
)

session["video"] = xr.DataArray(
    [
        ["trial001_cam-1.mp4", "trial001_cam-2.mp4"],
        ["trial002_cam-1.mp4", "trial002_cam-2.mp4"],
        ["trial003_cam-1.mp4", "trial003_cam-2.mp4"],
    ],
    dims=["trial", "cameras"],
    coords={"trial": [1, 2, 3], "cameras": ["cam-1", "cam-2"]},
)

session["pose"] = xr.DataArray(
    [
        ["trial001_cam-1.csv", "trial001_cam-2.csv"],
        ["trial002_cam-1.csv", "trial002_cam-2.csv"],
        ["trial003_cam-1.csv", "trial003_cam-2.csv"],
    ],
    dims=["trial", "cameras"],
    coords={"trial": [1, 2, 3], "cameras": ["cam-1", "cam-2"]},
)

dt["session"] = xr.DataTree(session)
```

### Mode 2: Session-long files per device

Use when each device has one long file covering many trials.
Example: one long video file per camera for the full session.

```python
import xarray as xr

session = xr.Dataset(
    coords={
        "video_file": [0],
        "cameras": ["cam-1", "2"],
        "trial": [1, 2, 3],
    }
)

# One file per camera for the full session
session["video"] = xr.DataArray(
    ["session_cam-1.mp4", "session_cam-2.mp4"],
    dims=["cameras"],
)

# Trial boundaries in session time
session["start_time"] = xr.DataArray([0.0, 30.0, 60.0], dims=["trial"])
session["stop_time"] = xr.DataArray([30.0, 60.0, 90.0], dims=["trial"])

dt["session"] = xr.DataTree(session)
```


### Mode 3: Real-world example (mixed trial/session alignment)

Use this when some modalities are trial-aligned and others are session-long:

- Video and pose are trial-aligned (one file per trial, per camera).
- Ephys is one session-long file.
- Audio files (from two microphones) are two session-long files.
- Audio has a constant offset relative to the reference timeline.

```python
import xarray as xr
import ethograph as eto

# Trial datasets are already loaded and include attrs["trial"]
dt = eto.from_datasets(ds_list)

# Trial table defines trial windows on the shared session clock
session = xr.Dataset(
    coords={
        "trial": [1, 2, 3],
        "cameras": ["cam-1", "cam-2"],
        "mics": ["mic-1", "mic-2"],
        "ephys_file": [0],
    }
)

session["start_time"] = xr.DataArray([0.0, 30.0, 60.0], dims=["trial"])
session["stop_time"] = xr.DataArray([20.0, 50.0, 90.0], dims=["trial"])

# Trial-aligned media
session["video"] = xr.DataArray(
    [
        ["trial001_cam-1.mp4", "trial001_cam-2.mp4"],
        ["trial002_cam-1.mp4", "trial002_cam-2.mp4"],
        ["trial003_cam-1.mp4", "trial003_cam-2.mp4"],
    ],
    dims=["trial", "cameras"],
)
session["pose"] = xr.DataArray(
    [
        ["trial001_cam-1.csv", "trial001_cam-2.csv"],
        ["trial002_cam-1.csv", "trial002_cam-2.csv"],
        ["trial003_cam-1.csv", "trial003_cam-2.csv"],
    ],
    dims=["trial", "cameras"],
)

# Session-long modalities (stored once)
session["audio"] = xr.DataArray(
    ["session_audio_ch1.wav", "session_audio_ch2.wav"],
    dims=["mics"],
)
session["ephys"] = xr.DataArray(["session_ephys.rhd"], dims=["ephys_file"])

dt["session"] = xr.DataTree(session)

# Constant stream offset: audio starts 230 ms after reference clock
dt.set_stream_offset("audio", 0.23)
```

This pattern is common in real experiments where behavior tracking is trial-based, while acquisition systems for ephys/audio run continuously across the session.

---

## Session table

The optional session table stores timing and alignment information at the tree level, separate from trial data. It lives as a child node named `"session"`.

```
TrialTree (root)
├── "session"  →  xr.Dataset
│   ├── start_time:  [10.0, 45.0]        (per-trial, trial dim)
│   ├── stop_time:   [25.0, 58.0]        (per-trial, trial dim)
│   ├── timestamps_ephys: [0.0, 0.001, ...]  (aligned timestamps, sample_ephys dim)
│   └── attrs:
│       ├── offset_audio: 0.2            (global stream offset)
│       ├── offset_video: 0.0
│       └── session_start_time, timestamps_reference
│
├── "1"  →  xr.Dataset (trial data, local time from 0)
├── "2"  →  xr.Dataset
└── ...
```

TODO add ntoes

## Stream alignment

TODO add notes.


---

## Accessing trials

Inspired by [xarray's](https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html#indexing) `.sel()` / `.isel()` pattern:

```python
# By trial number (label-based, like .sel)
ds = dt.trial(1) # Equivalent to dt[1].ds

# By position (index-based, like .isel)
ds = dt.itrial(0)   # first trial
```

Both return the trial's `xr.Dataset`, giving access to all variables, coordinates, and attributes:

```python
ds = dt.trial(1)
ds["speed"]                  # xr.DataArray
ds.attrs["fps"]              # 30.0
ds.coords["individuals"]     # ["bird1", "bird2"]

# Media files are stored at session level, not in trial attrs
dt.cameras                   # ["left", "right"]
dt.get_media(1, 'video', "left")      # "camera1_trial001.mp4"
```

---

## Iterating over trials

```python
# Iterate all trials
for trial_id, ds in dt.trial_items():
    print(f"Trial {trial_id}: {len(ds.time)} timepoints")

# Apply a function to every trial, returning a new TrialTree
dt_smoothed = dt.map_trials(lambda ds: smooth(ds))
```

---

## Modifying trials

There is an important distinction between in-place mutations and structural changes.

**In-place mutations** work directly through `.trial()` because the returned dataset shares its underlying data with the tree:

```python
# Modify an attribute
dt.trial(1).attrs["human_verified"] = True

# Modify existing array values
dt.trial(1)["speed"].values[:10] = 0.0
```

**Structural changes** (adding/removing variables, replacing a dataset) do not propagate back through `.trial()`. Use `update_trial()` instead:

```python
# Add a new variable to a trial
dt.update_trial(1, lambda ds: ds.assign(
    smoothed_speed=ds["speed"].rolling(time=5).mean()
))

# Replace a trial's dataset entirely
dt.update_trial(1, lambda _: new_dataset)
```

---

## Saving

```python
dt.save("path/to/trials.nc")

# Or save back to the original path
dt.save()
```

---

## Labels: `get_label_dt()`

Labels (segment annotations) are stored inside each trial dataset as interval variables (`onset_s`, `offset_s`, `labels`, `individual`) on a `segment` dimension. `get_label_dt()` extracts just the label data into a lightweight TrialTree, stripping all feature variables:

```python
label_dt = dt.get_label_dt()        # extract existing labels
empty_dt = dt.get_label_dt(empty=True)  # same structure, no label data
```

This is used internally by the GUI to maintain a separate label tree that can be saved independently (e.g. as `session_labels.nc`), keeping label files small and decoupled from the feature data. It is also used by the model training pipeline to store predictions alongside ground truth without duplicating feature arrays.

On save, labels are merged back into the main tree via `dt.overwrite_with_labels(label_dt)`.
