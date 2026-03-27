# TrialTree

`TrialTree` ([source code](https://github.com/Akseli-Ilmanen/EthoGraph/blob/main/ethograph/utils/trialtree.py)) is a thin wrapper around [xarray.DataTree](https://docs.xarray.dev/en/stable/user-guide/data-structures.html#datatree) that makes it easier to work with multi-trial behavioural datasets. Each trial is stored as a child node containing an `xr.Dataset`, and the tree provides convenience methods for accessing, iterating, and modifying trials.

The dataset format builds on [Movement](https://movement.neuroinformatics.dev/latest/user_guide/movement_dataset.html) conventions for representing pose estimation and behavioural time series. For the xarray Dataset structure expected inside each trial (dimensions, coordinates, `attrs["type"]`, etc.), see [Data Format](data_format.md).

```
TrialTree (root)
├── "session"  →  xr.Dataset  (timing, media filenames, stream offsets)
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
dt = eto.open("path/to/trials.nc")

# Access trial list
dt.trials  # [1, 2, 3, ...]
```

To create a TrialTree from scratch (single-trial, e.g. from a [Movement dataset](https://movement.neuroinformatics.dev/latest/user_guide/movement_dataset.html) or numpy array):

```python
ds = ...  # your xr.Dataset with a time dimension

# Wraps a single dataset into a TrialTree, adds empty label placeholders,
# and marks all non-label variables as features
dt = eto.dataset_to_basic_trialtree(ds)
```

For multi-trial data, set `attrs["trial"]` on each dataset and build from a list:

```python
datasets = [ds1, ds2]
for i, ds in enumerate(datasets):
    ds.attrs["trial"] = i + 1

dt = eto.from_datasets(datasets)
```

To include a session table with trial timing:

```python
import pandas as pd

session_table = pd.DataFrame({
    "trial": [1, 2],
    "start_time": [0.0, 120.5],
    "stop_time": [120.0, 245.0],
})

dt = eto.from_datasets(datasets, session_table=session_table)
```

See the [tutorials](https://github.com/Akseli-Ilmanen/EthoGraph/tree/main/tutorials) for full worked examples of creating `.nc` files from different data sources.

---

## Media files

Media filenames are stored in the session node via `dt.set_media()`. Call it once per stream. The available streams and their device dimensions are fixed:

| Stream | Default layout | Device dimension | Notes |
|--------|---------------|-----------------|-------|
| `"video"` | per-trial | `"camera"` | |
| `"audio"` | per-trial | `"microphone"` | |
| `"pose"` | per-trial | `"camera"` | |
| `"ephys"` | session-wide | *(none)* | Selected in GUI; no `set_media()` call needed. See [Ephys data](ephys-data.md). |

### Per-trial files

Each trial has its own media files. This is the default for video, audio, and pose.

```python
# Two cameras, three trials
dt.set_media("video",
    [["trial001_cam1.mp4", "trial001_cam2.mp4"],
     ["trial002_cam1.mp4", "trial002_cam2.mp4"],
     ["trial003_cam1.mp4", "trial003_cam2.mp4"]],
    device_labels=["left", "right"],
)

# Recommendation: pose files share the same camera labels
dt.set_media("pose",
    [["dlc_trial001_cam1.csv", "dlc_trial001_cam2.csv"],
     ["dlc_trial002_cam1.csv", "dlc_trial002_cam2.csv"],
     ["dlc_trial003_cam1.csv", "dlc_trial003_cam2.csv"]],
    device_labels=["left", "right"],
)

# Single microphone
dt.set_media("audio",
    [["mic_trial001.wav"],
     ["mic_trial002.wav"],
     ["mic_trial003.wav"]],
)
```

### Session-wide files

When a stream covers the entire session (one file, not split by trial), pass `per_trial=False`:

```python
# Single continuous video
dt.set_media("video", "full_session.mp4", per_trial=False)

# Session-wide with multiple cameras
dt.set_media("video",
    ["session_cam1.mp4", "session_cam2.mp4"],
    device_labels=["left", "right"],
    per_trial=False,
)
```

### Mixed alignment

A common real-world pattern: video and pose are per-trial, while audio runs continuously across the session. Ephys is always session-wide and is selected directly in the GUI — no `set_media("ephys", ...)` call needed. If the ephys clock differs from the reference, set a stream offset (see [Ephys data](ephys-data.md) and [Stream offsets](#stream-offsets)).

```python
dt = eto.from_datasets(ds_list, session_table=session_table)

# Per-trial video and pose
dt.set_media("video", video_filenames, device_labels=["cam-1", "cam-2"])
dt.set_media("pose", pose_filenames, device_labels=["cam-1", "cam-2"])

# Session-wide audio
dt.set_media("audio",
    ["session_ch1.wav", "session_ch2.wav"],
    device_labels=["mic-1", "mic-2"],
    per_trial=False,
)

# Audio starts 230 ms after the reference clock
dt.set_stream_offset("audio", 0.23)
# Ephys offset (if needed):
dt.set_stream_offset("ephys", 0.0)
```

### Querying media

```python
dt.get_media(1, "video", device="left")   # "trial001_cam1.mp4"
dt.devices("video")                        # ["left", "right"]
dt.cameras                                 # ["left", "right"]  (shortcut)
dt.mics                                    # ["mic-1", "mic-2"] (shortcut)
```

---

## Session table and timing

The session table is an `xr.Dataset` in the `"session"` child node. It holds trial timing (`start_time`, `stop_time`), media filenames, and stream offset attributes. Set it via `from_datasets(session_table=...)` or `dt.set_session_table()`.

```python
dt.session                    # the xr.Dataset, or None
dt.print_session()            # formatted summary
dt.session_to_dataframe()     # as a pandas DataFrame
```

### Trial timing

When `start_time` and `stop_time` are present, the timing API is available directly on the tree:

```python
dt.start_time(1)              # 0.0
dt.stop_time(1)               # 120.0 (or None if not known)
dt.trial_duration(1)          # 120.0
```

### Pynapple integration

Trial epochs are exposed as [pynapple](https://pynapple.org/) `IntervalSet` objects for restricting neural data:

```python
epoch = dt.trial_epoch(1)     # pynapple IntervalSet
spikes_t1 = dt.restrict(spikes, 1)  # restrict a TsGroup/Tsd to trial 1
```

### Stream offsets

For session-wide streams, `set_stream_offset()` specifies when sample 0 of the file occurs in session-absolute time. `source_start_time()` then computes the trial-relative offset:

```python
dt.set_stream_offset("ephys", 0.0)

# Per-trial streams always return 0
dt.source_start_time(1, "video")   # 0.0

# Session-wide: offset - trial_start
dt.source_start_time(2, "ephys")   # e.g. -120.5
```

---

## Accessing trials

Inspired by [xarray's](https://docs.xarray.dev/en/stable/getting-started-guide/quick-overview.html#indexing) `.sel()` / `.isel()` pattern:

```python
# By trial number (label-based, like .sel)
ds = dt.trial(1)

# By position (index-based, like .isel)
ds = dt.itrial(0)   # first trial
```

Both return the trial's `xr.Dataset`, giving access to all variables, coordinates, and attributes:

```python
ds = dt.trial(1)
ds["speed"]                  # xr.DataArray
ds.attrs["fps"]              # 30.0
ds.coords["individuals"]     # ["bird1", "bird2"]
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

## Filtering

```python
# Filter to trials matching a condition
dt_tone_a = dt.filter_by_attr("stimulus", "tone_A")
dt_tone_a.trials  # subset of trials where stimulus == "tone_A"
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
label_dt = dt.get_label_dt()            # extract existing labels
empty_dt = dt.get_label_dt(empty=True)  # same structure, no label data
```

This is used internally by the GUI to maintain a separate label tree that can be saved independently (e.g. as `session_labels.nc`), keeping label files small and decoupled from the feature data. It is also used by the model training pipeline to store predictions alongside ground truth without duplicating feature arrays.

On save, labels are merged back into the main tree via `dt.overwrite_with_labels(label_dt)`.