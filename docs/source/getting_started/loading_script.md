(target-loading-script)=
# Multi-trial setup (Python script)

You need a short script when:

- You have **multiple trials** — separate video/audio/pose files per trial
- You recorded from **multiple cameras**
- You have **multiple separate microphone files** (one `.wav` per mic)

The Create dialog handles only single files. Everything else uses
{func}`eto.from_datasets() <ethograph.from_datasets>` (xarray) or
{func}`eto.load_nap_data() <ethograph.load_nap_data>` (pynapple/NWB),
plus an **alignment file** for media references and trial timing.

---

## How alignment works

Media filenames, trial timing, and stream offsets are stored in an NWB
alignment file at `.ethograph/alignment.nwb` — not inside the data file itself.
This keeps data portable (filenames are stored as basename only) and lets you
change media paths without re-exporting features.

Two functions create alignment files:

| Function | When to use |
|----------|-------------|
| {func}`~ethograph.io.nwb_alignment.align_media_per_trial` | Media files map 1:1 to trials (most common) |
| {func}`~ethograph.io.nwb_alignment.align_media_from_streams` | Session-wide files, mixed per-trial + continuous, or explicit timestamps |

**Column naming convention**: `{stream}_{device}` — e.g. `video_cam-1`, `audio_mic-1`, `pose_cam-1`.

---

## Minimal example

::::{tab-set}

:::{tab-item} Xarray

```python
import numpy as np
import pandas as pd
import xarray as xr
import ethograph as eto
from ethograph.io.nwb_alignment import align_media_per_trial

# 1) Build one xr.Dataset per trial
datasets = []
for trial_id in range(1, 6):
    n_time = 9000
    ds = xr.Dataset(
        {"speed": xr.DataArray(
            np.random.randn(n_time),
            dims=["time"],
            coords={"time": np.arange(n_time) / 30.0},
        )},
    )
    ds.attrs["trial"] = trial_id
    ds.attrs["fps"] = 30.0
    datasets.append(ds)

dt = eto.from_datasets(datasets)
dt.save("session.nc")

# 2) Create alignment file for media
trial_table = pd.DataFrame({
    "trial": list(range(1, 6)),
    "video_cam-1": [f"trial{i:03d}.mp4" for i in range(1, 6)],
})

align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0},
    output_path=".ethograph/alignment.nwb",
)
```
:::

:::{tab-item} Pynapple

```python
import numpy as np
import pynapple as nap
import pandas as pd
from ethograph.io.nwb_alignment import align_media_per_trial

# 1) Save feature data as pynapple
speed = nap.Tsd(
    t=np.arange(45000) / 30.0,  # 5 trials × 5 min at 30 fps
    d=np.random.randn(45000),
)
trials = nap.IntervalSet(
    start=[i * 300.0 for i in range(5)],
    end=[(i + 1) * 300.0 - 0.5 for i in range(5)],
)
nap.save_file({"speed": speed, "trials": trials}, "session")

# 2) Create alignment file for media
trial_table = pd.DataFrame({
    "trial": list(range(1, 6)),
    "start_time": [i * 300.0 for i in range(5)],
    "stop_time": [(i + 1) * 300.0 - 0.5 for i in range(5)],
    "video_cam-1": [f"trial{i:03d}.mp4" for i in range(1, 6)],
})

align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0},
    output_path=".ethograph/alignment.nwb",
)
```
:::

:::{tab-item} NWB

```python
# NWB files already store trials and media references natively.
# If your NWB file was created with pynwb or NeuroConv, media
# paths are stored as ImageSeries in nwb.acquisition.
# No separate alignment step is needed — just load the .nwb in the GUI.
#
# For NWB files from DANDI that lack local media paths, the GUI
# creates .ethograph/alignment.nwb automatically on first load.
```
:::

::::

---

## Multiple cameras

::::{tab-set}

:::{tab-item} Xarray / Pynapple

```python
trial_table = pd.DataFrame({
    "trial": [1, 2],
    "video_cam-1": ["cam1_trial001.mp4", "cam1_trial002.mp4"],
    "video_cam-2": ["cam2_trial001.mp4", "cam2_trial002.mp4"],
    "pose_cam-1":  ["dlc_cam1_trial001.h5", "dlc_cam1_trial002.h5"],
    "pose_cam-2":  ["dlc_cam2_trial001.h5", "dlc_cam2_trial002.h5"],
})

align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0, "pose": 30.0},
    output_path=".ethograph/alignment.nwb",
)
```

Camera index determines which pose file is overlaid: device `cam-1` maps to `pose_cam-1`, etc.
:::

:::{tab-item} NWB

```python
# Multi-camera NWB files store each camera as a separate ImageSeries
# in nwb.acquisition (e.g. "video_cam-1", "video_cam-2").
# EthoGraph discovers cameras automatically from acquisition items.
```
:::

::::

---

## Session-wide audio

One continuous audio file covering all trials (with optional per-mic split):

::::{tab-set}

:::{tab-item} Using align_media_from_streams

Use {func}`~ethograph.io.nwb_alignment.align_media_from_streams` when mixing
per-trial video with session-wide audio:

```python
import pandas as pd
from ethograph.io.nwb_alignment import align_media_from_streams

trials = pd.DataFrame({
    "trial": [1, 2, 3],
    "start_time": [0.0, 300.0, 600.0],
    "stop_time": [299.5, 599.5, 899.5],
})

streams = [
    {"name": "video_cam-1", "files": ["t1.mp4", "t2.mp4", "t3.mp4"], "rate": 30.0},
    # Session-wide: one file, starting_time marks when it begins in session time
    {"name": "audio_mic-1", "files": ["session_ch1.wav"], "rate": 48000.0, "starting_time": 0.0},
    {"name": "audio_mic-2", "files": ["session_ch2.wav"], "rate": 48000.0, "starting_time": 0.0},
]

align_media_from_streams(trials, streams, ".ethograph/alignment.nwb")
```
:::

:::{tab-item} Per-trial audio

If audio is split per-trial, use `align_media_per_trial` with audio columns:

```python
trial_table = pd.DataFrame({
    "trial": [1, 2],
    "video_cam-1":  ["cam1_t1.mp4", "cam1_t2.mp4"],
    "audio_mic-1":  ["mic1_trial001.wav", "mic1_trial002.wav"],
    "audio_mic-2":  ["mic2_trial001.wav", "mic2_trial002.wav"],
})

align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0, "audio": 48000.0},
    output_path=".ethograph/alignment.nwb",
)
```
:::

::::

---

## Ephys with multiple trials

Ephys is session-wide — select the file in the GUI rather than embedding it in the dataset.

If the ephys clock differs from the behavioural reference, include it as a stream
with a `starting_time` offset:

```python
from ethograph.io.nwb_alignment import align_media_from_streams

streams = [
    {"name": "video_cam-1", "files": ["t1.mp4", "t2.mp4"], "rate": 30.0},
    {"name": "ephys_probe-1", "files": ["session.dat"], "rate": 30000.0, "starting_time": 0.0},
]

align_media_from_streams(trials_df, streams, ".ethograph/alignment.nwb")
```

See {doc}`loading_ephys` and the {doc}`../api/trialtree` for the full offset API.

---

## Session table and trial timing

Trial start/stop times can be included in the alignment table. This enables
session-mode navigation and restricting neural data to trial windows.

```python
trial_table = pd.DataFrame({
    "trial": list(range(1, 6)),
    "start_time": [i * 300.0 for i in range(5)],
    "stop_time":  [(i + 1) * 300.0 - 0.5 for i in range(5)],
    "video_cam-1": [f"trial{i:03d}.mp4" for i in range(1, 6)],
})

align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0},
    output_path=".ethograph/alignment.nwb",
)
```

If `start_time` / `stop_time` are omitted, durations are inferred from the media files.

---

## Trial conditions

Add metadata to each trial for filtering in the Navigation widget and for export:

::::{tab-set}

:::{tab-item} Xarray

```python
ds.attrs["stimulus"] = "tone_A"
```

See {ref}`Data requirements — Trial conditions <target-data-requirements>`.
:::

:::{tab-item} NWB alignment

Add extra columns to the trial table before creating the alignment:

```python
trial_table["stimulus"] = ["tone_A", "tone_B", "tone_A", "tone_B", "tone_A"]
align_media_per_trial(trial_table, ...)
```
:::

::::

---

## Full worked example

For a complete multi-trial example including all streams, see {ref}`Data requirements — Full example <target-data-requirements>`.

---

## Other / unsupported formats

If your data format is not covered by the options above:

### Option A — Convert to `.npy`

Save your data as a `.npy` file (shape `(n_samples,)` or `(n_samples, n_variables)`) and use the **4) Generate from npy file** dialog.

- Column names are assigned in the dialog.
- **High sampling rate?** Enable the **Downsample** checkbox in the I/O widget (e.g. factor 100 keeps 1 in 100 samples). In a script: {func}`eto.downsample_trialtree(dt, factor) <ethograph.downsample_trialtree>`.

See {doc}`loading_numpy` for full steps.

### Option B — High sampling-rate periodic data -> `.wav`

For signals you want to visualise quickly (e.g. 1 kHz pressure sensor, EMG), convert to `.wav` with [audioio](https://github.com/bendalab/audioio):

```python
import audioio
audioio.write_audio("signal.wav", data, sample_rate)
```

Load via **3) Generate from audio file**. Audio is displayed with min/max downsampling — the waveform and spectrogram render fast at any zoom level, no manual downsample step needed.

See {doc}`loading_audio` for full steps.

### Option C — Multi-dimensional data -> xarray script

For arrays with 3 or more dimensions (e.g. `time x individuals x keypoints x space`), create an {class}`xarray.Dataset` and build a {class}`~ethograph.io.trialtree.TrialTree`:

```python
import xarray as xr
import ethograph as eto

da = xr.DataArray(
    data,  # e.g. shape: (n_time, n_individuals, n_keypoints, 3)
    dims=["time", "individuals", "keypoints", "space"],
    coords={"time": time_vec},
)
ds = xr.Dataset({"position": da})
ds.attrs["fps"] = sample_rate
ds.attrs["trial"] = 1

dt = eto.from_datasets([ds])
dt.save("data.nc")
```

**High sampling rate?** Enable the **Downsample** checkbox in the I/O widget, or call {func}`~ethograph.downsample_trialtree` in your script before {meth}`~ethograph.io.trialtree.TrialTree.save`.

See the {doc}`../api/trialtree` and {doc}`data_requirements` for the full xarray format.

---

## References

- {doc}`../api/trialtree` — `from_datasets()`, `from_continuous()`, timing, iteration
- {doc}`data_requirements` — {class}`xarray.Dataset` structure, pynapple objects, NWB conventions
- {func}`~ethograph.io.nwb_alignment.align_media_per_trial` — per-trial alignment
- {func}`~ethograph.io.nwb_alignment.align_media_from_streams` — session-wide + mixed alignment
