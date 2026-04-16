(target-multi-trial)=
# Multi-trial setup

You need a short Python script when:

- You have **multiple trials** — separate video/audio/pose files per trial
- You recorded from **multiple cameras**
- You have **multiple microphone files** (one `.wav` per mic)

The {doc}`wizard` only handles single files. Everything else uses
{func}`eto.from_datasets() <ethograph.from_datasets>` (xarray) or
{func}`eto.load_nap_data() <ethograph.load_nap_data>` (pynapple/NWB), plus an
**alignment file** for media references and trial timing.

---

## How alignment works

Media filenames, trial timing, and stream offsets are stored in an NWB
alignment file at `.ethograph/alignment.nwb` — not inside the data file itself.
This keeps data portable (filenames are stored as basename only) and lets you
change media paths without re-exporting features.

Two functions create alignment files:

| Function | When to use |
|----------|-------------|
| {func}`~ethograph.io.nwb_alignment.align_media_per_trial` | Media files map 1:1 to trials |
| {func}`~ethograph.io.nwb_alignment.align_media_from_streams` | Session-wide files, mixed per-trial + continuous, or explicit timestamps |

**Column naming convention**: `{stream}_{device}` — e.g. `video_cam-1`,
`audio_mic-1`, `pose_cam-1`.

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

Camera index determines which pose file is overlaid: device `cam-1` maps to
`pose_cam-1`, etc.

For NWB: multi-camera NWB files store each camera as a separate
{class}`~pynwb.image.ImageSeries` in `nwb.acquisition` (e.g. `video_cam-1`,
`video_cam-2`). EthoGraph discovers cameras automatically from acquisition
items.

---

## Per-trial audio

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

For session-wide audio (one continuous file spanning all trials), see the
{ref}`Session-wide streams <target-session-wide-streams>` section below.

---

## Trial timing

Trial start/stop times are part of the alignment file's trials table. When
present, they enable session-mode navigation and let the GUI restrict neural
data to trial windows. See {doc}`../user_guide/nwb_alignment` for the full
alignment model.

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

If `start_time` / `stop_time` are omitted, durations are inferred from the
media files.

---

## Trial conditions

Add metadata to each trial for filtering in the Navigation widget and for
export:

::::{tab-set}

:::{tab-item} Xarray attrs

```python
ds.attrs["stimulus"] = "tone_A"
```
:::

:::{tab-item} Alignment column

Add extra columns to the trial table before creating the alignment:

```python
trial_table["stimulus"] = ["tone_A", "tone_B", "tone_A", "tone_B", "tone_A"]
align_media_per_trial(trial_table, ...)
```
:::

::::

---

## Full worked example

::::{tab-set}

:::{tab-item} Xarray

```python
import numpy as np
import pandas as pd
import xarray as xr
import ethograph as eto
from ethograph.io.nwb_alignment import align_media_per_trial

datasets = []
for trial_id in range(1, 11):
    n_time = 9000  # 5 minutes at 30 fps
    time_s = np.arange(n_time) / 30.0

    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                np.random.randn(n_time, 2, 4, 2),
                dims=["time", "space", "keypoints", "individuals"],
            ),
            "speed": xr.DataArray(
                np.abs(np.random.randn(n_time, 4, 2)),
                dims=["time", "keypoints", "individuals"],
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

dt = eto.from_datasets(datasets)
dt.save("session.nc")

# Alignment: media files + trial timing
trial_table = pd.DataFrame({
    "trial": list(range(1, 11)),
    "start_time": [i * 300.0 for i in range(10)],
    "stop_time": [(i + 1) * 300.0 - 0.5 for i in range(10)],
    "video_cam-1": [f"cam1_trial{tid:03d}.mp4" for tid in range(1, 11)],
    "video_cam-2": [f"cam2_trial{tid:03d}.mp4" for tid in range(1, 11)],
    "pose_cam-1":  [f"dlc_cam1_trial{tid:03d}.h5" for tid in range(1, 11)],
    "pose_cam-2":  [f"dlc_cam2_trial{tid:03d}.h5" for tid in range(1, 11)],
})

align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0, "pose": 30.0},
    output_path=".ethograph/alignment.nwb",
)
```
:::

:::{tab-item} NWB

```python
# NWB files already store trials, media references, and features together.
# For DANDI datasets:
#   1. Select the .nwb URL or downloaded file in the GUI
#   2. Click Load — trials, features, and media are read automatically
#
# For custom NWB files built with pynwb:
import pynwb
from datetime import datetime
from dateutil.tz import tzlocal

nwbfile = pynwb.NWBFile(
    session_description="My experiment",
    identifier="session-001",
    session_start_time=datetime.now(tzlocal()),
)

nwbfile.add_trial_column(name="stimulus", description="Stimulus type")
for i in range(10):
    nwbfile.add_trial(
        start_time=i * 300.0,
        stop_time=(i + 1) * 300.0 - 0.5,
        stimulus="tone_A" if i % 2 else "tone_B",
    )

from pynwb.behavior import BehavioralTimeSeries
behavior_mod = nwbfile.create_processing_module("behavior", "Behavioral data")
behavior_ts = BehavioralTimeSeries(name="BehavioralTimeSeries")
behavior_ts.create_timeseries(name="speed", data=speed_array, rate=30.0, unit="cm/s")
behavior_mod.add(behavior_ts)

with pynwb.NWBHDF5IO("session.nwb", "w") as io:
    io.write(nwbfile)
```
:::

::::

---

(target-session-wide-streams)=
## Session-wide streams

For setups where media files don't map 1:1 to trials — e.g. one continuous
audio file covering all trials, ephys recorded on a separate clock, or
explicit timestamps from a DAQ — use
{func}`~ethograph.io.nwb_alignment.align_media_from_streams`.

### Session-wide audio

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

### Ephys with multiple trials

Ephys is session-wide — select the file in the GUI rather than embedding it in
the dataset. If the ephys clock differs from the behavioural reference,
include it as a stream with a `starting_time` offset:

```python
streams = [
    {"name": "video_cam-1", "files": ["t1.mp4", "t2.mp4"], "rate": 30.0},
    {"name": "ephys_probe-1", "files": ["session.dat"], "rate": 30000.0, "starting_time": 0.5},
]

align_media_from_streams(trials_df, streams, ".ethograph/alignment.nwb")
```

See {doc}`loading_ephys` for supported file formats, Kilosort folder setup,
and channel mapping.

### Stream specification

Each entry in `streams` accepts:

| Key | Type | Description |
|-----|------|-------------|
| `name` | `str` | Stream identifier: `{stream}_{device}` (e.g. `video_cam-1`) |
| `files` | `list[str]` | One file (session-wide) or one per trial |
| `rate` | `float` | Sampling rate in Hz |
| `starting_time` | `float` | When the stream begins in session time. Default 0.0 |
| `timestamps` | `ndarray` | Explicit per-sample timestamps. Overrides `rate` when clocks drift |

---

## References

- {doc}`../api/trialtree` — `from_datasets()`, `from_continuous()`, timing, iteration
- {doc}`data_requirements` — {class}`xarray.Dataset` structure, pynapple objects, NWB conventions
- {func}`~ethograph.io.nwb_alignment.align_media_per_trial` — per-trial alignment
- {func}`~ethograph.io.nwb_alignment.align_media_from_streams` — session-wide + mixed alignment
