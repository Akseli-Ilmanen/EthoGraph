(target-data-requirements)=
# Data Format Requirements

EthoGraph supports three data backends. Pick the one that matches your workflow:

| Backend | Best for | Core object |
|---------|----------|-------------|
| **xarray** | Custom datasets, pose estimation, multi-dim arrays | {class}`xarray.Dataset` / {class}`~ethograph.io.trialtree.TrialTree` |
| **Pynapple** | Neuroscience time-series, NWB interop | {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` / {class}`~pynapple.TsGroup` |
| **NWB** | Standardised neurodata, DANDI archives | `.nwb` file (loaded via pynapple) |

The sections below show requirements and examples for each backend using tabs.

---

## Minimal working example

::::{tab-set}

:::{tab-item} Xarray
```python
import numpy as np
import xarray as xr
import ethograph as eto

ds = xr.Dataset(
    data_vars={
        "speed": xr.DataArray(
            np.random.randn(9000),
            dims=["time"],
            coords={"time": np.arange(9000) / 30.0},
        ),
    },
    coords={"individuals": ["mouse1"]},
)
ds.attrs["trial"] = 1
ds.attrs["fps"] = 30.0

dt = eto.from_datasets([ds])
dt.save("session.nc")
```
:::

:::{tab-item} Pynapple
```python
import numpy as np
import pynapple as nap
import ethograph as eto

speed = nap.Tsd(
    t=np.arange(9000) / 30.0,
    d=np.random.randn(9000),
)
nap.save_file({"speed": speed}, "session")
# Load in GUI: select the session.npz file
```
:::

:::{tab-item} NWB
```python
# NWB files from DANDI or NeuroConv are loaded directly — no conversion needed.
# In the GUI: select the .nwb file in the I/O widget and click Load.
#
# To create an NWB file programmatically, see the pynwb documentation:
# https://pynwb.readthedocs.io/en/stable/tutorials/general/plot_file.html
```
:::

::::

---

## Required attributes

::::{tab-set}

:::{tab-item} Xarray

Every trial's {class}`xarray.Dataset` **must** have:

| Attribute | Type | Description |
|-----------|------|-------------|
| `attrs["trial"]` | `int`, `str` | Trial identifier (1, 2, 3, ...). Must be unique across trials. |
| `attrs["fps"]` | `float` | Frame rate of the primary video. Not required for audio-only datasets. |

**Recommended:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `coords["individuals"]` | `str` array | Names of tracked subjects (e.g. `["mouse1", "mouse2"]`). |

```python
ds = xr.Dataset(
    coords={"individuals": ["bird1", "bird2"]},
)
ds.attrs["trial"] = 1
ds.attrs["fps"] = 30.0
```

:::

:::{tab-item} Pynapple

Pynapple objects carry timestamps natively — no `fps` or `trial` attribute is needed.

- **Timestamps**: Every {class}`~pynapple.Tsd` / {class}`~pynapple.TsdFrame` / {class}`~pynapple.TsdTensor` stores its own time axis.
- **Trials**: Defined by an {class}`~pynapple.IntervalSet` (either from NWB trials or created manually).

```python
import pynapple as nap
import numpy as np

# Trials as IntervalSet
trials = nap.IntervalSet(
    start=[0.0, 300.0, 600.0],
    end=[299.5, 599.5, 899.5],
)

# Feature data
speed = nap.TsdFrame(
    t=np.arange(27000) / 30.0,
    d=np.random.randn(27000, 2),
    columns=["mouse1", "mouse2"],
)
```
:::

:::{tab-item} NWB

NWB files follow the [NWB standard](https://www.nwb.org/). EthoGraph reads:

- **Trials**: `nwb.trials` table (columns: `start_time`, `stop_time`, plus custom columns).
- **Behavioural data**: {class}`~pynwb.TimeSeries` in `nwb.processing` modules.
- **Electrophysiology**: {class}`~pynwb.ecephys.ElectricalSeries` in `nwb.acquisition`.
- **Pose estimation**: `PoseEstimation` containers (ndx-pose extension).

If `nwb.trials` is absent, the entire recording is treated as a single trial.

EthoGraph uses {func}`~ethograph.io.nwb_import.read_trials_table` to extract trial boundaries
and {func}`~ethograph.io.nwb_import.probe_behavioral_series` to discover available time-series.
:::

::::

---

## Features (plottable variables)

::::{tab-set}

:::{tab-item} Xarray

Any `data_var` with at least one dimension whose name contains `"time"` appears
in the GUI's **Feature** dropdown:

```python
ds["speed"] = xr.DataArray(
    speed_values,
    dims=["time", "keypoints", "individuals"],
)
```

Different features can use different time coordinates with different sampling
rates (e.g. `time`, `time_accelerometer`, `time_video`).
:::

:::{tab-item} Pynapple

All {class}`~pynapple.Tsd`, {class}`~pynapple.TsdFrame`, and {class}`~pynapple.TsdTensor` objects in the loaded data dict are
automatically detected as features. Column names become selectable dimensions.

```python
# Single variable
speed = nap.Tsd(t=time_s, d=speed_values)

# Multi-column variable (e.g. x, y, z)
position = nap.TsdFrame(
    t=time_s,
    d=pos_array,                        # shape: (n_time, 3)
    columns=["x", "y", "z"],
)
```

Different features can have different sampling rates — pynapple handles
time alignment natively.
:::

:::{tab-item} NWB

EthoGraph discovers features automatically:

- **Behavioural series**: {func}`~ethograph.io.nwb_import.probe_behavioral_series` lists all
  {class}`~pynwb.TimeSeries` in processing modules (excluding `ecephys`, `ophys`, `ogen`).

NWB data is loaded via pynapple, so all NWB {class}`~pynwb.TimeSeries` become pynapple objects
internally.
:::

::::

---

## Specifying individuals

::::{tab-set}

:::{tab-item} Xarray

Individuals are stored as a **coordinate**, not an attribute. With multi-animal data, this allows the GUI to store separate labels and feature data for different individuals.

```python
ds = xr.Dataset(
    data_vars={
        "speed": xr.DataArray(
            speed_array,                # shape: (time, individuals)
            dims=["time", "individuals"],
        ),
    },
    coords={
        "time": time_values,
        "individuals": ["mouse1", "mouse2", "mouse3"],
    },
)
```

When labelling, the selected individual filters which labels are shown and created.
:::

:::{tab-item} Pynapple

Pynapple does not have a built-in concept of "individuals". Multi-subject
data is typically stored as separate objects in the data dict:

```python
data = {
    "speed_mouse1": nap.Tsd(t=time_s, d=speed_mouse1),
    "speed_mouse2": nap.Tsd(t=time_s, d=speed_mouse2),
}
```

Each object appears as a separate feature in the GUI. Individual selection
is not available for pynapple backends.
:::

:::{tab-item} NWB

NWB files represent a single subject per file — {attr}`~pynwb.file.NWBFile.subject` is a
singular {class}`~pynwb.file.Subject` object
([PyNWB docs](https://pynwb.readthedocs.io/en/stable/pynwb.file.html)).
Multi-subject experiments use separate `.nwb` files per subject.
Individual selection is not available when loading a single NWB file.
:::

::::

---

## Media files and alignment

Media filenames (video, audio, pose), trial timing, and stream offsets are stored
in an **NWB alignment file** (`.ethograph/alignment.nwb`), not inside individual
trial datasets. This keeps data and metadata separate, and filenames are stored
as **filename-only strings** so datasets remain portable.

::::{tab-set}

:::{tab-item} Xarray

After building your TrialTree, create an alignment file using
{func}`~ethograph.io.nwb_alignment.align_media_per_trial`:

```python
import pandas as pd
from ethograph.io.nwb_alignment import align_media_per_trial

# Trial table: one row per trial, columns named {stream}_{device}
trial_table = pd.DataFrame({
    "trial": [1, 2, 3],
    "video_cam-1": ["cam1_t1.mp4", "cam1_t2.mp4", "cam1_t3.mp4"],
    "pose_cam-1":  ["dlc_t1.h5",   "dlc_t2.h5",   "dlc_t3.h5"],
    "audio_mic-1": ["mic1_t1.wav", "mic1_t2.wav", "mic1_t3.wav"],
})

align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0, "pose": 30.0, "audio": 48000.0},
    output_path="my_project/.ethograph/alignment.nwb",
)
```

**Column naming convention**: `{stream}_{device}` — e.g. `video_cam-1`, `audio_mic-1`, `pose_cam-1`.

See {doc}`loading_script` for full multi-trial examples.
:::

:::{tab-item} Pynapple

When loading pynapple data (`.npz` or folder), the GUI creates the alignment
file automatically if media files are present in the project directory. You can
also create one manually:

```python
import pandas as pd
from ethograph.io.nwb_alignment import align_media_per_trial

trial_table = pd.DataFrame({
    "trial": [1],
    "video_cam-1": ["recording.mp4"],
    "audio_mic-1": ["recording.wav"],
})

align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0, "audio": 48000.0},
    output_path="my_project/.ethograph/alignment.nwb",
)
```
:::

:::{tab-item} NWB

NWB files already contain session metadata (trials table, {class}`~pynwb.image.ImageSeries` for
media). EthoGraph reads this directly — no separate alignment file is needed
for standalone `.nwb` files.

For NWB files loaded as part of a project directory (e.g. from DANDI), the GUI
creates `.ethograph/alignment.nwb` automatically from the source NWB metadata.

{class}`~ethograph.io.nwb_alignment.NWBAlignment` provides access to session metadata:

```python
from ethograph.io.nwb_alignment import NWBAlignment

alignment = NWBAlignment("my_project/.ethograph/alignment.nwb")
print(alignment.trials_df)
print(alignment.cameras)
print(alignment.start_time(trial=1))
alignment.close()
```
:::

::::

---

## Scenario-specific requirements

### Video + Pose

::::{tab-set}

:::{tab-item} Xarray

```python
import numpy as np
import xarray as xr
import ethograph as eto
from ethograph.io.nwb_alignment import align_media_per_trial

n_time = 9000
ds = xr.Dataset(
    data_vars={
        "position": xr.DataArray(
            np.random.randn(n_time, 2, 4, 1),
            dims=["time", "space", "keypoints", "individuals"],
        ),
        "speed": xr.DataArray(
            np.abs(np.random.randn(n_time, 4, 1)),
            dims=["time", "keypoints", "individuals"],
        ),
    },
    coords={
        "time": np.arange(n_time) / 30.0,
        "space": ["x", "y"],
        "keypoints": ["nose", "left_ear", "right_ear", "tail"],
        "individuals": ["mouse1"],
    },
)
ds.attrs["trial"] = 1
ds.attrs["fps"] = 30.0

dt = eto.from_datasets([ds])
dt.save("session.nc")

# Alignment: link media files
import pandas as pd
align_media_per_trial(
    pd.DataFrame({
        "trial": [1],
        "video_cam-1": ["trial001.mp4"],
        "pose_cam-1": ["trial001.h5"],
    }),
    stream_rates={"video": 30.0, "pose": 30.0},
    output_path=".ethograph/alignment.nwb",
)
```
:::

:::{tab-item} Pynapple
```python
import numpy as np
import pynapple as nap

position = nap.TsdFrame(
    t=np.arange(9000) / 30.0,
    d=np.random.randn(9000, 2),
    columns=["x", "y"],
)
speed = nap.Tsd(
    t=np.arange(9000) / 30.0,
    d=np.abs(np.random.randn(9000)),
)
nap.save_file({"position": position, "speed": speed}, "session")
# Load session.npz in the GUI, select the video/pose folders
```
:::

:::{tab-item} NWB
```python
# NWB files with PoseEstimation containers are supported directly.
# EthoGraph reads PoseEstimationSeries from processing modules.
# In the GUI: select the .nwb file → Load.
# Pose overlays appear automatically if PoseEstimation data is present.
```
:::

::::

### Audio only (no video)

::::{tab-set}

:::{tab-item} Xarray

When cameras are absent, the GUI enters **no-video mode**: a time slider replaces
the video player, and playback uses `sounddevice`. `fps` is not required.

```python
ds = xr.Dataset(coords={"individuals": ["bird1"]})
ds.attrs["trial"] = 1

dt = eto.from_datasets([ds])
dt.save("session.nc")

# Alignment
import pandas as pd
from ethograph.io.nwb_alignment import align_media_per_trial
align_media_per_trial(
    pd.DataFrame({"trial": [1], "audio_mic-1": ["song.wav"]}),
    stream_rates={"audio": 44100.0},
    output_path=".ethograph/alignment.nwb",
)
```
:::

:::{tab-item} Pynapple
```python
import numpy as np
import pynapple as nap

# Audio-only: just save your feature data
amplitude = nap.Tsd(t=np.arange(44100 * 60) / 44100.0, d=np.random.randn(44100 * 60))
nap.save_file({"amplitude": amplitude}, "session")
# Load session.npz in the GUI, select the audio folder
```
:::

:::{tab-item} NWB
```python
# NWB files with audio TimeSeries are loaded directly.
# If no video ImageSeries is present, the GUI enters no-video mode.
```
:::

::::

### Ephys with video/audio alignment

Ephys is a session-wide stream. The raw recording file is selected in the GUI — you
do not embed it in a dataset. If the ephys clock differs from the behavioural
reference, record the offset in the alignment file:

```python
from ethograph.io.nwb_alignment import align_media_from_streams

streams = [
    {"name": "video_cam-1", "files": ["t1.mp4", "t2.mp4"], "rate": 30.0},
    {"name": "ephys_probe-1", "files": ["session.dat"], "rate": 30000.0, "starting_time": 0.5},
]
align_media_from_streams(trials_df, streams, ".ethograph/alignment.nwb")
```

See {doc}`loading_ephys` for supported file formats, Kilosort folder setup, and channel mapping.

---

## Custom dimensions (optional)

Any dimension that co-occurs with a time dimension in at least one feature
variable is automatically discovered and gets a selection [combo box](https://www.pythonguis.com/docs/qcombobox/) in the GUI.

::::{tab-set}

:::{tab-item} Xarray

```python
ds["emg"] = xr.DataArray(
    emg_data,                            # shape: (time, channels)
    dims=["time", "channels"],
    coords={"channels": ["biceps", "triceps"]},
)
```

Dimensions **do not need to match across features**. For example, `position`
may have `(time, keypoints, space, individuals)` while `speed` only has
`(time, keypoints, individuals)`. The GUI creates combo boxes for the union of
all discovered dimensions. When a feature doesn't have a selected dimension,
that selection is silently ignored via {func}`~ethograph.utils.xr_utils.sel_valid`:

```python
import ethograph as eto

# "keypoints" and "individuals" are applied; "space" is silently dropped
data, used_kwargs = eto.sel_valid(
    ds["speed"],
    {"keypoints": "nose", "space": "x", "individuals": "mouse1"},
)
```
:::

:::{tab-item} Pynapple

Column names in a {class}`~pynapple.TsdFrame` become a selectable dimension. Objects with identical
column names share a single combo in the GUI.

```python
# position and velocity share "x", "y", "z" columns → one "space" combo
position = nap.TsdFrame(t=time_s, d=pos, columns=["x", "y", "z"])
velocity = nap.TsdFrame(t=time_s, d=vel, columns=["x", "y", "z"])
```
:::

:::{tab-item} NWB

Dimensions are discovered automatically from the NWB data structure when loaded
via pynapple. Multi-column {class}`~pynwb.TimeSeries` produce column-based combos.
:::

::::

---

## Color variables (optional)

Color variables are identified by name: any feature variable with **"rgb"** in
its name (case-insensitive) is automatically offered in the GUI's **Colors**
combo. No special `attrs` are required. The variable should have an `RGB`
dimension of size 3 with values in `[0, 1]` (float) or `[0, 255]` (int):

```python
ds["angle_rgb"] = xr.DataArray(
    rgb_values,                          # shape: (time, keypoints, individuals, 3)
    dims=["time", "keypoints", "individuals", "RGB"],
)
```

To compute angle-based RGB automatically from pose data, use
{func}`~ethograph.io.dataset.add_angle_rgb_to_ds`:

```python
import ethograph as eto

ds = eto.add_angle_rgb_to_ds(ds, smoothing_params={"sigma": 3})
# Creates ds["angles"] and ds["angle_rgb"]
```

---

## Changepoint variables (optional)

Changepoint arrays are binary (`0` or `1`) integer arrays that share the same
time dimension as their target feature. They require:

- `attrs["type"] = "changepoints"`
- `attrs["target_feature"]` — name of the feature variable they annotate

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

To compute changepoints programmatically, use {func}`~ethograph.io.dataset.add_changepoints_to_ds`:

```python
import ethograph as eto
from ethograph.features.changepoints import find_troughs_binary

ds = eto.add_changepoints_to_ds(
    ds,
    target_feature="speed",
    changepoint_name="troughs",
    changepoint_func=find_troughs_binary,
)
# Creates ds["speed_troughs"] with type="changepoints", target_feature="speed"
```

{func}`~ethograph.io.dataset.add_changepoints_to_ds` uses {func}`xarray.apply_ufunc` with `vectorize=True`, so
your detection function only needs to handle a 1-D signal.

---

## Audio changepoints (optional)

```{note}
Audio changepoints format is subject to change, still in development.
```

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



---

## Full example: multi-trial dataset

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

:::{tab-item} Pynapple

```python
import numpy as np
import pynapple as nap
from ethograph.io.nwb_alignment import align_media_per_trial
import pandas as pd

# Feature data (session-wide, pynapple handles trial restriction)
position = nap.TsdFrame(
    t=np.arange(90000) / 30.0,
    d=np.random.randn(90000, 2),
    columns=["x", "y"],
)
speed = nap.Tsd(
    t=np.arange(90000) / 30.0,
    d=np.abs(np.random.randn(90000)),
)

# Save pynapple data
nap.save_file({"position": position, "speed": speed}, "session")

# Alignment: media files + trial timing
trial_table = pd.DataFrame({
    "trial": list(range(1, 11)),
    "start_time": [i * 300.0 for i in range(10)],
    "stop_time": [(i + 1) * 300.0 - 0.5 for i in range(10)],
    "video_cam-1": [f"cam1_trial{tid:03d}.mp4" for tid in range(1, 11)],
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
# NWB files already store trials, media references, and features together.
# No separate alignment step is needed.
#
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

# Add trials
nwbfile.add_trial_column(name="stimulus", description="Stimulus type")
for i in range(10):
    nwbfile.add_trial(
        start_time=i * 300.0,
        stop_time=(i + 1) * 300.0 - 0.5,
        stimulus="tone_A" if i % 2 else "tone_B",
    )

# Add behavioural data as TimeSeries
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

