(target-data-loading)=
# Loading Data

EthoGraph supports three loading paths depending on your data format:

| Source | How to load |
|--------|-------------|
| `.nc` (NetCDF) | {class}`~ethograph.io.trialtree.TrialTree` (multi-trial) or single {class}`xarray.Dataset` |
| `.nwb` (NWB) | NWB file from DANDI, NeuroConv, or custom pynwb script |
| `.npz` / folder (Pynapple) | Pynapple-saved data |


---

## Create a session from your own data

Click **Create with own data** in the I/O widget. A dialog guides you through creating a session from several supported sources. After generation the I/O fields are auto-populated so you can click **Load** immediately.

The dialog handles **single-file** workflows. For multiple trials, multiple cameras, or multiple microphone files, use the **Multiple trials** tab in the wizard, or write a short Python script.

| Format | When to use | Guide |
|--------|-------------|-------|
| Pose file | DLC, SLEAP, LightningPose `.h5`/`.csv` | {doc}`loading_pose` |
| Audio file | Vocal / acoustic data | {doc}`loading_audio` |
| Numpy file | Pre-computed feature array | {doc}`loading_numpy` |
| Ephys recording | Raw electrophysiology +/- Kilosort | {doc}`loading_ephys` |
| Custom script | Multi-trial, multi-cam, multi-mic | {doc}`multi_trial` |

---

---

## Load a pre-made session

::::{tab-set}

:::{tab-item} NetCDF (.nc)

If you already have a `session.nc` file (e.g. from an ethograph pipeline or {doc}`multi_trial`):

1. In the **I/O** widget, select your session data **file** (`.nc`)
2. Select the video **folder** containing camera recordings (`.mp4`)
3. [Optional] Select the audio **folder** containing microphone recordings (`.wav`, `.mp3`, `.mp4`)
4. [Optional] Select the tracking **folder** containing pose estimation files (`.h5`, `.csv`)
5. [Optional] Select an ephys **file** (`.rhd`, `.abf`, ...) containing raw multi-channel data
6. [Optional] Select a **Kilosort folder** to display single neurons
7. Click **Load**

Media paths are resolved from `.ethograph/alignment.nwb` if present, with fallback to the folders selected above.
:::

:::{tab-item} NWB (.nwb)

NWB files are self-contained — EthoGraph reads trials, behavioural time series (e.g. {class}`~pynwb.behavior.PoseEstimationSeries`), and media references directly:

1. In the **I/O** widget, select your `.nwb` file
2. [Optional] Select a local video folder if the paths stored in {class}`~pynwb.image.ImageSeries` `.external_file` no longer point to the correct location
3. Click **Load**

:::

:::{tab-item} Pynapple (.npz / folder)

Pynapple data saved with {func}`~pynapple.save_file` or pynapple folders:

1. In the **I/O** widget, select your `.npz` file or pynapple folder
2. [Optional] Select video/audio/tracking folders
3. [Optional] Select a **pynapple folder** containing spike-sorted units to display single neurons
4. Click **Load**

:::

::::




## xarray: Dataset vs TrialTree

You can save your `session.nc` as either a plain {class}`xarray.Dataset` or a {class}`~ethograph.io.trialtree.TrialTree`. The GUI accepts both.

**Plain Dataset** — a single `xr.Dataset` saved with {func}`xarray.Dataset.to_netcdf`. When loaded, it is automatically treated as a single trial:

```python
import xarray as xr

ds = xr.Dataset(...)
ds.to_netcdf("session.nc")
# → GUI loads this as one trial
```

**TrialTree** — wraps multiple datasets (one per trial) into a single `.nc` file. Use this when your experiment has multiple trials:

```python
import ethograph as eto

# Multiple trials: one Dataset per trial
dt = eto.from_datasets([ds_trial1, ds_trial2, ds_trial3])
dt.save("session.nc")

# Access individual trials
ds = dt.trial(1)      # by ID
ds = dt.itrial(0)     # by index
```

If you have a session-long `xr.Dataset` that you want to split into trials, see {ref}`Splitting a continuous recording <target-from-continuous>` in the multi-trial guide.

See {doc}`../api/trialtree` for full TrialTree usage.



---

## Folder structure

::::{tab-set}

:::{tab-item} xarray (.nc)

```
~/.ethograph/                          # Global user defaults
    ├── mapping.txt                    # Default integer label_id → name mapping
    ├── space.yaml                     # Extra config for space plot
    └── gui_settings.yaml              # Across-session GUI state

my_project/
    ├── session.nc                     # Behavioural dataset (TrialTree or plain Dataset)
    ├── session_labels.tsv             # Session labels
    ├── session_metadata.tsv           # Trial-level metadata
    ├── .ethograph/
    │   ├── alignment.nwb              # Media paths, trial timing, stream offsets
    │   └── local_settings.yaml        # Session-specific GUI state
    │
    ├── labels/
    │   ├── backups/
    │   │  └── session_labels_20240315_101230.tsv
    │   └── predictions_asformer_20240215/
    │       ├── trial1.npy
    │       └── trial2.npy

    ├── video/
    │   ├── camera1_trial001.mp4
    │   └── camera2_trial001.mp4
    ├── pose/                      # External pose files (DLC, SLEAP, ...)
    │   ├── trial001_pose.h5
    │   └── ...
    ├── audio/
    │   ├── mic1_trial001.wav
    │   └── ...
    └── ephys/
        ├── recording.rhd
        └── kilosort4/
            ├── params.py
            ├── spike_times.npy
            ├── spike_clusters.npy
            ├── channel_positions.npy
            ├── channel_map.npy
            ├── templates.npy
            └── cluster_info.tsv
```

:::

:::{tab-item} NWB (.nwb)

```
~/.ethograph/                          # Global user defaults
    ├── mapping.txt                    # Default integer label_id → name mapping
    ├── space.yaml                     # Extra config for space plot
    └── gui_settings.yaml              # Across-session GUI state

my_project/
    ├── session.nwb                    # Self-contained: trials, time series,
    │                                  # pose (PoseEstimationSeries), video
    │                                  # refs (ImageSeries.external_file)
    |
    ├── session_labels.tsv             # Session labels
    ├── session_metadata.tsv           # Trial-level metadata
    │    
    ├── .ethograph/
    │   ├── alignment.nwb              # inherit/overwrite alignment in session.nwb
    │   └── local_settings.yaml        # Session-specific GUI state
    │
    ├── labels/
    │   ├── backups/
    │   │   └── session_labels_20240315_101230.tsv
    │   └── predictions_asformer_20240215/
    │       ├── trial1.npy
    │       └── trial2.npy
    │
    └── video/                         # Only needed if ImageSeries paths
        ├── camera1_trial001.mp4       # no longer point to the right location
        └── camera2_trial001.mp4
```

Pose estimation and other behavioural time series are stored inside the `.nwb` file — no external tracking folder needed.

:::

:::{tab-item} Pynapple (.npz / folder)

```
~/.ethograph/                          # Global user defaults
    ├── mapping.txt                    # Default integer label_id → name mapping
    ├── space.yaml                     # Extra config for space plot
    └── gui_settings.yaml              # Across-session GUI state

my_project/
    ├── position.npz                   # Pynapple Tsd/TsdFrame objects
    ├── speed.npz
    ├── units.npz                      # TsGroup of spike times
    │
    ├── labels.tsv                     # Session labels
    ├── metadata.tsv                   # Trial-level metadata
    │
    ├── .ethograph/
    │   ├── alignment.nwb              # Media paths, trial timing, stream offsets
    │   └── local_settings.yaml        # Session-specific GUI state
    │
    ├── labels/
    │   ├── backups/
    │   │   └── labels_20240315_101230.tsv
    │   └── predictions_asformer_20240215/
    │       ├── trial1.npy
    │       └── trial2.npy
    │
    ├── video/
    │   ├── camera1_trial001.mp4
    │   └── ...
    └── audio/
        ├── mic1_trial001.wav
        └── ...
```

:::

::::

The `.ethograph/alignment.nwb` sidecar is created automatically by the GUI on first load when media files are present.

```{toctree}
:maxdepth: 1

loading_pose
loading_audio
loading_numpy
loading_ephys
multi_trial
```
