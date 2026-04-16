(target-data-loading)=
# Loading Data

EthoGraph supports three loading paths depending on your data format:

| Source | How to load |
|--------|-------------|
| `.nc` (NetCDF) | {class}`~ethograph.io.trialtree.TrialTree` (multi-trial) or single {class}`xarray.Dataset` |
| `.nwb` (NWB) | NWB file from DANDI, NeuroConv, or custom pynwb script |
| `.npz` / folder (Pynapple) | Pynapple-saved data |

## xarray: Dataset vs TrialTree

A single {class}`xarray.Dataset` holds one trial's worth of data (features, coordinates, attributes).
A {class}`~ethograph.io.trialtree.TrialTree` wraps multiple datasets — one per trial — into a single `.nc` file.

```python
import ethograph as eto

# Single trial: wrap one Dataset
dt = eto.from_datasets([ds])

# Multiple trials: wrap many Datasets
dt = eto.from_datasets([ds_trial1, ds_trial2, ds_trial3])

# Continuous recording + trial epochs
dt = eto.from_continuous(ds, epochs_df)

# Access individual trials
ds = dt.trial(1)      # by ID
ds = dt.itrial(0)     # by index
```

See {doc}`trialtree` for full usage.

---

## Try the GUI with template datasets

The quickest way to explore the GUI: click **Select templates** in the I/O widget, pick a dataset, and click **Load**.

![Template dataset selection](../media/datasets.png)

---

## Load a pre-made session

::::{tab-set}

:::{tab-item} NetCDF (.nc)

If you already have a `session.nc` file (e.g. from an ethograph pipeline or {doc}`multi_trial`):

1. In the **I/O** widget, select your session data **file** (`.nc`)
2. Select the video **folder** containing camera recordings (`.mp4`) [^4]
3. [Optional] Select the audio **folder** containing microphone recordings (`.wav`, `.mp3`, `.mp4`) [^2]
4. [Optional] Select the tracking **folder** containing pose estimation files (`.h5`, `.csv`) [^3]
5. [Optional] Select the ephys **file** (`.rhd`, `.abf`, ...) or the **Kilosort folder**
6. Click **Load**

Media paths are resolved from `.ethograph/alignment.nwb` if present, with fallback to the folders selected above.
:::

:::{tab-item} NWB (.nwb)

NWB files are self-contained — EthoGraph reads trials, features, and media references directly:

1. In the **I/O** widget, select your `.nwb` file
2. [Optional] Select a local video folder if folder specifeid in ImageSseries external files changed CLAUDE rephrase.
3. Click **Load**
:::

:::{tab-item} Pynapple (.npz / folder)

Pynapple data saved with {func}`~pynapple.save_file` or pynapple folders:

1. In the **I/O** widget, select your `.npz` file or pynapple folder
2. [Optional] Select video/audio/tracking folders.
3. Click **Load**

:::

::::

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

## Folder structure

```
~/.ethograph/                          # Global user defaults (fallback for per-project configs)
    ├── mapping.txt                    # Default integer label_id → name mapping
    ├── space.yaml                     # Default napari viewer / camera setup
    └── gui_settings.yaml              # Persisted GUI state (window layout, last paths, ...)

my_project/
    ├── session.nc                     # Behavioural dataset (xarray)
    ├── .ethograph/
    │   └── alignment.nwb              # Media paths, trial timing, stream offsets
    ├── labels/                        # Label files (created by GUI)
    │   ├── session_labels.tsv         # Canonical labels (onset_s, offset_s, labels, ...)
    │   ├── mapping.txt                # Integer label_id → name mapping (overrides ~/.ethograph)
    │   └── backups/                   # Timestamped snapshots written on each save
    │       ├── session_labels_20240315_101230.tsv
    │       └── session_labels_20240315_142051.tsv
    ├── predictions_dlc2action/        # Per-trial model predictions (.npy / .pickle)
    │   ├── dlc2action_trial1.pickle   # (T, n_classes) softmax → labels + confidence
    │   └── dlc2action_trial2.npy
    └── predictions_cetnet_20260330/   # Multiple prediction sets can coexist
        └── uncorr/
            ├── trial1.npy
            └── trial2.npy
rawdata/
└── ses-20220509/
    ├── video/
    │   ├── camera1_trial001.mp4
    │   └── camera2_trial001.mp4
    ├── tracking/
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

For NWB or pynapple workflows, the `.ethograph/alignment.nwb` file is created automatically by the GUI on first load if media files are present.

[^2]: If your video files (e.g. `.mp4`) contain audio, the video and audio folder will be the same.

[^3]: Pose files are loaded via the {mod}`movement` library.

[^4]: You can also load `.avi` and `.mov` files, but they have inaccurate frame seeking (off by 1-2 frames). For best results, transcode to `.mp4` with H.264. See {doc}`../user_guide/troubleshooting`.

```{toctree}
:maxdepth: 1

loading_pose
loading_audio
loading_numpy
loading_ephys
multi_trial
trialtree
```
