(target-nwb-alignment)=
# NWB alignment

EthoGraph stores session metadata (trial timing, media paths, stream
offsets) in NWB. For `.nwb` sources the source file is used directly —
no sidecar is created and edits go back into the source. For other
formats (xarray `.nc`, `.npz`, pynapple folders) a sidecar
**`.ethograph/alignment.nwb`** is written next to the data file to tie
features to video, audio, ephys, and trial structure.

This page explains what the alignment file contains and how to read / write
it. For the step-by-step workflow of creating one alongside a multi-trial
dataset, see {doc}`../getting_started/multi_trial`.

---

## What the alignment file contains

| Concept | Stored as | Read via |
|---------|-----------|----------|
| **Trial timing** | `nwb.trials` table with `start_time`, `stop_time`, and custom columns | `alignment.trials_df`, `alignment.start_time(trial)`, `alignment.stop_time(trial)` |
| **Media files** | {class}`~pynwb.image.ImageSeries` in `nwb.acquisition` per stream/device | `alignment.resolve_media_path(trial, stream, device)` |
| **Stream rates** | `rate` field on each ImageSeries | `alignment.get_stream_rate(stream, device)` |
| **Stream offsets** | `starting_time` on ImageSeries — when sample 0 occurs in session time | `alignment.stream_offset_for_trial(trial, stream, device)` |
| **Cameras / mics** | Device names parsed from ImageSeries names | `alignment.cameras`, `alignment.mics` |

---

## Trial-relative vs session-absolute time

EthoGraph uses two time conventions, and the alignment file is what
connects them:

- **Trial-relative** (`onset_s`, `offset_s`, internal feature time): each
  trial starts at `0.0`. This matches pose trackers, video files, and
  per-trial audio recordings.
- **Session-absolute** (`onset_global`, `offset_global`, ephys timestamps):
  time is measured from the start of the recording session.

Conversion uses the trials table: `onset_global = alignment.start_time(trial) + onset_s`.

---

## Reading an existing alignment file

```python
from ethograph.io.nwb_alignment import NWBAlignment

alignment = NWBAlignment("my_project/.ethograph/alignment.nwb")
print(alignment.trials_df)
print(alignment.cameras)          # ["cam-1", "cam-2"]
print(alignment.mics)             # ["mic-1"]
print(alignment.start_time(1))    # 0.0
alignment.close()
```

The same interface is exposed via `dt.nwb_alignment` on a loaded TrialTree
and via `app_state.nwb_alignment` inside the GUI.

See {class}`~ethograph.io.nwb_alignment.NWBAlignment` for the full API.

---

## Creating an alignment file

Two builders cover the common cases. Both produce a valid NWB file that the
GUI loads without further conversion.

### `align_media_per_trial` — media maps 1:1 to trials

```python
import pandas as pd
from ethograph.io.nwb_alignment import align_media_per_trial

trial_table = pd.DataFrame({
    "trial": [1, 2, 3],
    "start_time": [0.0, 300.0, 600.0],
    "stop_time": [299.5, 599.5, 899.5],
    "video_cam-1": ["t1.mp4", "t2.mp4", "t3.mp4"],
    "pose_cam-1":  ["t1.h5", "t2.h5", "t3.h5"],
    "audio_mic-1": ["t1.wav", "t2.wav", "t3.wav"],
})

align_media_per_trial(
    trial_table,
    stream_rates={"video": 30.0, "pose": 30.0, "audio": 48000.0},
    output_path=".ethograph/alignment.nwb",
)
```

Column convention: `{stream}_{device}` — e.g. `video_cam-1`, `audio_mic-1`,
`pose_cam-1`. Extra columns (e.g. `stimulus`, `condition`) become trial
attributes and flow through to label TSV exports.

### `align_media_from_streams` — session-wide or mixed files

For recordings where audio or ephys is one continuous session file, or the
ephys clock has its own starting offset, use the stream-oriented builder:

```python
from ethograph.io.nwb_alignment import align_media_from_streams

streams = [
    {"name": "video_cam-1", "files": ["t1.mp4", "t2.mp4", "t3.mp4"], "rate": 30.0},
    {"name": "audio_mic-1", "files": ["session.wav"], "rate": 48000.0, "starting_time": 0.0},
    {"name": "ephys_probe-1", "files": ["session.dat"], "rate": 30000.0, "starting_time": 0.5},
]
align_media_from_streams(trials_df, streams, ".ethograph/alignment.nwb")
```

See {doc}`../getting_started/multi_trial` for full worked examples and
stream-spec options (`timestamps`, `starting_time`, per-device rates).

---

## References

- {class}`~ethograph.io.nwb_alignment.NWBAlignment` — reader API
- {func}`~ethograph.io.nwb_alignment.align_media_per_trial` — per-trial builder
- {func}`~ethograph.io.nwb_alignment.align_media_from_streams` — session-wide builder
- {doc}`../getting_started/multi_trial` — multi-trial setup walkthrough
- {doc}`../getting_started/data_requirements` — dataset format requirements
