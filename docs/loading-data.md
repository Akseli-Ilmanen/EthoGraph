# Loading Data

1. In `I/O` widget, select your session data **file** (`.nc`) 
2. Select the video **folder** containing camera recordings (`.mp4`, `.mov`, `.avi`)
3. [Optional] Select the audio **folder** containing microphone recordings  `.wav`, `.mp3`, `.mp4` [^1]
4. [Optional] Select the tracking **folder** containing pose estimation files (`.h5`, `.csv`) [^2]
5. Click `Load` to load the dataset and populate the interface



## Folder Structure

```
processed_data/
    └── ses-20220509/
        ├── sesssion.nc              # Main behavioral dataset (required)
        └── labels/                  # Label labels (created by GUI)
            ├── data_labels_20240315_143022.nc
            └── data_labels_20240316_091045.nc
rawdata/
└── ses-20220509/
    ├── video/
    │   ├── camera1_trial001.mp4     # Camera recordings (required)
    │   ├── camera1_trial002.mp4
    │   ├── camera2_trial001.mp4
    │   └── camera2_trial002.mp4
    ├── audio/                       # Microphone recordings (optional)
    │   ├── mic1_trial001.wav
    │   ├── mic1_trial002.wav
    │   └── ...
    └── tracking/                    # Pose estimation files (optional)
        ├── trial001_pose.h5
        ├── trial001_pose.csv
        ├── trial002_pose.h5
        ├── trial002_pose.csv
        └── ...
```

[^1]: If your video files (e.g. `.mp4`) contain audio, the video and audio folder will be the same.
[^2]: Loading of pose estimation points and tracks occurs via `movement` library. See further here [Movement IO](https://movement.neuroinformatics.dev/latest/user_guide/input_output.html).

