# Loading Data

EthoGraph works with NetCDF (`.nc`) session files. You can either load a pre-made `session.nc`[^1] or create one from your own data using the built-in creation dialog.

---

## Option 1: Load a pre-made session.nc

If you already have a `session.nc` file (e.g. from an ethograph pipeline or custom script):

1. In the `I/O` widget, select your session data **file** (`.nc`)
2. Select the video **folder** containing camera recordings (`.mp4`, `.mov`, `.avi`)
3. [Optional] Select the audio **folder** containing microphone recordings `.wav`, `.mp3`, `.mp4` [^2]
4. [Optional] Select the tracking **folder** containing pose estimation files (`.h5`, `.csv`) [^3]
5. Click `Load` to load the dataset and populate the interface

---

## Option 2: Create a session.nc from your own data

If you don't have a `session.nc` file yet, click the **Create session.nc** button in the `I/O` widget. A dialog will guide you through creating one from several supported data sources:

### 1) From a pose file (DeepLabCut, SLEAP, ...)

Use this if you have pose estimation output from tracking software.

- **Source software**: Select the software that generated the file (DeepLabCut, SLEAP, LightningPose, etc.)
- **Pose file**: The tracking output file
- **Video file**: Path to corresponding video file (`.mp4`, `.mov`, `.avi`).
- **Video frame rate**
- **Output path**: Where to save the generated `session.nc`

After generation, the I/O fields are auto-populated so you can click `Load` immediately.

### 2) From an xarray dataset (Movement style)

Use this if you have a [Movement](https://movement.neuroinformatics.dev/latest/user_guide/movement_dataset.html)-style xarray dataset saved as `.nc`.

- **Dataset file**: The Movement-style `.nc` file
- **Video file**: Path to corresponding video file (`.mp4`, `.mov`, `.avi`).
- **Output path**: Where to save the generated `session.nc`

### 3) From an audio file

Use this if your primary data source is audio (e.g. birdsong recordings). If your `.mp4` video contains audio, you can use that same file as the audio source.

- **Video file**: Path to video file (`.mp4`, `.mov`, `.avi`).
- **Audio file**: Path to corresponding audio file (`.wav`, `.mp3`, `.mp4`, `.flac`)
- **Video frame rate**: 
- **Audio sample rate**
- **Individuals** (optional): Comma-separated list of individual names (e.g. `bird1, bird2, bird3`)
- **Load video motion features**: If enabled, extracts a frame-to-frame motion intensity signal from the video using FFmpeg. This provides a 1D movement proxy aligned to the same time axis as your audio.
- **Output path**: Where to save the generated `session.nc`

### 4) From a numpy (.npy) file

Use this if you have pre-computed features stored as a numpy array. The file should contain a 2D array with shape `(n_samples, n_variables)` or `(n_variables, n_samples)`. Longer dimension is assumed to be `n_samples`.

- **Video file**: Path to corresponding video file (`.mp4`, `.mov`, `.avi`).
- **Npy file**: Path to `.npy` file containing your feature array
- **Video frame rate**: 
- **Data sampling rate**: The sampling rate of your numpy data (in Hz)
- **Individuals** (optional): Comma-separated list of individual names (e.g. `bird1, bird2, bird3`)
- **Load video motion features**: If enabled, extracts a frame-to-frame motion intensity signal from the video using FFmpeg. This provides a 1D movement proxy aligned to the same time axis as your audio.
- **Output path**: Where to save the generated `session.nc`

### 5) Tutorials for custom .nc files

For more advanced use cases, see the [tutorials](https://github.com/Akseli-Ilmanen/EthoGraph/tree/main/tutorials) for creating custom `.nc` files programmatically.

---

## Folder Structure

```
processed_data/
    └── ses-20220509/
        ├── session.nc                 # Main behavioral dataset (required)
        └── labels/                    # Label files (created by GUI)
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
[^1]: `session.nc` is just an example file name, you may call it differently.
[^2]: If your video files (e.g. `.mp4`) contain audio, the video and audio folder will be the same.
[^3]: Loading of pose estimation points and tracks occurs via the `movement` library. See [Movement IO](https://movement.neuroinformatics.dev/latest/user_guide/input_output.html).
