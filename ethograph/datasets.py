"""Unified dataset registry — single source of truth for all example datasets.

Merges what was previously split across:
- ``EXAMPLE_DATASETS`` in ``utils/download.py`` (download manifests)
- ``TEMPLATES`` in ``gui/dialog_select_template.py`` (GUI card definitions)
- ``EXAMPLE_CONFIGS`` in ``utils/download.py`` (per-dataset config files)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr

DOWNLOAD_BASE = Path.home() / ".ethograph" / "example_data"

DATASETS: dict[str, dict] = {
    "moll2025": {
        # Display
        "name": "Moll et al., 2025 — Tool-using crows",
        "image": "moll1.png",
        "paper_url": "https://doi.org/10.1016/j.cub.2025.08.033",
        # Structure
        "folder": "Moll2025",
        "nc_filename": "Trial_data.nc",
        "has_audio": False,
        "import_labels": True,
        "media": [
            {
                "video_cam-1": "2024-12-17_115_Crow1-cam-1.mp4",
                "pose_cam-1": "2024-12-17_115_Crow1-cam-1DLC.csv",
            },
            {
                "video_cam-1": "2024-12-18_041_Crow1-cam-1.mp4",
                "pose_cam-1": "2024-12-18_041_Crow1-cam-1DLC.csv",
            },
        ],
        # Download
        "release_tag": "moll2025",
        "size_mb": 14,
        "extra_gui_assets": ["Trial_data_labels.tsv"],
        "assets_notebook": [
            "Trial_data.nc",
            "2024-12-17_115_Crow1-cam-1.mp4",
            "2024-12-17_115_Crow1-cam-1DLC.csv",
            "2024-12-17_115_Crow1-cam-2DLC.csv",
            "2024-12-17_115_Crow1_DLC_3D.csv",
            "2024-12-17_115_Crow1-cam-1_s3d.npy",
            "2024-12-18_041_Crow1-cam-1.mp4",
            "2024-12-18_041_Crow1-cam-1DLC.csv",
            "2024-12-18_041_Crow1-cam-2DLC.csv",
            "2024-12-18_041_Crow1_DLC_3D.csv",
            "2024-12-18_041_Crow1-cam-1_s3d.npy",
        ],
        "assets_pynapple": [
            "beakTip_speed.npz",
            "beakTip_velocity.npz",
            "beakTip_position.npz",
            "trials.npz",
        ],
        # Configs written to dest/.ethograph/
        "configs": {
            "mapping.txt": (
                "0 background\n"
                "1 pullOutStick\n"
                "2 diagonalToBox\n"
                "3 toss\n"
                "4 swoop\n"
                "5 reachLeftCorner\n"
                "6 right\n"
                "7 pullOutAlongWall\n"
                "8 swoopOut\n"
                "9 stickToDisp\n"
                "10 stickInDisp\n"
                "11 lookToPellet\n"
                "12 snapPellet\n"
                "13 eat\n"
                "14 reachRightCorner\n"
                "15 nodding\n"
            ),
            "space.yaml": (
                "# Reference geometry for Moll2025 setup.\n"
                "# Points are 3D [x, y, z] in metres. Edges connect points by index.\n"
                "\n"
                "references:\n"
                "  - name: setup\n"
                "    vertices:\n"
                "      - [-7.00,  0.00, 0.65]   # 0: floor front-left\n"
                "      - [-7.00,  9.80, 0.65]   # 1: floor back-left\n"
                "      - [ 6.80,  9.80, 0.65]   # 2: floor back-right\n"
                "      - [ 6.80,  0.00, 0.65]   # 3: floor front-right\n"
                "      - [-7.00,  0.00, 2.75]   # 4: ceiling front-left\n"
                "      - [-7.00,  9.80, 2.75]   # 5: ceiling back-left\n"
                "      - [ 6.80,  9.80, 2.75]   # 6: ceiling back-right\n"
                "      - [ 6.80,  0.00, 2.75]   # 7: ceiling front-right\n"
                "    edges:\n"
                "      - [0, 1]\n"
                "      - [1, 2]\n"
                "      - [2, 3]\n"
                "      - [3, 0]\n"
                "      - [4, 5]\n"
                "      - [5, 6]\n"
                "      - [6, 7]\n"
                "      - [7, 4]\n"
                "      - [0, 4]\n"
                "      - [1, 5]\n"
                "      - [2, 6]\n"
                "      - [3, 7]\n"
                "    color: black\n"
            ),
        },
    },
    "birdpark": {
        "name": "Rüttimann et al., 2025 — Zebra finches in BirdPark",
        "image": "birdpark0.png",
        "paper_url": "https://doi.org/10.7717/peerj.20203",
        "folder": "BirdPark",
        "nc_filename": "copExpBP08_trim.nc",
        "has_audio": True,
        "media": [
            {
                "video_cam-1": "BP_2021-05-25_08-12-51_655154_0380000.mp4",
                "audio_mic-1": "BP_2021-05-25_08-12-51_655154_0380000.wav",
            },
        ],
        "release_tag": "birdpark",
        "size_mb": 76,
    },
    "philodoptera": {
        "name": "Philodoptera — Motor control of sound production in crickets",
        "image": "cricket0.png",
        "paper_url": "",
        "folder": "Philodoptera",
        "nc_filename": "philodoptera.nc",
        "has_audio": True,
        "media": [
            {
                "video_cam-1": "philodoptera.mp4",
                "audio_mic-1": "philodoptera.wav",
                "pose_cam-1": "philodoptera.csv",
            },
        ],
        "release_tag": "philodoptera",
        "size_mb": 4,
    },
    "lockbox": {
        "name": "Reiske et al., 2025 — Mouse Lockbox",
        "image": "lockbox2.gif",
        "paper_url": "https://arxiv.org/abs/2505.15408",
        "folder": "Lockbox",
        "nc_filename": "lockbox.nc",
        "has_audio": False,
        "media": [
            {
                "video_front-view": "2021-02-15_07-32-44_segment1_mouse324_ball_front-view.mp4",
                "video_side-view": "2021-02-15_07-32-44_segment1_mouse324_ball_side-view.mp4",
                "video_top-down-view": "2021-02-15_07-32-44_segment1_mouse324_ball_top-down-view.mp4",
                "pose_front-view": "2021-02-15_07-32-44_segment1_mouse324_ball_front-view-tracks_individual_0.csv",
                "pose_side-view": "2021-02-15_07-32-44_segment1_mouse324_ball_side-view-tracks_individual_0.csv",
                "pose_top-down-view": "2021-02-15_07-32-44_segment1_mouse324_ball_top-down-view-tracks_individual_0.csv",  # noqa: E501
            },
            {
                "video_front-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_front-view.mp4",
                "video_side-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_side-view.mp4",
                "video_top-down-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_top-down-view.mp4",
                "pose_front-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_front-view-tracks_individual_0.csv",  # noqa: E501
                "pose_side-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_side-view-tracks_individual_0.csv",  # noqa: E501
                "pose_top-down-view": "2021-05-31_07-34-21_segment2_mouse291_sliding-door_top-down-view-tracks_individual_0.csv",  # noqa: E501
            },
            {
                "video_front-view": "2021-05-31_07-34-21_segment3_mouse291_stick_front-view.mp4",
                "video_side-view": "2021-05-31_07-34-21_segment3_mouse291_stick_side-view.mp4",
                "video_top-down-view": "2021-05-31_07-34-21_segment3_mouse291_stick_top-down-view.mp4",
                "pose_front-view": "2021-05-31_07-34-21_segment3_mouse291_stick_front-view-tracks_individual_0.csv",
                "pose_side-view": "2021-05-31_07-34-21_segment3_mouse291_stick_side-view-tracks_individual_0.csv",
                "pose_top-down-view": "2021-05-31_07-34-21_segment3_mouse291_stick_top-down-view-tracks_individual_0.csv",  # noqa: E501
            },
        ],
        "release_tag": "lockbox",
        "size_mb": 70,
        "extra_gui_assets": ["_downsample_info.json"],
        "assets_notebook": [
            "lockbox.nc",
            "2021-02-15_07-32-44_segment1_mouse324_ball_front-view.mp4",
            "2021-02-15_07-32-44_segment1_mouse324_ball_front-view-tracks_individual_0.csv",
            "2021-02-15_07-32-44_segment1_mouse324_ball_side-view.mp4",
            "2021-02-15_07-32-44_segment1_mouse324_ball_side-view-tracks_individual_0.csv",
            "2021-02-15_07-32-44_segment1_mouse324_ball_top-down-view.mp4",
            "2021-02-15_07-32-44_segment1_mouse324_ball_top-down-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_front-view.mp4",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_front-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_side-view.mp4",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_side-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_top-down-view.mp4",
            "2021-05-31_07-34-21_segment2_mouse291_sliding-door_top-down-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment3_mouse291_stick_front-view.mp4",
            "2021-05-31_07-34-21_segment3_mouse291_stick_front-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment3_mouse291_stick_side-view.mp4",
            "2021-05-31_07-34-21_segment3_mouse291_stick_side-view-tracks_individual_0.csv",
            "2021-05-31_07-34-21_segment3_mouse291_stick_top-down-view.mp4",
            "2021-05-31_07-34-21_segment3_mouse291_stick_top-down-view-tracks_individual_0.csv",
        ],
    },
    "canary": {
        "name": "Giraudon et al. 2021 - Canary song",
        "image": "canary.png",
        "dataset_url": "https://zenodo.org/records/6521932",
        "folder": "Canary",
        "nc_filename": None,
        "has_audio": True,
        "audio_file": "100_marron1_May_24_2016_62101389.wav",
        "labels_file": "100_marron1_May_24_2016_62101389.audacity.txt",
        "release_tag": "canary",
        "size_mb": 2,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_gui_assets(key: str) -> list[str]:
    """Derive the GUI asset list from dataset metadata.

    Collects: nc_filename + unique media filenames + extra_gui_assets +
    standalone files (audio_file, labels_file).
    """
    ds = DATASETS[key]
    assets: list[str] = []
    seen: set[str] = set()

    def _add(name: str | None) -> None:
        if name and name not in seen:
            assets.append(name)
            seen.add(name)

    _add(ds.get("nc_filename"))
    for row in ds.get("media", []):
        for fname in row.values():
            _add(fname)
    for extra in ds.get("extra_gui_assets", []):
        _add(extra)
    _add(ds.get("audio_file"))
    _add(ds.get("labels_file"))
    return assets


def get_notebook_assets(key: str) -> list[str]:
    """Return notebook assets — explicit list if provided, else same as GUI."""
    return DATASETS[key].get("assets_notebook") or get_gui_assets(key)


def dataset_dir(key: str) -> Path:
    """Return the local download directory for a dataset."""
    return DOWNLOAD_BASE / DATASETS[key]["folder"]


def is_dataset_downloaded(key: str) -> bool:
    """Check whether all GUI assets for a dataset are present locally."""
    if key not in DATASETS:
        return False
    dest = dataset_dir(key)
    return all((dest / name).exists() for name in get_gui_assets(key))


def sample_data() -> "xr.Dataset":
    """Download and return one trial of real pose data (Moll et al., 2025).

    Downloads the Moll2025 crow tool-use dataset (~14 MB) on first call,
    then returns ``itrial(0)`` — an :class:`xarray.Dataset` with position,
    velocity, speed, and acceleration for 14 keypoints.

    Returns
    -------
    xarray.Dataset
        Single-trial dataset with dimensions ``(time, space, keypoints, individuals)``.

    Examples
    --------
    >>> import ethograph as eto
    >>> ds = eto.sample_data()
    >>> ds["speed"]
    <xarray.DataArray 'speed' (time: ..., keypoints: 14, individuals: 1)>
    """
    from ethograph.utils.download import download_assets

    key = "moll2025"
    info = DATASETS[key]
    dest = dataset_dir(key)
    nc_path = dest / info["nc_filename"]

    if not nc_path.exists():
        dest.mkdir(parents=True, exist_ok=True)
        download_assets(
            release_tag=info["release_tag"],
            assets=[info["nc_filename"]],
            dest=dest,
        )

    import ethograph as eto

    dt = eto.open(str(nc_path))
    return dt.itrial(0)


def resolve_dataset_paths(key: str) -> dict:
    """Resolve a dataset's metadata into absolute paths for the IO widget."""
    ds = DATASETS[key]
    dest = dataset_dir(key)
    nc_filename = ds.get("nc_filename")
    result = {
        "name": ds["name"],
        "dataset_key": key,
        "nc_file_path": str(dest / nc_filename) if nc_filename else "",
        "video_folder": str(dest),
        "audio_folder": str(dest) if ds.get("has_audio") else "",
        "pose_folder": str(dest),
        "import_labels": ds.get("import_labels", False),
    }
    if ds.get("labels_file"):
        result["labels_file"] = str(dest / ds["labels_file"])
    if ds.get("audio_file"):
        result["audio_file"] = str(dest / ds["audio_file"])
    return result
