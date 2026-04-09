"""Download example datasets from GitHub releases."""

from pathlib import Path
from typing import Callable
from urllib.request import urlopen

_RELEASE_BASE = "https://github.com/Akseli-Ilmanen/EthoGraph/releases/download"

# Default mapping written to ~/.ethograph/mapping.txt if it doesn't exist.
DEFAULT_MAPPING = (
    "0 Background\n"
    "1 Idle\n"
    "2 Carry\n"
    "3 Deposit\n"
    "4 Dunk\n"
    "5 Feed\n"
    "6 Fly\n"
    "7 Hook\n"
    "8 Insert\n"
    "9 Look\n"
    "10 Manipulate\n"
    "11 Peck\n"
    "12 Probe\n"
    "13 Pull\n"
    "14 Push\n"
    "15 Reach\n"
    "16 Regrip\n"
    "17 Retrieve\n"
    "18 Shake\n"
    "19 Step\n"
    "20 Walk\n"
    "21 Wipe\n"
)

# Per-dataset configs that override the global default.
# Written into dest/.ethograph/ after download.
EXAMPLE_CONFIGS: dict[str, dict[str, str]] = {
    "moll2025": {
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
            "# Reference geometry for Moll2025 aviary.\n"
            "# Points are 3D [x, y, z] in metres. Edges connect points by index.\n"
            "\n"
            "references:\n"
            "  - name: aviary\n"
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
}


_OPEN_TSV_VBS = (
    'Set objExcel = CreateObject("Excel.Application")\n'
    "objExcel.Visible = True\n"
    "objExcel.Workbooks.OpenText WScript.Arguments(0), , , 1, 1, False, True\n"
)


def _tsv_reg_content(vbs_path: Path) -> str:
    """Generate .reg file content pointing to *vbs_path*."""
    escaped = str(vbs_path).replace("\\", "\\\\")
    return (
        "Windows Registry Editor Version 5.00\n"
        "\n"
        "[HKEY_CLASSES_ROOT\\.tsv]\n"
        '@="TsvFile"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile]\n"
        '@="Tab-Separated Values"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile\\DefaultIcon]\n"
        '@="excel.exe,1"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile\\shell]\n"
        '@="openexcel"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile\\shell\\openexcel]\n"
        '@="Open TSV with Excel"\n'
        '"FriendlyAppName"="Excel (Tab-Separated)"\n'
        "\n"
        "[HKEY_CLASSES_ROOT\\TsvFile\\shell\\openexcel\\command]\n"
        f'@="wscript.exe \\"{escaped}\\" \\"%1\\""\n'
    )


_SETUP_TSV_MAC = """\
#!/bin/bash
# Double-click this file to make .tsv files always open in Excel.

if ! command -v duti &> /dev/null; then
    echo "Installing duti (requires Homebrew)..."
    brew install duti
fi

if command -v duti &> /dev/null; then
    duti -s com.microsoft.Excel .tsv all
    echo "Done — .tsv files will now open with Excel."
else
    echo "Could not install duti."
    echo "Manual alternative: right-click any .tsv → Get Info → Open With → Microsoft Excel → Change All"
fi
"""


def ensure_default_configs() -> None:
    """Write default configs to ``~/.ethograph/`` if they don't exist yet."""
    import sys

    from ethograph.utils.paths import SETTINGS_DIR
    global_dir = Path.home() / SETTINGS_DIR
    global_dir.mkdir(parents=True, exist_ok=True)
    mapping = global_dir / "mapping.txt"
    if not mapping.exists():
        mapping.write_text(DEFAULT_MAPPING, encoding="utf-8")

    if sys.platform == "win32":
        internal = global_dir / "_internal"
        internal.mkdir(parents=True, exist_ok=True)
        vbs = internal / "open-tsv.vbs"
        if not vbs.exists():
            vbs.write_text(_OPEN_TSV_VBS, encoding="utf-8")
        reg = global_dir / "Double-click to open TSV files in Excel.reg"
        if not reg.exists():
            reg.write_text(_tsv_reg_content(vbs), encoding="utf-8")
    elif sys.platform == "darwin":
        setup = global_dir / "Double-click to open TSV files in Excel.command"
        if not setup.exists():
            setup.write_text(_SETUP_TSV_MAC, encoding="utf-8")
            setup.chmod(0o755)

EXAMPLE_DATASETS = {
    "moll2025": {
        "release_tag": "moll2025",
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
        "assets_gui": [
            "Trial_data.nc",
            "Trial_data_labels.tsv",
            "2024-12-17_115_Crow1-cam-1.mp4",
            "2024-12-17_115_Crow1-cam-1DLC.csv",
            "2024-12-18_041_Crow1-cam-1.mp4",
            "2024-12-18_041_Crow1-cam-1DLC.csv",
        ],
        "size_mb": 14,
    },
    "birdpark": {
        "release_tag": "birdpark",
        "assets_gui": [
            "copExpBP08_trim.nc",
            "BP_2021-05-25_08-12-51_655154_0380000.mp4",
            "BP_2021-05-25_08-12-51_655154_0380000.wav",
        ],
        "assets_notebook": [
            "copExpBP08_trim.nc",
            "BP_2021-05-25_08-12-51_655154_0380000.mp4",
            "BP_2021-05-25_08-12-51_655154_0380000.wav",
        ],
        "size_mb": 76,
    },
    "philodoptera": {
        "release_tag": "philodoptera",
        "assets_gui": [
            "philodoptera.nc",
            "philodoptera.mp4",
            "philodoptera.wav",
            "philodoptera.csv",
        ],
        "assets_notebook": [
            "philodoptera.nc",
            "philodoptera.mp4",
            "philodoptera.wav",
            "philodoptera.csv",
        ],
        "size_mb": 4,
    },
    "lockbox": {
        "release_tag": "lockbox",
        "assets_gui": [
            "lockbox.nc",
            "_downsample_info.json",
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
        "size_mb": 70,
    },
    "canary": {
        "release_tag": "canary",
        "assets_gui": [
            "100_marron1_May_24_2016_62101389.audacity.txt",
            "100_marron1_May_24_2016_62101389.wav",
        ],
        "assets_notebook": [
            "100_marron1_May_24_2016_62101389.audacity.txt",
            "100_marron1_May_24_2016_62101389.wav",
        ],
        "size_mb": 2,
    },
}


def write_example_configs(dataset_key: str, dest: Path) -> None:
    """Write bundled config files into ``dest/.ethograph/``."""
    configs = EXAMPLE_CONFIGS.get(dataset_key)
    if not configs:
        return
    from ethograph.utils.paths import SETTINGS_DIR
    config_dir = Path(dest) / SETTINGS_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, content in configs.items():
        (config_dir / name).write_text(content, encoding="utf-8")


def download_assets(
    release_tag: str,
    assets: list[str],
    dest: Path,
    on_progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Download asset files from a GitHub release to *dest*.

    Parameters
    ----------
    release_tag : str
        GitHub release tag (e.g. ``"moll2025"``).
    assets : list[str]
        Filenames to download.
    dest : Path
        Local directory to save files into (created if missing).
    on_progress : callable, optional
        ``(completed_count, current_filename)`` callback.
    cancelled : callable, optional
        Returns ``True`` to abort the download loop.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(assets):
        if cancelled and cancelled():
            return
        local_path = dest / name
        if local_path.exists():
            if on_progress:
                on_progress(i + 1, name)
            continue
        url = f"{_RELEASE_BASE}/{release_tag}/{name}"
        if on_progress:
            on_progress(i, name)
        with urlopen(url) as resp:  # noqa: S310
            local_path.write_bytes(resp.read())
        if on_progress:
            on_progress(i + 1, name)


def is_downloaded(release_tag: str, dest: Path) -> bool:
    """Check whether all GUI assets for a dataset are already present."""
    info = EXAMPLE_DATASETS.get(release_tag)
    if info is None:
        return False
    return all((Path(dest) / name).exists() for name in info["assets_gui"])


def download_example_dataset(
    key: str,
    dest: Path,
    verbose: bool = True,
) -> Path | None:
    """High-level helper: download an example dataset by key.

    Downloads assets and writes bundled configs (e.g. ``mapping.txt``)
    into ``dest/.ethograph/``.

    Parameters
    ----------
    key : str
        One of ``"moll2025"``, ``"birdpark"``, ``"philodoptera"``.
    dest : Path
        Directory to download into.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    Path to ``mapping.txt`` if one was created, otherwise ``None``.
    """
    info = EXAMPLE_DATASETS[key]
    assets = info["assets_notebook"]

    def _print_progress(count: int, name: str) -> None:
        total = len(assets)
        if count < total:
            print(f"Downloading {name}... ({count}/{total})")
        else:
            print(f"  {name} ({count}/{total})")

    download_assets(
        release_tag=info["release_tag"],
        assets=assets,
        dest=dest,
        on_progress=_print_progress if verbose else None,
    )

    ensure_default_configs()
    write_example_configs(key, dest)
    from ethograph.utils.paths import SETTINGS_DIR
    mapping_path = Path(dest) / SETTINGS_DIR / "mapping.txt"
    if mapping_path.exists():
        if verbose:
            print(f"  mapping: {mapping_path}")
        return mapping_path
    return None


def setup_birdpark_continuous(
    dest: Path | None = None,
    n_trials: int = 3,
    chunk: float = 20.0,
    verbose: bool = True,
) -> Path:
    """Create a BirdPark Continuous variant with a multi-trial alignment NWB.

    Symlinks (or copies on Windows) the BirdPark assets into *dest* and
    creates a ``.ethograph/alignment.nwb`` that treats the single 60 s
    recording as session-wide continuous media split into *n_trials* trials.

    Parameters
    ----------
    dest
        Output directory. Defaults to ``~/.ethograph/example_data/BirdParkContinuous``.
    n_trials
        Number of equal-length trials to split the recording into.
    chunk
        Duration of each trial in seconds.
    verbose
        Print progress to stdout.

    Returns
    -------
    Path to the created directory.
    """
    import shutil

    import numpy as np
    import pandas as pd
    import pynwb
    from pynwb import NWBHDF5IO
    from pynwb.image import ImageSeries

    if dest is None:
        dest = Path.home() / ".ethograph" / "example_data" / "BirdParkContinuous"

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    bp_info = EXAMPLE_DATASETS["birdpark"]
    bp_src = Path.home() / ".ethograph" / "example_data" / "BirdPark"

    for asset in bp_info["assets_gui"]:
        src = bp_src / asset
        dst = dest / asset
        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(
                f"BirdPark asset not found: {src}. "
                "Download birdpark first via download_example_dataset('birdpark', ...)"
            )
        shutil.copy2(src, dst)
        if verbose:
            print(f"  copied {asset}")

    # Also copy the .nc
    nc_name = "copExpBP08_trim.nc"
    nc_dst = dest / nc_name
    if not nc_dst.exists():
        shutil.copy2(bp_src / nc_name, nc_dst)

    video_name = "BP_2021-05-25_08-12-51_655154_0380000.mp4"
    audio_name = "BP_2021-05-25_08-12-51_655154_0380000.wav"
    fps = 47.68

    epochs = pd.DataFrame({
        "trial": list(range(1, n_trials + 1)),
        "start_time": [i * chunk for i in range(n_trials)],
        "stop_time": [(i + 1) * chunk for i in range(n_trials)],
    })

    from datetime import datetime
    from uuid import uuid4
    from dateutil.tz import tzlocal

    nwbfile = pynwb.NWBFile(
        session_description="BirdPark continuous — session-wide media alignment.",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    nwbfile.add_trial_column(name="trial", description="Trial number")
    nwbfile.add_trial_column(name="video_cam-1", description="video filename")
    nwbfile.add_trial_column(name="audio_mic-1", description="audio filename")
    for _, row in epochs.iterrows():
        nwbfile.add_trial(
            start_time=float(row["start_time"]),
            stop_time=float(row["stop_time"]),
            trial=row["trial"],
            **{"video_cam-1": video_name, "audio_mic-1": audio_name},
        )

    session_end = float(epochs["stop_time"].max())
    n_video_frames = int(session_end * fps)
    video_ts = np.arange(n_video_frames) / fps

    nwbfile.create_device(name="cam-1", description="video device cam-1")
    nwbfile.add_acquisition(
        ImageSeries(
            name="video_cam-1",
            description="video from cam-1",
            external_file=[video_name],
            format="external",
            starting_frame=np.array([0], dtype=np.int32),
            timestamps=video_ts,
        )
    )

    nwb_path = dest / ".ethograph" / "alignment.nwb"
    nwb_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    ensure_default_configs()

    if verbose:
        print(f"BirdPark Continuous ready at {dest}")
        print(f"  {n_trials} trials x {chunk}s, alignment: {nwb_path}")

    return dest


def setup_moll2025_pynapple(
    dest: Path | None = None,
    verbose: bool = True,
) -> Path:
    """Create a Moll2025 Pynapple variant with alignment NWB linking to original media.

    Copies pynapple ``.npz`` files and labels TSV, writes configs
    (``mapping.txt``, ``space.yaml``), and creates
    ``.ethograph/alignment.nwb`` whose trials table references the video
    and pose files in the original ``Moll2025`` folder.

    Parameters
    ----------
    dest
        Output directory.  Defaults to
        ``~/.ethograph/example_data/Moll2025_pynapple``.
    verbose
        Print progress to stdout.

    Returns
    -------
    Path to the created directory.
    """
    import shutil

    import numpy as np
    import pandas as pd
    import pynapple as nap
    import pynwb
    from pynwb import NWBHDF5IO
    from pynwb.image import ImageSeries

    if dest is None:
        dest = Path.home() / ".ethograph" / "example_data" / "Moll2025_pynapple"

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    moll_src = Path.home() / ".ethograph" / "example_data" / "Moll2025"
    if not moll_src.exists():
        raise FileNotFoundError(
            f"Moll2025 source not found: {moll_src}. "
            "Download moll2025 first via download_example_dataset('moll2025', ...)"
        )

    # Copy pynapple assets + extras
    pynapple_assets = EXAMPLE_DATASETS["moll2025"]["assets_pynapple"]
    extra_assets = ["beakTip_angle_rgb.npz", "beakTip_speed_troughs.npz"]
    labels_tsv = "Trial_data_labels.tsv"

    for asset in pynapple_assets + extra_assets:
        src = moll_src / asset
        dst_file = dest / asset
        if dst_file.exists() or not src.exists():
            continue
        shutil.copy2(src, dst_file)
        if verbose:
            print(f"  copied {asset}")

    labels_src = moll_src / labels_tsv
    labels_dst = dest / labels_tsv
    if labels_src.exists() and not labels_dst.exists():
        shutil.copy2(labels_src, labels_dst)
        if verbose:
            print(f"  copied {labels_tsv}")

    # Write configs
    write_example_configs("moll2025", dest)

    # Read trial epochs from pynapple trials.npz
    trials_npz = dest / "trials.npz"
    trials_obj = nap.load_file(str(trials_npz))
    if isinstance(trials_obj, nap.IntervalSet):
        trials_ep = trials_obj
    elif isinstance(trials_obj, dict):
        trials_ep = next(
            (v for v in trials_obj.values() if isinstance(v, nap.IntervalSet)),
            None,
        )
    else:
        trials_ep = None
    if trials_ep is None:
        raise ValueError(f"No IntervalSet found in {trials_npz}")

    # Moll2025: trial 1 = recording 115, trial 2 = recording 41
    media_per_trial = [
        {
            "video_cam-1": "2024-12-17_115_Crow1-cam-1.mp4",
            "pose_cam-1": "2024-12-17_115_Crow1-cam-1DLC.csv",
        },
        {
            "video_cam-1": "2024-12-18_041_Crow1-cam-1.mp4",
            "pose_cam-1": "2024-12-18_041_Crow1-cam-1DLC.csv",
        },
    ]

    fps = 200.0

    from datetime import datetime
    from uuid import uuid4
    from dateutil.tz import tzlocal

    nwbfile = pynwb.NWBFile(
        session_description="Moll2025 pynapple — alignment to original media.",
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
    )

    nwbfile.add_trial_column(name="trial", description="Trial number")
    nwbfile.add_trial_column(name="video_cam-1", description="video filename")
    nwbfile.add_trial_column(name="pose_cam-1", description="pose filename")
    for i in range(len(trials_ep)):
        trial_id = i + 1
        media = media_per_trial[i] if i < len(media_per_trial) else {}
        nwbfile.add_trial(
            start_time=float(trials_ep.start[i]),
            stop_time=float(trials_ep.end[i]),
            trial=trial_id,
            **{
                "video_cam-1": media.get("video_cam-1", ""),
                "pose_cam-1": media.get("pose_cam-1", ""),
            },
        )

    # Per-trial ImageSeries — each trial has its own video file
    nwbfile.create_device(name="cam-1", description="video device cam-1")
    video_files = [media_per_trial[i]["video_cam-1"] for i in range(len(trials_ep))]
    n_frames_per_trial = [
        int((float(trials_ep.end[i]) - float(trials_ep.start[i])) * fps)
        for i in range(len(trials_ep))
    ]
    timestamps_parts = []
    starting_frames = []
    frame_count = 0
    for i in range(len(trials_ep)):
        t0 = float(trials_ep.start[i])
        n = n_frames_per_trial[i]
        ts = t0 + np.arange(n) / fps
        timestamps_parts.append(ts)
        starting_frames.append(frame_count)
        frame_count += n

    nwbfile.add_acquisition(
        ImageSeries(
            name="video_cam-1",
            description="video from cam-1",
            external_file=video_files,
            format="external",
            starting_frame=np.array(starting_frames, dtype=np.int32),
            timestamps=np.concatenate(timestamps_parts),
        )
    )

    nwb_path = dest / ".ethograph" / "alignment.nwb"
    nwb_path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)

    if verbose:
        print(f"Moll2025 Pynapple ready at {dest}")
        print(f"  {len(trials_ep)} trials, alignment: {nwb_path}")
        print(f"  video/pose from: {moll_src}")

    return dest
