"""Download example datasets from GitHub releases."""

import logging
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

from ethograph.datasets import (
    DATASETS,
    DOWNLOAD_BASE,
    dataset_dir,
    get_gui_assets,
    get_notebook_assets,
    is_dataset_downloaded,
    resolve_dataset_paths,
)

logger = logging.getLogger(__name__)

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


def _register_tsv_windows(vbs_path: Path) -> None:
    """Register .tsv → Excel file association via the current-user registry."""
    import winreg

    cls = r"Software\Classes"
    vbs_str = str(vbs_path)

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\.tsv") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "TsvFile")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "Tab-Separated Values")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile\DefaultIcon") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "excel.exe,1")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile\shell") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "openexcel")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile\shell\openexcel") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, "Open TSV with Excel")
        winreg.SetValueEx(key, "FriendlyAppName", 0, winreg.REG_SZ, "Excel (Tab-Separated)")

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{cls}\TsvFile\shell\openexcel\command") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, f'wscript.exe "{vbs_str}" "%1"')


def _register_tsv_mac() -> None:
    """Set Excel as default app for .tsv on macOS via duti (if available)."""
    import subprocess

    try:
        subprocess.run(["duti", "-s", "com.microsoft.Excel", ".tsv", "all"], check=True)
    except FileNotFoundError:
        logger.debug("duti not installed — skipping .tsv association on macOS")


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
        try:
            _register_tsv_windows(vbs)
        except OSError:
            logger.debug("Could not write TSV registry keys", exc_info=True)
    elif sys.platform == "darwin":
        _register_tsv_mac()


def write_example_configs(dataset_key: str, dest: Path) -> None:
    """Write bundled config files into ``dest/.ethograph/``."""
    configs = DATASETS.get(dataset_key, {}).get("configs")
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


def build_alignment_nwb(key: str) -> None:
    """Create an alignment.nwb from the dataset's media mapping and NC file."""
    ds = DATASETS[key]
    media_rows = ds.get("media")
    if not media_rows:
        return
    dest = dataset_dir(key)
    nc_filename = ds.get("nc_filename")
    if not nc_filename:
        return
    nc_path = dest / nc_filename
    if not nc_path.exists():
        return

    import pandas as pd
    import ethograph as eto
    from ethograph.utils.nwb import build_nwb_from_trial_table

    dt = eto.open(str(nc_path))
    trials = dt.trials
    fps = dt.itrial(0).attrs.get("fps", None)

    rows = []
    for trial_id, media in zip(trials, media_rows):
        row = {"trial": trial_id}
        row.update(media)
        rows.append(row)

    trial_table = pd.DataFrame(rows)
    nwb_path = dest / ".ethograph" / "alignment.nwb"
    stream_rates: dict[str, float] = {}
    if fps is not None:
        stream_rates["video"] = float(fps)
        stream_rates["pose"] = float(fps)

    audio_cols = [c for c in trial_table.columns if c.startswith("audio_")]
    if audio_cols:
        first_audio_file = dest / str(trial_table[audio_cols[0]].iloc[0])
        if first_audio_file.exists():
            from ethograph.utils.audio import get_audio_sr
            sr = get_audio_sr(str(first_audio_file))
            if sr is not None:
                stream_rates["audio"] = float(sr)

    build_nwb_from_trial_table(trial_table, stream_rates=stream_rates, output_path=nwb_path)
    logger.info("Created alignment NWB: %s", nwb_path)


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
        One of the keys in ``DATASETS``.
    dest : Path
        Directory to download into.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    Path to ``mapping.txt`` if one was created, otherwise ``None``.
    """
    info = DATASETS[key]
    assets = get_notebook_assets(key)

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
        dest = DOWNLOAD_BASE / "BirdParkContinuous"

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    bp_src = dataset_dir("birdpark")

    for asset in get_gui_assets("birdpark"):
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
        dest = DOWNLOAD_BASE / "Moll2025_pynapple"

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    moll_src = dataset_dir("moll2025")
    if not moll_src.exists():
        raise FileNotFoundError(
            f"Moll2025 source not found: {moll_src}. "
            "Download moll2025 first via download_example_dataset('moll2025', ...)"
        )

    # Copy pynapple assets + extras
    pynapple_assets = DATASETS["moll2025"].get("assets_pynapple", [])
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
    media_per_trial = DATASETS["moll2025"]["media"]
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


# ---------------------------------------------------------------------------
# Backward-compat re-exports (used by external code / notebooks)
# ---------------------------------------------------------------------------

EXAMPLE_DATASETS = DATASETS
EXAMPLE_CONFIGS = {k: v["configs"] for k, v in DATASETS.items() if "configs" in v}

def is_downloaded(key: str, dest: Path) -> bool:  # noqa: ARG001
    """Deprecated — use ``is_dataset_downloaded(key)`` instead."""
    return is_dataset_downloaded(key)
