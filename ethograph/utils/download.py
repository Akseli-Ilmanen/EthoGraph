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
