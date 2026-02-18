"""Download example datasets from GitHub releases."""

from pathlib import Path
from typing import Callable
from urllib.request import urlopen

_RELEASE_BASE = "https://github.com/Akseli-Ilmanen/EthoGraph/releases/download"

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
        "assets_gui": [
            "Trial_data.nc",
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
            "copExpBP08_trim.mp4",
            "copExpBP08_trim.wav",
        ],
        "assets_notebook": [
            "copExpBP08_trim.nc",
            "copExpBP08_trim.mp4",
            "copExpBP08_trim.wav",
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
}


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
) -> None:
    """High-level helper: download an example dataset by key.

    Parameters
    ----------
    key : str
        One of ``"moll2025"``, ``"birdpark"``, ``"philodoptera"``.
    dest : Path
        Directory to download into.
    verbose : bool
        Print progress to stdout.
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
