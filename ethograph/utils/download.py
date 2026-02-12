"""Download example datasets from GitHub releases."""

from pathlib import Path
from typing import Callable
from urllib.request import urlopen

_RELEASE_BASE = "https://github.com/Akseli-Ilmanen/EthoGraph/releases/download"

EXAMPLE_DATASETS = {
    "moll2025": {
        "release_tag": "moll2025",
        "assets": [
            "Trial_data.nc",
            "2024-12-17_115_Crow1-cam-1.mp4",
            "2024-12-17_115_Crow1-cam-1DLC.csv",
            "2024-12-17_115_Crow1-cam-2DLC.csv",
            "2024-12-17_115_Crow1_DLC_3D.csv",
            "2024-12-18_041_Crow1-cam-1.mp4",
            "2024-12-18_041_Crow1-cam-1DLC.csv",
            "2024-12-18_041_Crow1-cam-2DLC.csv",
            "2024-12-18_041_Crow1_DLC_3D.csv",
        ],
        "size_mb": 21,
    },
    "birdpark": {
        "release_tag": "birdpark",
        "assets": [
            "copExpBP08_trim.nc",
            "copExpBP08_trim.mp4",
            "copExpBP08_trim.wav",
        ],
        "size_mb": 76,
    },
    "philodoptera": {
        "release_tag": "philodoptera",
        "assets": [
            "philodoptera.nc",
            "philodoptera.mp4",
            "philodoptera.wav",
            "philodoptera.csv",
        ],
        "size_mb": 4,
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
    """Check whether all assets for a dataset are already present."""
    info = EXAMPLE_DATASETS.get(release_tag)
    if info is None:
        return False
    return all((Path(dest) / name).exists() for name in info["assets"])


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

    def _print_progress(count: int, name: str) -> None:
        total = len(info["assets"])
        if count < total:
            print(f"Downloading {name}... ({count}/{total})")
        else:
            print(f"  {name} ({count}/{total})")

    download_assets(
        release_tag=info["release_tag"],
        assets=info["assets"],
        dest=dest,
        on_progress=_print_progress if verbose else None,
    )
