"""Path utilities with zero internal dependencies (stdlib only)."""

import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

SETTINGS_DIR = ".ethograph"


def get_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up from *start* until ``pyproject.toml`` is found.

    Parameters
    ----------
    start : Path, optional
        Directory to start searching from.  Defaults to the current
        working directory.

    Returns
    -------
    Path
        Absolute path to the project root.

    Raises
    ------
    FileNotFoundError
        If no ``pyproject.toml`` is found in any ancestor directory.

    Examples
    --------
    >>> import ethograph as eto
    >>> eto.get_project_root()
    PosixPath('/home/user/code/ethograph')
    """
    if start is not None:
        path = start.resolve()
    else:
        path = Path.cwd().resolve()
    for parent in [path] + list(path.parents):
        if (parent / "pyproject.toml").exists():
            if parent.parent.name != "deps":
                return parent
            continue
    fallback = Path(__file__).resolve()
    for parent in fallback.parents:
        if (parent / "pyproject.toml").exists():
            if parent.parent.name != "deps":
                return parent
            continue
    raise FileNotFoundError(f"Could not find project root starting from {path}")


def extract_pattern_groups(
    filenames: list[str | Path], pattern: str, convert_numeric: bool = True
) -> list[dict[str, str | int]]:
    """Extract named groups from filenames using a regex pattern.

    Parameters
    ----------
    filenames
        List of file paths
    pattern
        Regex pattern with named groups, e.g.:
        r'(?P<camera>cam[12])_trial(?P<trial>\\d+)\\.mp4'
    convert_numeric
        If True, automatically convert purely numeric strings to integers
        (e.g., "001" -> 1, "100" -> 100)

    Returns
    -------
    List of dicts mapping group names to extracted values (str or int)

    Examples
    --------
    >>> files = ["cam1_trial001.mp4", "cam2_trial001.mp4"]
    >>> pattern = r"(?P<camera>cam[12])_trial(?P<trial>\\d+)\\.mp4"
    >>> extract_pattern_groups(files, pattern, convert_numeric=True)
    [{'camera': 'cam1', 'trial': 1}, {'camera': 'cam2', 'trial': 1}]

    >>> extract_pattern_groups(files, pattern, convert_numeric=False)
    [{'camera': 'cam1', 'trial': '001'}, {'camera': 'cam2', 'trial': '001'}]
    """
    regex = re.compile(pattern)
    results = []
    for f in filenames:
        fname = Path(f).name
        match = regex.search(fname)
        if match:
            groups = match.groupdict()
            if convert_numeric:
                groups = {k: int(v) if v.isdigit() else v for k, v in groups.items()}
            results.append(groups)
    return results


def check_paths_exist(nc_paths):
    missing_paths = [p for p in nc_paths if not os.path.exists(p)]
    if missing_paths:
        print("Error: The following test_nc_paths do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        exit(1)


def find_config(name: str, data_dir: Path | str | None = None) -> Path | None:
    """Find a config file by walking up from *data_dir*, then falling back to global.

    Search order:
    1. Walk up from *data_dir* looking for ``.ethograph/{name}`` in each ancestor.
       This lets a shared ``.ethograph/`` in a parent directory serve multiple
       sessions, while per-session overrides are found first.
    2. ``~/.ethograph/{name}``  (global user default)

    Parameters
    ----------
    name
        Config filename, e.g. ``"mapping.txt"`` or ``"arena.yaml"``.
    data_dir
        Directory of the loaded data file.  Pass ``None`` to skip the
        walk-up search (e.g. at application startup before any file is loaded).

    Returns
    -------
    First existing path, or ``None`` if none are found.

    Examples
    --------
    >>> find_config("mapping.txt", "/data/project/session_01")
    PosixPath('/data/project/.ethograph/mapping.txt')
    """
    if data_dir is not None:
        d = Path(data_dir).resolve()
        for parent in [d] + list(d.parents):
            candidate = parent / SETTINGS_DIR / name
            if candidate.exists():
                return candidate

    global_candidate = Path.home() / SETTINGS_DIR / name
    if global_candidate.exists():
        return global_candidate

    return None


def default_config_dir(data_dir: Path | str | None = None) -> Path:
    """Return the ``.ethograph/`` directory where new configs should be written.

    Uses ``data_dir/.ethograph/`` when a data directory is known, otherwise
    falls back to ``~/.ethograph/``.
    """
    if data_dir is not None:
        return Path(data_dir) / SETTINGS_DIR
    return Path.home() / SETTINGS_DIR


def find_mapping_file(data_dir: Path | str | None = None) -> Path | None:
    """Find mapping.txt. Convenience wrapper around :func:`find_config`."""
    return find_config("mapping.txt", data_dir)


def find_nwb_file(data_dir: Path | str | None = None) -> Path | None:
    """Find alignment.nwb in ``.ethograph/`` relative to *data_dir*.

    Search order:
    1. ``<data_dir>/.ethograph/alignment.nwb``
    2. Any ``.nwb`` file in ``<data_dir>/.ethograph/``
    3. Walk up parent directories looking for ``.ethograph/alignment.nwb``

    Returns
    -------
    Path or None
    """
    if data_dir is None:
        return None
    d = Path(data_dir).resolve()

    # Check immediate .ethograph directory
    ethograph_dir = d / SETTINGS_DIR
    if ethograph_dir.is_dir():
        candidate = ethograph_dir / "alignment.nwb"
        if candidate.exists():
            return candidate
        nwb_files = list(ethograph_dir.glob("*.nwb"))
        if nwb_files:
            return nwb_files[0]

    # Check one parent up only
    parent = d.parent
    if parent != d:
        ethograph_dir = parent / SETTINGS_DIR
        if ethograph_dir.is_dir():
            candidate = ethograph_dir / "alignment.nwb"
            if candidate.exists():
                return candidate
            nwb_files = list(ethograph_dir.glob("*.nwb"))
            if nwb_files:
                return nwb_files[0]

    return None


def extract_trial_info_from_filename(path):
    """
    Extract session_date, trial_num, and bird from a DLC filename.
    Expected filename format: YYYY-MM-DD_NNN_Bird_...
    """
    filename = os.path.basename(path)
    parts = filename.split("_")
    if len(parts) >= 3:
        session_date = parts[0]
        trial_num = int(parts[1])
        bird = parts[2]
        return session_date, trial_num, bird
    else:
        raise ValueError(f"Filename format not recognized: {filename}")


def auto_git_commit(label_path: Path) -> None:
    """Auto-commit a label file if it lives inside a git repository."""
    file_dir = str(label_path.parent)
    try:
        root_result = subprocess.run(
            ["git", "-C", file_dir, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise ValueError("git is not installed or not found on PATH.")

    if root_result.returncode != 0:
        raise ValueError(
            f"Remote backup folder is not inside a git repository: {file_dir}\n"
            f"git said: {root_result.stderr.strip()}\n"
            "Run 'git init' in an ancestor folder first, or use 'Save with timestamp' mode."
        )

    repo_root = Path(root_result.stdout.strip())
    rel_path = label_path.relative_to(repo_root)

    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "add", str(rel_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        commit_result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "commit",
                "-m",
                f"Labels updated: {label_path.name}",
            ],
            capture_output=True,
            text=True,
        )
        if commit_result.returncode != 0:
            msg = commit_result.stderr.strip() or commit_result.stdout.strip()
            if "nothing to commit" in msg:
                logger.info("git: nothing to commit for %s", label_path.name)
                return
            raise subprocess.CalledProcessError(
                commit_result.returncode,
                "git commit",
                output=commit_result.stdout,
                stderr=commit_result.stderr,
            )
        subprocess.run(
            ["git", "-C", str(repo_root), "push"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Auto-committed and pushed %s to git", label_path.name)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"git commit/push failed: {e.stderr.strip() or e.stdout.strip() or str(e)}")
