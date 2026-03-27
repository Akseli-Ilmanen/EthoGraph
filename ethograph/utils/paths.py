"""Path utilities with zero internal dependencies (stdlib only)."""

import glob
import json
import os
from pathlib import Path
import re
import ethograph as eto
from ethograph.utils.validation import (
    VIDEO_EXTENSIONS,
    AUDIO_EXTENSIONS,
    POSE_EXTENSIONS,
)


def find_media_files(folder: str | Path, extensions: set[str] | list[str], recursive: bool = False) -> list[Path]:
    """Find all files with given extensions in a folder.
    
    Parameters
    ----------
    folder
        Path to search
    extensions
        Set or list of file extensions to match (e.g., {'.mp4', '.avi'})
    recursive
        If True, search nested subdirectories
        
    Returns
    -------
    Sorted list of matching file paths
    """
    folder = Path(folder)
    pattern = "**/*" if recursive else "*"
    files = []
    for ext in extensions:
        files.extend(folder.glob(f"{pattern}{ext}"))
    return sorted(files)


def extract_pattern_groups(filenames: list[str | Path], pattern: str, convert_numeric: bool = True) -> list[dict[str, str | int]]:
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
    >>> files = ['cam1_trial001.mp4', 'cam2_trial001.mp4']
    >>> pattern = r'(?P<camera>cam[12])_trial(?P<trial>\\d+)\\.mp4'
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


def group_files_by(files: list[Path], key_func) -> dict:
    """Group files by a key extracted from each filename.
    
    Parameters
    ----------
    files
        List of file paths
    key_func
        Function that extracts a grouping key from a Path
        
    Returns
    -------
    Dict mapping keys to lists of file paths
    
    Examples
    --------
    >>> files = [Path('cam1_001.mp4'), Path('cam1_002.mp4'), Path('cam2_001.mp4')]
    >>> group_files_by(files, lambda p: p.stem.split('_')[0])
    {'cam1': [Path('cam1_001.mp4'), Path('cam1_002.mp4')], 
     'cam2': [Path('cam2_001.mp4')]}
    """
    groups: dict = {}
    for f in files:
        key = key_func(f)
        groups.setdefault(key, []).append(f)
    return groups



def check_paths_exist(nc_paths):
    missing_paths = [p for p in nc_paths if not os.path.exists(p)]
    if missing_paths:
        print("Error: The following test_nc_paths do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        exit(1)
 



def find_mapping_file(data_dir: Path | str | None = None) -> Path | None:
    """Find mapping.txt using priority hierarchy.

    Search order:
    1. ``data_dir/.ethograph/mapping.txt``  (local, next to loaded data)
    2. ``~/.ethograph/mapping.txt``          (global user config)
    3. ``project_root/configs/mapping.txt``  (project fallback)

    Parameters
    ----------
    data_dir
        Directory of the loaded data file.  Pass ``None`` to skip the local
        search (e.g. at application startup before any file is loaded).

    Returns
    -------
    First existing path, or ``None`` if none are found.
    """
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir) / ".ethograph" / "mapping.txt")
    candidates.append(Path.home() / ".ethograph" / "mapping.txt")
    try:
        candidates.append(eto.get_project_root() / "configs" / "mapping.txt")
    except FileNotFoundError:
        pass
    return next((p for p in candidates if p.exists()), None)


def gui_default_settings_path() -> Path:
    """Get the default path for gui_settings.yaml in the project root."""
    settings_path = eto.get_project_root() / "configs" / "gui_settings.yaml"
    settings_path.touch(exist_ok=True)
    return settings_path


def extract_trial_info_from_filename(path):
    """
    Extract session_date, trial_num, and bird from a DLC filename.
    Expected filename format: YYYY-MM-DD_NNN_Bird_...
    """
    filename = os.path.basename(path)
    parts = filename.split('_')
    if len(parts) >= 3:
        session_date = parts[0]
        trial_num = int(parts[1])
        bird = parts[2]
        return session_date, trial_num, bird
    else:
        raise ValueError(f"Filename format not recognized: {filename}")

def get_session_path(user: str, datatype: str, bird: str, session: str, data_folder_type: str):
    """
    Args:
        user (str): e.g. 'Akseli_right' or 'Alice_home'.
        datatype (str): Type of data (e.g., 'rawdata' or 'derivatives').
        bird (str): Name of the bird (e.g., 'Ivy', 'Poppy', or 'Freddy').
        session (str): Date of the session in 'YYYYMMDD_XX' format.
        data_folder_type (str): 'rigid_local', 'working_local', or 'working_backup'

    Returns:
        subject_folder (str): Path to the subject folder
        session_path (str): Path to the rawdata/derivatives session folder
        data_folder (str): Path to parent data folder
    """
    breakpoint()
    # Desktop path (Windows default, swap for Linux/mac if needed)
    desktop_path = os.path.join(os.environ.get("USERPROFILE", os.environ.get("HOME")), "Desktop")
    
    # Load user_paths.json
    with open(os.path.join(desktop_path, "user_paths.json"), "r") as f:
        paths = json.load(f)
        

    # Select the data folder
    if data_folder_type == "rigid_local":
        data_folder = paths[user]["rigid_local_data_folder"]
    elif data_folder_type == "working_local":
        data_folder = paths[user]["working_local_data_folder"]
    elif data_folder_type == "working_backup":
        data_folder = paths[user]["working_backup_data_folder"]
    else:
        raise ValueError("Unknown data folder type.")

    # Bird mapping
    if bird == "Ivy":
        sub_name = "sub-01_id-Ivy"
    elif bird == "Poppy":
        sub_name = "sub-02_id-Poppy"
    elif bird == "Freddy":
        sub_name = "sub-03_id-Freddy"
    else:
        raise ValueError("Unknown bird type.")

    # Subject folder
    subject_folder = os.path.join(data_folder, datatype, sub_name)
    print(f"Subject folder: {subject_folder}")

    # Find session folder
    matches = [d for d in os.listdir(subject_folder) if session in d]

    if len(matches) != 1:
        raise RuntimeError(
            "Likely causes:\n1) Multiple or no folders found containing the session date."
            "\n2) Paths wrong in Desktop/user_paths.json."
        )

    session_path = os.path.join(subject_folder, matches[0])

    return subject_folder, session_path, data_folder
