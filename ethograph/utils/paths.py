"""Path utilities with zero internal dependencies (stdlib only)."""

import json
import os
from pathlib import Path
from ethograph import get_project_root

def check_paths_exist(nc_paths):
    missing_paths = [p for p in nc_paths if not os.path.exists(p)]
    if missing_paths:
        print("Error: The following test_nc_paths do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        exit(1)
 



def gui_default_settings_path() -> Path:
    """Get the default path for gui_settings.yaml in the project root."""
    settings_path = get_project_root() / "configs" / "gui_settings.yaml"
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
