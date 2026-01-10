"""Custom code for importing data from lab-internal matlab codes to python"""

import re
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union

from moveseg.utils.io import TrialTree


def extract_csv_file_info(file_name: str, session_number: str) -> Tuple[str, str, str, str, str]:
    """
    Extracts session_date, trial, subject_id, and generates folder name and mp4 file name.

    Args:
        file_name: A string with the format 'YYYY-MM-DD_trial_SubjectID_DLC_3D[.csv|.m]'
        session_number: Suffix after date: 'XX' in 'YYYYMMDD_XX'

    Returns:
        session_date: The date in 'YYYY-MM-DD' format
        trial: The trial number or identifier after the date
        subject_id: The subject ID in the file
        folder_name: The folder name in 'YYYYMMDD_XX_SubjectID' format
        mp4_file_name: The MP4 filename in 'YYYY-MM-DD_trial_SubjectID-cam-1.mp4' format
    """
    pattern = r'(\d{4}-\d{2}-\d{2})_(\d+)_([A-Za-z]+)_DLC_3D'
    match = re.search(pattern, file_name)
    if not match:
        raise ValueError('Filename does not match the expected format.')
    session_date, trial, subject_id = match.groups()
    folder_name = f"{session_date.replace('-', '')}_{session_number}_{subject_id}\\"
    mp4_file_name = f"{session_date}_{str(trial).zfill(3)}_{subject_id}-cam-1.mp4"
    return session_date, trial, subject_id, folder_name, mp4_file_name


def get_all_trials_path_info(all_trials_path):
    """
    Extract subject_id, session_date, session_number, and dataset_name from a path string.
    Args:
        all_trials_path (str): Path string to parse.
    Returns:
        subject_id (str), session_date (str), session_number (str), dataset_name (str)
    """
    path = all_trials_path.replace('\\', '/')

    id_match = re.search(r'id-([^/\\]+)', path)
    subject_id = id_match.group(1) if id_match else ''

    date_sess_match = re.search(r'date-(\d{8})_(\d{2})', path)
    if date_sess_match:
        session_date = date_sess_match.group(1)
        session_number = date_sess_match.group(2)
    else:
        session_date = ''
        session_number = ''

    dataset_name = f'{session_date}-{session_number}_{subject_id}' 

    return subject_id, session_date, session_number, dataset_name






from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import xarray as xr
from scipy.io import loadmat


def update_nc_with_matlab_trials(
    nc_path: str | Path,
    matlab_path: str | Path,
    output_path: Optional[str | Path] = None
) -> xr.DataTree:
    """
    Update NetCDF trial data with information from MATLAB AllTrials structure.
    
    Parameters
    ----------
    nc_path : str | Path
        Path to the NetCDF file containing trial data
    matlab_path : str | Path
        Path to the MATLAB file containing AllTrials structure
    output_path : Optional[str | Path]
        Path to save updated NetCDF. If None, overwrites input file
        
    Returns
    -------
    xr.DataTree
        Updated DataTree with MATLAB trial information
        
    Raises
    ------
    FileNotFoundError
        If input files don't exist
    ValueError
        If required data structures are missing
    """
    nc_path = Path(nc_path)
    matlab_path = Path(matlab_path)
    
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
    if not matlab_path.exists():
        raise FileNotFoundError(f"MATLAB file not found: {matlab_path}")
    
    # Load DataTree
    dt = TrialTree.load(nc_path)
    
    # Load MATLAB data
    mat_data = loadmat(
        matlab_path, 
        squeeze_me=True, 
        struct_as_record=False
    )
    
    if 'AllTrials' not in mat_data:
        raise ValueError(f"'AllTrials' structure not found in {matlab_path}")
    
    all_trials = mat_data['AllTrials']
    
    # Create lookup for efficient trial matching
    matlab_trials_dict = {
        trial.trial_num: trial 
        for trial in all_trials
    }
    
    # Update each dataset in the tree
    for node_name, node in dt.children.items():
        if node.ds is None or 'trial' not in node.ds.attrs:
            continue
            
        trial_num = node.ds.attrs['trial']
        
        if trial_num not in matlab_trials_dict:
            continue
            
        trial = matlab_trials_dict[trial_num]
        ds = node.ds
        
        # Update attributes
        if hasattr(trial.info, 'poscat'):
            ds.attrs['poscat'] = trial.info.poscat
        if hasattr(trial.info, 'num_pellets'):
            ds.attrs['num_pellets'] = trial.info.num_pellets
        
        # Process boundary events
        event_data = _extract_boundary_events(trial.info)
        if event_data is not None:
            ds['boundary_events'] = ('events', event_data)
            ds = ds.assign_coords(
                events=('events', ['disp_out', 'disp_in', 'box_in', 'box_out'])
            )
        
        # Initialize labels
        n_time = len(ds.coords['time'])
        n_individuals = len(ds.coords['individuals'])
        ds['labels'] = (
            ('time', 'individuals'), 
            np.zeros((n_time, n_individuals))
        )
        
        # Update labels if available
        if hasattr(trial, 'motif_infos') and hasattr(trial.motif_infos, 'beakTip'):
            if 'bird' in ds.attrs:
                bird = ds.attrs['bird']
                labels = trial.motif_infos.beakTip.labels
                ds['labels'].loc[dict(individuals=bird)] = labels
    
    # Save updated tree
    if output_path is None:
        output_path = nc_path
    else:
        output_path = Path(output_path)
        
    dt.to_netcdf(output_path)
    
    return dt


def _extract_boundary_events(trial_info) -> Optional[np.ndarray]:
    """
    Extract boundary events from trial info structure.
    
    Parameters
    ----------
    trial_info : object
        MATLAB trial info structure
        
    Returns
    -------
    Optional[np.ndarray]
        Array of boundary events or None if extraction fails
    """
    event_data = np.zeros(4)
    
    if not hasattr(trial_info, 'stick_in_out_disp'):
        return None
        
    disp_out_in = trial_info.stick_in_out_disp
    
    # Extract disp_out and disp_in
    for j, val in enumerate([0, 1]):
        try:
            event_data[j] = int(disp_out_in[val]) - 1
        except (IndexError, TypeError, ValueError):
            event_data[j] = np.nan
    
    # Extract box_in and box_out
    if hasattr(trial_info, 'first_in_last_out'):
        for j, idx in enumerate([0, 1], start=2):
            try:
                event_data[j] = int(trial_info.first_in_last_out[idx-2]) - 1
            except (IndexError, TypeError, ValueError):
                event_data[j] = np.nan
    else:
        event_data[2:] = np.nan
    
    return event_data