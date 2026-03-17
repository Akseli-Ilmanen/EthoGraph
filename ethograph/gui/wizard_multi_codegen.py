"""Generate executable Python code for temporal alignment setup."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ethograph.utils.validation import (
    VIDEO_EXTENSIONS,
    AUDIO_EXTENSIONS,
    POSE_EXTENSIONS,
    EPHYS_EXTENSIONS,
)

if TYPE_CHECKING:
    from ethograph.gui.wizard_overview import ModalityConfig, WizardState
    from ethograph.gui.wizard_media_files import FilePattern


def _infer_value_pattern(values: list[str]) -> str:
    if not values:
        return r'[^/._-]+'

    # All numeric
    if all(v.isdigit() for v in values):
        lengths = {len(v) for v in values}
        return rf'\d{{{lengths.pop()}}}' if len(lengths) == 1 else r'\d+'

    # Detect prefix + numeric suffix (very common)
    m = [re.match(r'^([A-Za-z_]+)(\d+)$', v) for v in values]
    if all(m):
        prefixes = {x.group(1) for x in m}
        nums = [x.group(2) for x in m]

        if len(prefixes) == 1:
            prefix = re.escape(prefixes.pop())
            lengths = {len(n) for n in nums}

            if len(lengths) == 1:
                return rf'{prefix}\d{{{lengths.pop()}}}'
            return rf'{prefix}\d+'

    # Detect shared prefix
    prefix = os.path.commonprefix(values)
    if prefix and len(prefix) > 1:
        return rf'{re.escape(prefix)}[^/._-]+'

    # Same length fallback
    lengths = {len(v) for v in values if v}
    if len(lengths) == 1:
        return rf'[^/._-]{{{lengths.pop()}}}'

    # Small enumerations
    if len(values) <= 10:
        return '|'.join(re.escape(v) for v in values)

    return r'[^/._-]+'

TOKEN_PATTERN = r'[^/\.]+'

def _segments_to_regex(pattern: FilePattern) -> str | None:
    from ethograph.gui.wizard_media_files import FOLDER_POSITION
    from ethograph.gui.wizard_media_files import _tokenize, _find_token_spans

    has_roles = any(
        seg.varying and seg.role and seg.role != "ignore"
        for seg in pattern.segments
        if seg.position != FOLDER_POSITION
    )

    if not has_roles or not pattern.files:
        return None

    ref_file = pattern.files[0].stem

    tokens = _tokenize(ref_file, pattern.tokenize_mode)
    spans = _find_token_spans(ref_file, tokens)

    regex_parts = []
    prev_end = 0

    for idx, seg in enumerate(pattern.segments):

        if seg.position == FOLDER_POSITION:
            continue

        # Insert delimiter before token
        if idx < len(spans) and prev_end < spans[idx][0]:
            delim = ref_file[prev_end:spans[idx][0]]
            regex_parts.append(re.escape(delim))

        if idx < len(spans):
            prev_end = spans[idx][1]

        # Build token regex
        if seg.varying and seg.role and seg.role != "ignore":

            value_pattern = _infer_value_pattern(seg.values)
            regex_parts.append(f"(?P<{seg.role}>{value_pattern})")

        elif not seg.varying:

            regex_parts.append(re.escape(seg.text))

        else:

            regex_parts.append(TOKEN_PATTERN)

    # Anchor regex to full filename
    return "^" + "".join(regex_parts) + "$"



def generate_alignment_code(state: WizardState) -> str:
    """Generate executable Python code that reproduces the user's alignment setup.
    
    Parameters
    ----------
    state
        Complete wizard state with modality configs and trial table
        
    Returns
    -------
    Python code as a string
    """
    lines = []
    
    # ─── Imports ───
    lines.append("# conda activate ethograph")
    lines.append("import re")
    lines.append("import natsort")
    lines.append("import numpy as np")
    lines.append("import pandas as pd")
    lines.append("import xarray as xr")
    lines.append("from pathlib import Path")
    lines.append("import ethograph as eto")
    lines.append("from movement.io import load_poses")
    lines.append("from movement.kinematics import compute_velocity, compute_speed")
    lines.append("")
    lines.append("# ─── 1. Find media files ───")
    lines.append("# Regex patterns may not always work directly, update for your needs.")
    lines.append("")
    
    # ─── File finding logic ───
    enabled_modalities = []
    for name in ["video", "pose", "audio", "ephys"]:
        cfg: ModalityConfig = getattr(state, name)
        if cfg.enabled:
            enabled_modalities.append((name, cfg))
    
    

    
    for name, cfg in enabled_modalities:
        if cfg.file_mode == "single":
            lines.append(f'{name}_file = Path("{_escape_path(cfg.single_file_path)}")')
            lines.append("")
        elif cfg.folder_path:
            lines.append(f'{name}_folder = Path("{_escape_path(cfg.folder_path)}")')
            
            # Detect which extensions are actually present
            if cfg.pattern and cfg.pattern.files:
                found_extensions = sorted(set(f.suffix.lower() for f in cfg.pattern.files if f.suffix))
            else:
                # Fallback to showing first 3 standard extensions
                found_extensions = _get_extension_list(name)[:3]
            
            # Show actual glob patterns based on what was found
            if cfg.nested_subfolders:
                if len(found_extensions) == 1:
                    ext = found_extensions[0]
                    lines.append(f'{name}_files = natsort.natsorted({name}_folder.glob("**/*{ext}"))')
                else:
                    patterns = ", ".join(f'"**/*{ext}"' for ext in found_extensions)
                    lines.append(f'{name}_files = natsort.natsorted(p for pattern in [{patterns}] for p in {name}_folder.glob(pattern))')
            else:
                if len(found_extensions) == 1:
                    ext = found_extensions[0]
                    lines.append(f'{name}_files = natsort.natsorted({name}_folder.glob("*{ext}"))')
                else:
                    patterns = ", ".join(f'"*{ext}"' for ext in found_extensions)
                    lines.append(f'{name}_files = natsort.natsorted(p for pattern in [{patterns}] for p in {name}_folder.glob(pattern))')
            
            # Auto-generate regex pattern from role assignments
            if cfg.pattern and cfg.pattern.segments:
                regex_pattern = _segments_to_regex(cfg.pattern)
                if regex_pattern:
                    lines.append("")
                    # Check if pattern has trial role (alignment indicator)
                    has_trial_role = any(seg.role == "trial" for seg in cfg.pattern.segments)
                    
                    if has_trial_role and cfg.file_mode == "multi_regular":
                        # Aligned mode: build by_trial dict
                        lines.append(f'{name}_pattern = re.compile(r"{_escape_regex(regex_pattern)}")')
                        lines.append(f'{name}_by_trial = {{}}')
                        lines.append(f'for file_path in {name}_files:')
                        lines.append(f'    match = {name}_pattern.search(file_path.stem)')
                        lines.append(f'    if match:')
                        lines.append(f'        trial = int(match["trial"]) if match["trial"].isdigit() else match["trial"]')
                        lines.append(f'        {name}_by_trial.setdefault(trial, []).append(file_path)')
                    else:
                        # Continuous mode: build metadata list for non-trial roles
                        lines.append(f'{name}_pattern = re.compile(r"{_escape_regex(regex_pattern)}")')
                        lines.append(f'{name}_metadata = []')
                        lines.append(f'for f in {name}_files:')
                        lines.append(f'    match = {name}_pattern.search(f.stem)')
                        lines.append(f'    if match:')
                        lines.append(f'        groups = match.groupdict()')
                        lines.append(f'        groups = {{k: int(v) if v.isdigit() else v for k, v in groups.items()}}')
                        lines.append(f'        {name}_metadata.append(groups)')
                    lines.append("")
            lines.append("")
    
    # ─── Session table ───
    lines.append("# ─── 2. Create session table ───")
    lines.append("")
    
    # Check if we have metadata with trial info
    has_trial_metadata = False
    trial_source = None
    for name, cfg in enabled_modalities:
        if cfg.pattern and cfg.pattern.segments:
            if any(seg.role == "trial" for seg in cfg.pattern.segments):
                has_trial_metadata = True
                trial_source = name
                break
    
    if state.trial_table_path:
        # User imported CSV/TSV - show import statement
        sep = "\\t" if state.trial_table_path.endswith(".tsv") else ","
        lines.append(f'# Load session table from imported file')
        lines.append(f'session_table = pd.read_csv("{_escape_path(state.trial_table_path)}", sep="{sep}")')
        lines.append("")
    elif state.trial_table is not None and len(state.trial_table) <= 5:
        # Small table - show full inline
        lines.append("session_table = pd.DataFrame({")
        for col in state.trial_table.columns:
            vals = state.trial_table[col].tolist()
            lines.append(f"    '{col}': {vals},")
        lines.append("})")
        lines.append("")
    elif state.trial_table is not None:
        # Large table without source path - show template
        n_trials = len(state.trial_table)
        lines.append(f"# {n_trials} trials - construct programmatically")
        lines.append("session_table = pd.DataFrame({")
        lines.append(f"    'trial': list(range(1, {n_trials + 1})),")
        if "start_time" in state.trial_table.columns:
            lines.append(f"    'start_time': ...,  # Your trial start times")
            lines.append(f"    'stop_time': ...,   # Your trial stop times")
        lines.append("})")
        lines.append("")
    elif has_trial_metadata:
        # Extract trials from metadata
        lines.append(f"# Extract unique trials from {trial_source} metadata")
        lines.append(f"unique_trials = sorted(set(m['trial'] for m in {trial_source}_metadata))")
        lines.append("session_table = pd.DataFrame({")
        lines.append("    'trial': unique_trials,")
        lines.append("})")
        lines.append("")
    else:
        # No trial table - infer from file count
        lines.append("# Infer number of trials from file count")
        file_source = None
        for name, cfg in enabled_modalities:
            if name in ["video", "pose", "audio"] and cfg.folder_path:
                file_source = name
                break
        
        if file_source:
            lines.append(f"n_trials = len({file_source}_files)")
            lines.append("session_table = pd.DataFrame({")
            lines.append("    'trial': list(range(1, n_trials + 1)),")
            lines.append("})")
        else:
            lines.append("# TODO: Specify number of trials")
            lines.append("n_trials = ...  # Set your trial count")
            lines.append("session_table = pd.DataFrame({")
            lines.append("    'trial': list(range(1, n_trials + 1)),")
            lines.append("})")
        lines.append("")
    
    # ─── Load datasets ───
    lines.append("# ─── 3. Load trial datasets ───")
    lines.append("")
    
    # Check if pose has trial metadata
    pose_cfg = state.pose if hasattr(state, 'pose') else None
    has_pose_trial_metadata = (
        pose_cfg and pose_cfg.enabled and pose_cfg.pattern and 
        any(seg.role == "trial" for seg in pose_cfg.pattern.segments)
    )
    
    if has_pose_trial_metadata:
        # Use metadata to loop over trials
        lines.append("# TODO: Replace with your actual dataset loading logic")
        lines.append("ds_list = []")
        lines.append("for trial in session_table['trial']:")
        lines.append("    # Find pose file(s) for this trial")
        lines.append("    trial_pose_files = [")
        lines.append("        pose_files[i] for i, m in enumerate(pose_metadata)")
        lines.append("        if m.get('trial') == trial")
        lines.append("    ]")
        lines.append("    if not trial_pose_files:")
        lines.append("        continue")
        lines.append("    ")
        lines.append("    # Load first camera/view (adjust if multi-camera)")
        lines.append("    ds = load_poses.from_dlc_file(trial_pose_files[0], fps=30)")
        lines.append('    ds["velocity"] = compute_velocity(ds.position)')
        lines.append('    ds["speed"] = compute_speed(ds.position)')
        lines.append('    ds.attrs["trial"] = trial')
        lines.append("    ds_list.append(ds)")
    else:
        # Fallback to enumeration
        lines.append("# TODO: Replace with your actual dataset loading logic")
        lines.append("ds_list = []")
        lines.append("for idx, file in enumerate(pose_files, start=1):")
        lines.append("    ds = load_poses.from_dlc_file(file, fps=30)")
        lines.append('    ds["velocity"] = compute_velocity(ds.position)')
        lines.append('    ds["speed"] = compute_speed(ds.position)')
        lines.append('    ds.attrs["trial"] = idx')
        lines.append("    ds_list.append(ds)")
    lines.append("")
    
    # ─── Create TrialTree ───
    lines.append("# ─── 4. Create TrialTree ───")
    lines.append("")
    if state.trial_table is not None:
        lines.append("dt = eto.from_datasets(ds_list, session_table=session_table)")
    else:
        lines.append("dt = eto.from_datasets(ds_list)")
    lines.append("# Inspect first trial of TrialTree")
    lines.append("dt.itrial(0)")
    lines.append("")
    
    # ─── Set media files ───
    lines.append("# ─── 5. Store media files in session table ───")
    lines.append("# Note: Session table stores FILENAMES only, not full paths")
    lines.append("")
    
    # Detect alignment mode per modality
    has_media = any(cfg.enabled and name in ["video", "pose", "audio"] 
                    for name in ["video", "pose", "audio"] 
                    for cfg in [getattr(state, name)])
    
    if has_media:
        # Separate aligned and continuous modalities
        aligned_modalities = []
        continuous_modalities = []
        
        for name, cfg in enabled_modalities:
            if name not in ["video", "pose", "audio"]:
                continue
            if cfg.file_mode == "multi_regular":
                aligned_modalities.append((name, cfg))
            elif cfg.file_mode == "multi_irregular":
                continuous_modalities.append((name, cfg))
        
        if aligned_modalities:
            lines.append("# Aligned to trials: build filenames from by_trial dicts")
            lines.append("session = dt.session if dt.session else xr.Dataset()")
            lines.append("")
            
            # Build all filenames lists in one loop over trials
            for name, cfg in aligned_modalities:
                lines.append(f'{name}_filenames = []')
            
            lines.append('for trial in session_table["trial"]:')
            for name, cfg in aligned_modalities:
                device_dim = "cameras" if name in ["video", "pose"] else "mics"
                lines.append(f'    {name}_filenames.append([filename.name for filename in {name}_by_trial.get(trial, [])])')
            lines.append("")
            
            # Now create DataArrays with proper camera names
            for name, cfg in aligned_modalities:
                device_dim = "cameras" if name in ["video", "pose"] else "mics"
                device_names = state.camera_names if name in ["video", "pose"] else state.mic_names
                
                lines.append(f'session["{name}"] = xr.DataArray(')
                lines.append(f'    {name}_filenames,')
                lines.append(f'    dims=["trial", "{device_dim}"],')
                lines.append(f'    coords={{"trial": session_table["trial"], "{device_dim}": {device_names}}}')
                lines.append(f')')
            lines.append("")
        
        if continuous_modalities:
            if not aligned_modalities:
                lines.append("session = dt.session if dt.session else xr.Dataset()")
                lines.append("")
            
            lines.append("# Continuous session: one file per device covering entire session")
            for name, cfg in continuous_modalities:
                device_dim = "cameras" if name in ["video", "pose"] else "mics"
                device_names = state.camera_names if name in ["video", "pose"] else state.mic_names
                
                lines.append(f"# {name.capitalize()}: continuous session (per_trial=False)")
                if device_names:
                    # Map devices to files in order
                    lines.append(f'{name}_filenames = dict()')
                    lines.append(f'for i, device in enumerate({device_names}):')
                    lines.append(f'    if i < len({name}_files):')
                    lines.append(f'        {name}_filenames[device] = {name}_files[i].name')
                    lines.append(f'session["{name}"] = xr.DataArray(')
                    lines.append(f'    [{name}_filenames.get(c) for c in {device_names}],')
                    lines.append(f'    dims=["{device_dim}"],')
                    lines.append(f'    coords={{"{device_dim}": {device_names}}}')
                    lines.append(f')')
                else:
                    lines.append(f'{name}_filename = {name}_files[0].name if {name}_files else None')
                    lines.append(f'session["{name}"] = xr.DataArray({name}_filename)')
                lines.append("")
        
        lines.append("dt['session'] = xr.DataTree(session)")
        lines.append("")
    
    # ─── Stream offsets ───
    lines.append("# ─── 6. Set stream offsets (temporal alignment) ───")
    lines.append("")
    
    # Handle per-modality offsets
    has_offsets = False
    for name, cfg in enabled_modalities:
        if cfg.file_mode == "multi_irregular":  # Continuous mode
            if cfg.offset_constant_across_devices:
                if cfg.constant_offset != 0.0:
                    has_offsets = True
                    lines.append(f'dt.set_stream_offset("{name}", {cfg.constant_offset})')
            else:
                # Per-device offsets stored as session variables
                if cfg.device_offsets:
                    has_offsets = True
                    lines.append(f"# Per-device offsets for {name}")
                    for device, offset in sorted(cfg.device_offsets.items()):
                        if offset != 0.0:
                            lines.append(f'session = dt.session if dt.session else xr.Dataset()')
                            lines.append(f'session["offset_{name}_{device}"] = xr.DataArray({offset}, attrs={{"units": "seconds"}})')
                            lines.append(f'dt["session"] = xr.DataTree(session)')
        elif cfg.file_mode == "multi_regular":  # Aligned mode
            if cfg.constant_offset != 0.0:
                has_offsets = True
                lines.append(f'dt.set_stream_offset("{name}", {cfg.constant_offset})')
    
    if not has_offsets:
        lines.append("# No stream offsets configured")
    
    lines.append("")
    
    # ─── Save ───
    if state.output_path:
        lines.append("# ─── 7. Save to NetCDF ───")
        lines.append("")
        lines.append(f'dt.save("{_escape_path(state.output_path)}")')
    
    return "\n".join(lines)


def _escape_path(path: str) -> str:
    """Escape backslashes for Python string literals."""
    return path.replace("\\", "\\\\")


def _escape_regex(pattern: str) -> str:
    """Escape only quotes in regex patterns for raw Python string literals.
    
    Since we use raw strings r"...", backslashes are already literal.
    We only need to escape double quotes to avoid breaking the string.
    """
    return pattern.replace('"', '\\"')


def _get_extension_list(modality: str) -> list[str]:
    """Return list of file extensions for a modality."""
    if modality == "video":
        return sorted(VIDEO_EXTENSIONS)
    elif modality == "audio":
        return sorted(AUDIO_EXTENSIONS)
    elif modality == "pose":
        return sorted(POSE_EXTENSIONS)
    elif modality == "ephys":
        return sorted(EPHYS_EXTENSIONS)
    return []

