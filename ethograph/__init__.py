"""ethograph"""

__version__ = "0.1.0"

from ethograph.utils.trialtree import SESSION_NODE,TrialTree
from ethograph.utils.io import (
    add_angle_rgb_to_ds,
    add_changepoints_to_ds,
    downsample_trialtree,
    get_project_root,
    dataset_to_basic_trialtree,
)
from ethograph.utils.xr_utils import get_time_coord, sel_valid, trees_to_df



def open(path: str) -> TrialTree:
    """Open a TrialTree from a NetCDF file. Shorthand for ``TrialTree.open``."""
    return TrialTree.open(path)


def from_datasets(datasets: list, session_table=None) -> TrialTree:
    """Create a TrialTree from a list of datasets. Shorthand for ``TrialTree.from_datasets``."""
    return TrialTree.from_datasets(datasets, session_table=session_table)
