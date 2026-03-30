<<<<<<< HEAD
"""ethograph"""

__version__ = "0.1.0"

from ethograph.utils.trialtree import SESSION_NODE, TrialTree
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
=======
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ethograph")
except PackageNotFoundError:
    # package is not installed
    pass

    
from ethograph.io.trialtree import TrialTree
from ethograph.io.dataset import (
    add_angle_rgb_to_ds,
    add_changepoints_to_ds,
    downsample_trialtree,
    dataset_to_basic_trialtree,
)
from ethograph.utils.xr_utils import get_time_coord, sel_valid
from ethograph.utils.paths import get_project_root


def open(path: str) -> TrialTree:
    """Load a TrialTree from a saved NetCDF file.

    Shorthand for :meth:`TrialTree.open <ethograph.io.trialtree.TrialTree.open>`.

    Parameters
    ----------
    path : str or Path
        Path to a ``.nc`` file previously saved with ``dt.save()``.

    Returns
    -------
    TrialTree

    Examples
    --------
    >>> import ethograph as eto
    >>> dt = eto.open("experiment.nc")
    >>> dt.trials
    [1, 2, 3]
    >>> ds = dt.itrial(0)
    """
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    return TrialTree.open(path)


def from_datasets(datasets: list, session_table=None) -> TrialTree:
<<<<<<< HEAD
    """Create a TrialTree from a list of datasets. Shorthand for ``TrialTree.from_datasets``."""
=======
    """Build a TrialTree from a list of per-trial xarray Datasets.

    Shorthand for :meth:`TrialTree.from_datasets <ethograph.io.trialtree.TrialTree.from_datasets>`.

    Each dataset must have ``attrs["trial"]`` set to a unique trial
    identifier.

    Parameters
    ----------
    datasets : list[xarray.Dataset]
        One Dataset per trial.
    session_table : xarray.Dataset or pandas.DataFrame, optional
        Session-level metadata indexed by trial ID (e.g. start/stop times,
        condition labels).

    Returns
    -------
    TrialTree

    Examples
    --------
    >>> import xarray as xr, numpy as np, ethograph as eto
    >>> trials = []
    >>> for i in range(1, 4):
    ...     ds = xr.Dataset({"speed": xr.DataArray(np.random.rand(300), dims=["time"])})
    ...     ds.attrs["trial"] = i
    ...     trials.append(ds)
    >>> dt = eto.from_datasets(trials)
    >>> dt.trials
    [1, 2, 3]
    """
>>>>>>> bbdb95118885b151f0e39e30378a0ec171e43955
    return TrialTree.from_datasets(datasets, session_table=session_table)
