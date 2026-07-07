"""Skeleton visualization.

Provides the config layer (templates, NWB skeleton → config, validation) and
the ``PrecomputedRenderer``. Rendering to the GUI is handled by the pygfx pose
overlay (:mod:`ethograph.gui.pose_overlay`); this module no longer depends on
napari.
"""

from typing import Any

from ethograph.skeleton.config import (
    config_to_arrays,
    load_yaml_config,
    nwb_skeleton_to_config,
    save_yaml_config,
    validate_config,
)
from ethograph.skeleton.renderers import PrecomputedRenderer
from ethograph.skeleton.state import SkeletonState
from ethograph.skeleton.templates import TEMPLATES

__all__ = [
    "load_skeleton_config",
    "save_skeleton_config",
    "nwb_skeleton_to_config",
    "config_to_arrays",
    "validate_config",
    "TEMPLATES",
    "SkeletonState",
    "PrecomputedRenderer",
]


def load_skeleton_config(path: str) -> dict[str, Any]:
    """Load skeleton configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file

    Returns
    -------
    dict
        Skeleton configuration dictionary
    """
    return load_yaml_config(path)


def save_skeleton_config(config: dict[str, Any], path: str) -> None:
    """Save skeleton configuration to a YAML file.

    Parameters
    ----------
    config : dict
        Skeleton configuration dictionary
    path : str
        Path where the YAML file will be saved
    """
    save_yaml_config(config, path)
