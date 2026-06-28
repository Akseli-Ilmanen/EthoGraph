"""Skeleton renderers for Movement napari plugin."""

from ethograph.skeleton.renderers.base import BaseRenderer
from ethograph.skeleton.renderers.precomputed import (
    PrecomputedRenderer,
)

__all__ = ["BaseRenderer", "PrecomputedRenderer"]
