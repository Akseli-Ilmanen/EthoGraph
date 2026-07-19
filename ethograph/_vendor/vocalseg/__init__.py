"""Vendored copy of Tim Sainburg's vocalization-segmentation (``vocalseg``).

Source: https://github.com/timsainb/vocalization-segmentation (MIT, 2019).

Only the two segmentation entry points ethograph uses are vendored
(``dynamic_threshold_segmentation`` and ``continuity_segmentation``) plus their
shared spectrogram helpers. The upstream matplotlib/seaborn plotting functions
that ethograph never calls were dropped to avoid extra dependencies. Imports
were rewritten to be package-relative. See the sibling LICENSE file.
"""

import numpy as np

# Upstream vocalseg predates NumPy 2.0, which removed np.product.
if not hasattr(np, "product"):
    np.product = np.prod

from .continuity_filtering import continuity_segmentation
from .dynamic_thresholding import dynamic_threshold_segmentation

__all__ = ["continuity_segmentation", "dynamic_threshold_segmentation"]
