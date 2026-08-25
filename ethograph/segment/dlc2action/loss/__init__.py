#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version.
# A copy is included in dlc2action/LICENSE.AGPL.
#
# Vendored from DLC2Action — see NOTICE.md
# ruff: noqa
"""Losses.

There is no dedicated loss class in `dlc2action`. Instead we use regular `torch.nn.Module` instances that take
prediction
and target as input and return loss value as output.
"""

from ethograph.segment.dlc2action.loss.mse import *
from ethograph.segment.dlc2action.loss.ms_tcn import *

__pdoc__ = {
    "ms_tcn.MS_TCN_Loss.dump_patches": False,
    "ms_tcn.MS_TCN_Loss.training": False,
    "mse.MSE.dump_patches": False,
    "mse.MSE.training": False,
    "asymmetric_loss": False,
}
