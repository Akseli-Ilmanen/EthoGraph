#
# Copyright 2020-present by A. Mathis Group and contributors. All rights reserved.
#
# This project and all its files are licensed under GNU AGPLv3 or later version.
# A copy is included in LICENSE.APGPL.
#
# Vendored from DLC2Action — see NOTICE.md
# ruff: noqa
"""Vendored DLC2Action: the model and loss layer, and upstream's own configs.

This package is a partial copy of `DLC2Action
<https://github.com/amathislab/DLC2Action>`_ — the parts that define
*architectures*, *losses* and their default hyperparameters. Upstream's
toolbox layer (project, data stores, feature extraction, SSL, metrics,
transformers) is deliberately not vendored: this project has its own data
model, training loop and metrics.

Upstream's own ``__init__`` imports that toolbox, so it is replaced by this
file. Nothing else in ``model/`` or ``loss/`` depends on it.

The adapters that put these under this project's registry contract live in
:mod:`ethograph.segment.models.vendored`; the loss adapter in
:mod:`ethograph.segment.losses`.
"""

from ethograph.segment.dlc2action.version import VERSION, __version__

__all__ = ["VERSION", "__version__"]
