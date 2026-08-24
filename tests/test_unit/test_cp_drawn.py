"""Superseded by ``test_changepoint_snap.py`` — this file can be deleted.

The functions it exercised (``_get_dataset_changepoint_indices``,
``extract_cp_times``) no longer exist: the lineplot draws its own changepoints
from ``XarrayLoader.select`` and a click snaps to ``XarrayLoader.get_cp_times``,
both reading ``changepoint_fired``.
"""
