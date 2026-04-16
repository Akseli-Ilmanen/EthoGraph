.. _target-pynapple-io-api:

Pynapple IO
===========

.. currentmodule:: ethograph.io.pynapple

Loaders and augmenters for pynapple-backed data (NWB files and ``.npz``
pynapple folders). These are the pynapple counterparts of the xarray
builders in :doc:`dataset`.

----

Loading
-------

.. autofunction:: load_nap_data

.. autofunction:: detect_trials

.. autofunction:: flatten_nap_folder

.. autofunction:: get_metadata

.. autofunction:: parse_nwb_types

----

Building
----------

.. autofunction:: add_changepoints_to_nap

.. autofunction:: add_angle_rgb_to_nap

----

NWB import probes
-----------------

.. currentmodule:: ethograph.io.nwb_import

These helpers inspect an NWB file to discover what is available for
import (trials, behavioural series, electrical series, label sources).

.. autofunction:: read_trials_table

.. autofunction:: probe_electrical_series

.. autofunction:: probe_label_sources
