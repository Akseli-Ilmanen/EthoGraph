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

----

Building
----------

.. autofunction:: add_changepoints_to_nap

.. autofunction:: add_angle_rgb_to_nap
