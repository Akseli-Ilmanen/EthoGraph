.. _target-dataset-api:

Dataset
=======

.. currentmodule:: ethograph.io.dataset

Dataset-level builders and augmenters that operate on an
:class:`xarray.Dataset` or :class:`~ethograph.io.trialtree.TrialTree`. Use
these to build a basic tree from a single long recording, downsample, or
attach changepoint / colour features.

For the pynapple equivalents (``load_nap_data``,
``add_changepoints_to_nap``, ...) see :doc:`pynapple_io`.

----

Resampling
----------

.. autofunction:: downsample_trialtree

----

Building
----------

.. autofunction:: add_changepoints_to_ds

.. autofunction:: add_angle_rgb_to_ds

----

Pynapple equivalents
--------------------

.. currentmodule:: ethograph.io.pynapple

.. autofunction:: add_changepoints_to_nap

.. autofunction:: add_angle_rgb_to_nap
