.. _target-changepoints-api:

Changepoints
============

.. currentmodule:: ethograph.features.changepoints

Changepoint detection finds candidate behavioural boundaries in kinematic,
audio, and spectral time series. Ethograph exposes detectors (binary masks
aligned to the signal's time axis), merging and snapping utilities, and
feature generators that turn changepoints into model-ready inputs.

See :doc:`../advanced/changepoints/index` for the conceptual overview and
the :ref:`changepoint features <target-changepoint-features-api>` section
below for the representations used by downstream segmentation models.

----

Detection
---------

Detectors consume a 1-D signal and return a binary (0/1) mask of the same
length. NaN boundaries in the input are always added as changepoints so that
valid/invalid transitions are not lost.

.. autofunction:: find_peaks_binary

.. autofunction:: find_troughs_binary

.. autofunction:: find_nearest_turning_points_binary

.. autofunction:: add_NaN_boundaries

To attach detectors to a :class:`xarray.Dataset` or a pynapple object:

.. autofunction:: ethograph.io.dataset.add_changepoints_to_ds

.. autofunction:: ethograph.io.pynapple.add_changepoints_to_nap

----

Merging and time extraction
---------------------------

Changepoints stored as multiple ``kind="changepoint_feature"`` (legacy: ``attrs["type"] == "changepoints"``)
DataArrays can be merged into a single boolean mask, or converted directly
to absolute times (seconds) for use by the GUI and correction pipeline.

.. autofunction:: merge_changepoints

.. autofunction:: extract_cp_times

.. autofunction:: snap_to_nearest_changepoint_time

----

Label correction
----------------

Snap interval-based labels to nearby changepoints. The full pipeline
(``correct_changepoints``) runs purge → stitch → snap → purge; see
:doc:`../advanced/changepoints/correction` for parameter guidance.

.. autofunction:: correct_changepoints

.. autofunction:: correct_changepoints_automatic

.. autofunction:: correct_changepoints_dense

----

.. _target-changepoint-features-api:

Changepoint features
--------------------

Once changepoints have been curated, they are converted into learnable
features for action- and audio-segmentation models (transformers, MS-TCN,
DLC2Action, ASFormer, ...). Three complementary representations let the
model pick up changepoint structure at different scales:

.. list-table::
   :header-rows: 1
   :widths: 25 60

   * - Feature
     - Example (changepoints at t=4 and t=8)
   * - **Binary changepoints** — exact positions
     - ``0 0 0 0 1 0 0 0 1 0 0 0 0 0``
   * - **Smooth changepoints** — proximity to nearest changepoint
     - ``0 0 0 .3 1 .3 0 .3 1 .3 0 0 0 0``
   * - **Segment IDs** — unique ID per inter-changepoint region
     - ``0 0 0 0 1 1 1 1 2 2 2 2 2 2``

Smooth changepoints use a Laplacian kernel centred at each changepoint
index :math:`i`:

.. math::

   \text{smooth}(t) = \sum_i \exp\!\left(-\frac{|t - i|}{\sigma}\right)

Laplacian peaks are narrow (so they pinpoint the changepoint) but have long
tails (so they remain visible from far away). Passing several
``sigmas`` — e.g. ``[0.5, 3, 5]`` — yields a multi-scale view.

Weighted variants emphasise changepoints where a target signal ``x``
(typically speed) is low, via :math:`\exp(-x / (\bar{x} + \epsilon))`. This
helps models distinguish speed minima before/after movements from minima
occurring *within* a movement.

.. autofunction:: more_changepoint_features

----

Storage format
--------------

Kinematic changepoints are stored as binary (``int8``) DataArrays sharing
their feature's time axis, tagged with ``kind = "changepoint_feature"`` (plus the legacy ``attrs["type"] = "changepoints"``)
and ``attrs["target_feature"]``. Audio changepoints, which would be
prohibitively large at audio sample rates, are stored instead as
``audio_cp_onsets`` / ``audio_cp_offsets`` float pairs. See
:doc:`../advanced/changepoints/kinematic` and
:doc:`../advanced/changepoints/audio` for examples.
