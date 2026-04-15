.. _target-api:

API reference
=============

.. toctree::
   :hidden:

   api/trialtree


TrialTree
---------

The core data structure — see the full :doc:`TrialTree API <api/trialtree>`
for detailed documentation with interleaved examples.


Top-level functions
-------------------

.. currentmodule:: ethograph

.. autosummary::
   :toctree: api
   :nosignatures:

   open
   from_datasets
   load_nap_data
   downsample_trialtree
   sel_valid
   align_media_per_trial
   align_media_from_streams
   NWBAlignment


.. rubric:: Modules

.. autosummary::
   :toctree: api
   :recursive:
   :nosignatures:

   ethograph.io.dataset
   ethograph.io.nwb_alignment
   ethograph.io.nwb_import
   ethograph.io.pynapple
   ethograph.labels.intervals
   ethograph.labels.ml
   ethograph.labels.tsv_store
   ethograph.labels.predictions
   ethograph.labels.crowsetta_format
   ethograph.labels.converters
   ethograph.labels.export
   ethograph.labels.plots
   ethograph.utils.xr_utils
