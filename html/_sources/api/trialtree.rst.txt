.. _target-trialtree-api:
.. _target-trialtree:

TrialTree
=========

.. currentmodule:: ethograph.io.trialtree

:class:`TrialTree` is a wrapper around :class:`xarray.DataTree` that stores
one :class:`xarray.Dataset` per trial. Build one from a list of datasets,
then access each trial by ID or index:

.. code-block:: python

   import numpy as np, xarray as xr, ethograph as eto

   # Build: one xr.Dataset per trial
   datasets = []
   for i in range(1, 4):
       ds = xr.Dataset({"speed": xr.DataArray(np.random.rand(300), dims=["time"],
                                               coords={"time": np.arange(300) / 30.0})})
       ds.attrs["trial"] = i
       ds.attrs["fps"] = 30.0
       datasets.append(ds)

   dt = eto.from_datasets(datasets)

   # Access by trial ID (label-based, like xr.Dataset.sel)
   ds = dt.trial(2)
   ds.attrs["trial"]   # 2
   ds["speed"]          # the speed DataArray for trial 2

   # Access by integer index (0-based, like xr.Dataset.isel)
   ds = dt.itrial(0)
   ds.attrs["trial"]   # 1

   # List all trial IDs
   dt.trials            # [1, 2, 3]

   # Save / load
   dt.save("session.nc")
   dt = eto.open("session.nc")

Media file paths, trial timing, and stream offsets live in a separate
NWB alignment file (``<project>/.ethograph/alignment.nwb``), accessed via
``dt.nwb_alignment``. See :ref:`Media and alignment <target-trialtree-media-api>`.

For the :class:`xarray.Dataset` structure expected inside each trial, see
:doc:`../getting_started/data_requirements`.

.. autoclass:: TrialTree
   :no-members:
   :no-inherited-members:

----

Creating
--------

.. automethod:: TrialTree.open

.. automethod:: TrialTree.from_datasets

.. automethod:: TrialTree.from_continuous

.. automethod:: TrialTree.from_datatree

----

Accessing trials
----------------

.. automethod:: TrialTree.trial

.. automethod:: TrialTree.itrial

.. autoproperty:: TrialTree.trials

.. automethod:: TrialTree.get_all_trials

.. automethod:: TrialTree.get_common_attrs

.. automethod:: TrialTree.get_trial_metadata

----

Iterating over trials
---------------------

.. code-block:: python

   for trial_id, ds in dt.trial_items():
       print(f"Trial {trial_id}: {len(ds.time)} timepoints")

   # Apply a function to every trial, returning a new TrialTree
   dt_smoothed = dt.map_trials(lambda ds: smooth(ds))

.. automethod:: TrialTree.trial_items

.. automethod:: TrialTree.map_trials

----

Modifying trials
----------------

**In-place mutations** work directly through :meth:`~TrialTree.trial` because
the returned dataset shares its underlying data with the tree:

.. code-block:: python

   dt.trial(1).attrs["human_verified"] = True
   dt.trial(1)["speed"].values[:10] = 0.0

**Structural changes** (adding/removing variables) require
:meth:`~TrialTree.update_trial`:

.. code-block:: python

   dt.update_trial(1, lambda ds: ds.assign(
       smoothed_speed=ds["speed"].rolling(time=5).mean()
   ))

.. automethod:: TrialTree.update_trial

----

Filtering
---------

.. code-block:: python

   dt_tone_a = dt.filter_by_attr("stimulus", "tone_A")

.. automethod:: TrialTree.filter_by_attr

----

.. _target-trialtree-media-api:

Media files and alignment
-------------------------

Media filenames, trial timing, and stream offsets are stored in an **NWB
alignment file** at ``<project>/.ethograph/alignment.nwb``, not inside the
``.nc`` dataset. The alignment is accessed via ``dt.nwb_alignment``, which
is an :class:`~ethograph.io.nwb_alignment.NWBAlignment` (or its null-object
:class:`~ethograph.io.nwb_alignment.EmpytAlignment` when no NWB file is found).

Create alignment files with :func:`~ethograph.io.nwb_alignment.align_media_per_trial`
or :func:`~ethograph.io.nwb_alignment.align_media_from_streams`:

.. code-block:: python

   import pandas as pd
   from ethograph.io.nwb_alignment import align_media_per_trial

   trial_table = pd.DataFrame({
       "trial": [1, 2, 3],
       "video_cam-1": ["t1.mp4", "t2.mp4", "t3.mp4"],
       "pose_cam-1":  ["t1.h5", "t2.h5", "t3.h5"],
   })

   align_media_per_trial(
       trial_table,
       stream_rates={"video": 30.0, "pose": 30.0},
       output_path=".ethograph/alignment.nwb",
   )

Columns follow the ``{stream}_{device}`` convention (e.g. ``video_cam-1``,
``audio_mic-1``).

.. autofunction:: ethograph.io.nwb_alignment.align_media_per_trial

.. autofunction:: ethograph.io.nwb_alignment.align_media_from_streams

Reading alignment metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   alignment = dt.nwb_alignment
   alignment.cameras           # ["cam-1", "cam-2"]
   alignment.mics              # ["mic-1"]
   alignment.start_time(1)     # trial 1 start (seconds)
   alignment.get_media(1, "video", "cam-1")  # "t1.mp4"

.. autoclass:: ethograph.io.nwb_alignment.NWBAlignment
   :members: cameras, mics, start_time, stop_time, get_media, devices,
             resolve_media_path, stream_offset_for_trial, get_stream_rate,
             electrical_series, trials_df, has_real_timing, print_session
   :no-inherited-members:

.. autofunction:: ethograph.io.nwb_alignment.discover_nwb

.. autofunction:: ethograph.io.nwb_alignment.make_nwb_alignment

----

.. _target-trialtree-session-api:

Session table and timing
-------------------------

Trial timing (start/stop) is stored in the NWB alignment file's trials
table. When present, it enables session-mode navigation and restricting
neural data to trial windows.

.. code-block:: python

   alignment = dt.nwb_alignment
   alignment.start_time(1)      # 0.0
   alignment.stop_time(1)       # 120.0
   alignment.trials_df          # full trials DataFrame

----

.. _target-trialtree-offsets-api:

Stream offsets
--------------

For session-wide streams (e.g. ephys), the offset specifies when sample 0
of the file occurs in session time. This is stored in the alignment file
via :func:`~ethograph.io.nwb_alignment.align_media_from_streams` using the
``starting_time`` field:

.. code-block:: python

   from ethograph.io.nwb_alignment import align_media_from_streams

   streams = [
       {"name": "video_cam-1", "files": ["t1.mp4", "t2.mp4"], "rate": 30.0},
       {"name": "ephys_probe-1", "files": ["session.dat"],
        "rate": 30000.0, "starting_time": 0.5},
   ]
   align_media_from_streams(trials_df, streams, ".ethograph/alignment.nwb")

To query offsets at runtime:

.. code-block:: python

   alignment = dt.nwb_alignment
   alignment.stream_offset_for_trial(1, "ephys")   # offset in seconds

----

Labels
------

Labels are stored in a **TSV file** alongside the ``.nc`` file (see
:doc:`Label Storage <../user_guide/export_labels>` for the full format spec).
The TSV uses columns ``onset_s``, ``offset_s``, ``labels`` (int), ``individual``,
and ``trial``, plus per-trial metadata columns.

.. code-block:: python

   from ethograph.labels.tsv_store import load_labels_tsv, save_labels_tsv

   df = load_labels_tsv("data_labels.tsv")
   print(df[df["trial"] == 1])  # labels for trial 1

----

Saving
------

.. code-block:: python

   dt.save("path/to/session.nc")
   dt.save()  # overwrite the file it was loaded from

When saving to a new directory, the alignment NWB is automatically copied
alongside the ``.nc`` file.

.. automethod:: TrialTree.save

----

NWB import helpers
------------------

These functions probe NWB files to discover available data for import:

.. autofunction:: ethograph.io.nwb_import.read_trials_table

.. autofunction:: ethograph.io.nwb_import.probe_behavioral_series

.. autofunction:: ethograph.io.nwb_import.probe_electrical_series

.. autofunction:: ethograph.io.nwb_import.probe_label_sources

----

Dataset utilities
-----------------

Functions for building and augmenting datasets:

.. autofunction:: ethograph.io.dataset.downsample_trialtree

.. autofunction:: ethograph.io.dataset.add_changepoints_to_ds

.. autofunction:: ethograph.io.dataset.add_angle_rgb_to_ds
