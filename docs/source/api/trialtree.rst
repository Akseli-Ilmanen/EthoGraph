.. _target-trialtree-api:
.. _target-trialtree:

TrialTree
=========

.. currentmodule:: ethograph.io.trialtree

TrialTree is a thin wrapper around :class:`xarray.DataTree` for
multi-trial behavioural datasets. Each trial is a child node holding an
:class:`xarray.Dataset`, and the tree provides convenience methods for
accessing, iterating, and modifying trials.

.. code-block:: text

   TrialTree (root)
   +-- "session"  ->  xr.Dataset  (timing, media filenames, stream offsets)
   +-- "1"  ->  xr.Dataset  (trial 1: features, coords, attrs)
   +-- "2"  ->  xr.Dataset  (trial 2)
   +-- "3"  ->  xr.Dataset  (trial 3)
   +-- ...

The dataset format builds on :mod:`movement` conventions for representing
pose estimation and behavioural time series. For the :class:`xarray.Dataset`
structure expected inside each trial, see :doc:`../getting_started/data_requirements`.

.. autoclass:: TrialTree
   :no-members:
   :no-inherited-members:

----

Opening and creating
--------------------

Open an existing ``.nc`` file or build from a list of datasets.

.. code-block:: python

   import ethograph as eto

   dt = eto.open("path/to/trials.nc")
   dt.trials  # [1, 2, 3, ...]

.. automethod:: TrialTree.open

.. automethod:: TrialTree.from_datasets

To create a :class:`TrialTree` from a single dataset (e.g. from a
:mod:`movement` dataset or numpy array):

.. code-block:: python

   ds = ...  # your xr.Dataset with a time dimension
   dt = eto.dataset_to_basic_trialtree(ds)

.. autofunction:: ethograph.io.dataset.dataset_to_basic_trialtree

----

Accessing trials
----------------

Inspired by :meth:`~xarray.Dataset.sel` / :meth:`~xarray.Dataset.isel`:

.. code-block:: python

   ds = dt.trial(1)      # by trial ID (label-based)
   ds = dt.itrial(0)     # by integer index (0-based)

.. automethod:: TrialTree.trial

.. automethod:: TrialTree.itrial

.. autoproperty:: TrialTree.trials

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

Media files
-----------

Media filenames are stored in the session node via :meth:`~TrialTree.set_media`.
Call it once per stream:

.. list-table::
   :header-rows: 1

   * - Stream
     - Default layout
     - Device dimension
   * - ``"video"``
     - per-trial
     - ``"cameras"``
   * - ``"audio"``
     - per-trial
     - ``"mics"``
   * - ``"pose"``
     - per-trial
     - ``"cameras"``
   * - ``"ephys"``
     - session-wide
     - *(none)*

.. code-block:: python

   # Two cameras, three trials
   dt.set_media("video",
       [["trial001_cam1.mp4", "trial001_cam2.mp4"],
        ["trial002_cam1.mp4", "trial002_cam2.mp4"],
        ["trial003_cam1.mp4", "trial003_cam2.mp4"]],
       device_labels=["left", "right"],
   )

   # Session-wide audio
   dt.set_media("audio",
       ["session_ch1.wav", "session_ch2.wav"],
       device_labels=["mic-1", "mic-2"],
       per_trial=False,
   )

.. automethod:: TrialTree.set_media

.. automethod:: TrialTree.get_media

.. automethod:: TrialTree.devices

.. autoproperty:: TrialTree.cameras

.. autoproperty:: TrialTree.mics

----

.. _target-trialtree-session-api:

Session table and timing
-------------------------

The session table is an :class:`xarray.Dataset` in the ``"session"`` child
node. It holds trial timing, media filenames, and stream offset attributes.

.. code-block:: python

   import pandas as pd

   session_table = pd.DataFrame({
       "trial": [1, 2],
       "start_time": [0.0, 120.5],
       "stop_time": [120.0, 245.0],
   })
   dt = eto.from_datasets(datasets, session_table=session_table)

   dt.start_time(1)          # 0.0
   dt.stop_time(1)           # 120.0
   dt.trial_duration(1)      # 120.0

.. autoproperty:: TrialTree.session

.. automethod:: TrialTree.set_session_table

.. automethod:: TrialTree.session_to_dataframe

.. automethod:: TrialTree.print_session

.. automethod:: TrialTree.start_time

.. automethod:: TrialTree.stop_time

.. automethod:: TrialTree.trial_duration

----

Pynapple integration
--------------------

Trial epochs are exposed as :class:`pynapple.IntervalSet` objects for
restricting neural data to trial windows:

.. code-block:: python

   epoch = dt.trial_epoch(1)
   spikes_t1 = dt.restrict(spikes, 1)

.. autoproperty:: TrialTree.trials_ep

.. automethod:: TrialTree.trial_epoch

.. automethod:: TrialTree.restrict

----

.. _target-trialtree-offsets-api:

Stream offsets
--------------

For session-wide streams, :meth:`~TrialTree.set_stream_offset` specifies
when sample 0 of the file occurs in session-absolute time.
:meth:`~TrialTree.source_start_time` then computes the trial-relative offset:

.. code-block:: python

   dt.set_stream_offset("ephys", 0.0)
   dt.source_start_time(1, "video")   # 0.0 (per-trial)
   dt.source_start_time(2, "ephys")   # e.g. -120.5 (session-wide)

.. automethod:: TrialTree.set_stream_offset

.. automethod:: TrialTree.source_start_time

----

Labels
------

Labels (segment annotations) are stored as interval variables (``onset_s``,
``offset_s``, ``labels``, ``individual``) on a ``segment`` dimension.
:meth:`~TrialTree.get_label_dt` extracts just the label data into a
lightweight :class:`TrialTree`, stripping all feature variables.

.. code-block:: python

   label_dt = dt.get_label_dt()
   empty_dt = dt.get_label_dt(empty=True)

.. automethod:: TrialTree.get_label_dt

.. automethod:: TrialTree.overwrite_with_labels

----

Saving
------

.. code-block:: python

   dt.save("path/to/trials.nc")
   dt.save()  # overwrite the file it was loaded from

.. automethod:: TrialTree.save
