# TrialTree

Hierarchical container for multi-trial behavioral datasets. Inherits from {class}xarray.DataTree. Each trial is a child node holding an {class}xarray.Dataset with the trial identifier stored in `ds.attrs["trial"]`. Provides methods for loading, accessing, and saving trial datasets, as well as managing session-level metadata and media.

See also:
- {mod}movement for movement analysis workflows
- {mod}pyqt for Qt-based GUI components

::: ethograph.utils.trialtree.TrialTree
    options:
      show_root_heading: true
      show_bases: false
      members:
        - open
        - from_datasets
        - trials
        - trial_items
        - trial
        - itrial
        - get_all_trials
        - get_common_attrs
        - session
        - set_session_table
        - session_to_dataframe
        - cameras
        - mics
        - set_media
        - get_media
        - devices
        - start_time
        - stop_time
        - trial_duration
        - trials_ep
        - trial_epoch
        - get_label_dt
        - overwrite_with_labels
        - overwrite_with_attrs
        - map_trials
        - update_trial
        - filter_by_attr
        - save
