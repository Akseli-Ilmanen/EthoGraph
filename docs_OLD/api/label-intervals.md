# Label Intervals

Interval-based label creation, editing, and conversion utilities. Labels are stored as a pandas {class}pandas.DataFrame or {class}xarray.Dataset with columns `onset_s`, `offset_s`, `labels`, and `individual`.

See also:
- {mod}movement.features.label_intervals for related interval utilities
- {mod}audian for audio annotation tools

::: ethograph.utils.label_intervals
    options:
      members:
        - dense_to_intervals
        - intervals_to_dense
        - intervals_to_xr
        - xr_to_intervals
        - add_interval
        - delete_interval
        - find_interval_at
        - get_interval_bounds
        - purge_short_intervals
        - stitch_intervals
        - snap_boundaries
        - crowsetta_to_intervals
        - build_mapping_from_labels
        - write_mapping_file
        - resolve_crowsetta_mapping
