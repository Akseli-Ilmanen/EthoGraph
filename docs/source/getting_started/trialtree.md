(target-trialtree)=
# TrialTree

{class}`~ethograph.io.trialtree.TrialTree` is the core data structure in ethograph — a thin wrapper around {class}`xarray.DataTree` for multi-trial behavioural datasets. Each trial is stored as a child node containing an {class}`xarray.Dataset`.

```
TrialTree (root)
+-- "session"  ->  xr.Dataset  (timing, media filenames, stream offsets)
+-- "1"  ->  xr.Dataset  (trial 1: features, coords, attrs)
+-- "2"  ->  xr.Dataset  (trial 2)
+-- ...
```

The dataset format builds on {mod}`movement` conventions for representing pose estimation and behavioural time series.

```{seealso}
For the full API with all methods, parameters, and code examples, see {doc}`../api/trialtree`.
```

## Key concepts

### Trials

Access trials by ID or by index, iterate over them, or apply a function to all trials at once. See {meth}`~ethograph.io.trialtree.TrialTree.trial`, {meth}`~ethograph.io.trialtree.TrialTree.itrial`, {meth}`~ethograph.io.trialtree.TrialTree.trial_items`, {meth}`~ethograph.io.trialtree.TrialTree.map_trials`.

### Media files

Video, audio, and pose filenames are stored at the session level (not inside individual trial datasets). Filenames are stored as **filename-only strings** so datasets remain valid when folders are moved. See {meth}`~ethograph.io.trialtree.TrialTree.set_media`, {meth}`~ethograph.io.trialtree.TrialTree.get_media`.

| Stream | Device dimension |
|--------|-----------------|
| `"video"` | `"cameras"` |
| `"audio"` | `"mics"` |
| `"pose"` | `"cameras"` |
| `"ephys"` | *(none — session-wide, selected in GUI)* |

### Session table and timing

Trial start/stop times, stream offsets, and session metadata are stored in the `"session"` child node. When timing is available, pynapple integration provides {class}`pynapple.IntervalSet` objects for restricting neural data to trial windows. See {meth}`~ethograph.io.trialtree.TrialTree.start_time`, {meth}`~ethograph.io.trialtree.TrialTree.trial_epoch`, {meth}`~ethograph.io.trialtree.TrialTree.restrict`.

### Stream offsets

When multiple streams (video, audio, ephys) run on different clocks, {meth}`~ethograph.io.trialtree.TrialTree.set_stream_offset` records the offset so traces align correctly. See {meth}`~ethograph.io.trialtree.TrialTree.source_start_time`.

### Labels

Labels are stored as interval variables (`onset_s`, `offset_s`, `labels`, `individual`) on a `segment` dimension. {meth}`~ethograph.io.trialtree.TrialTree.get_label_dt` extracts a lightweight label-only tree that can be saved independently. See {meth}`~ethograph.io.trialtree.TrialTree.overwrite_with_labels`.

### Modifying trials

In-place mutations (changing attribute values, modifying existing arrays) work directly through {meth}`~ethograph.io.trialtree.TrialTree.trial`. Structural changes (adding/removing variables) require {meth}`~ethograph.io.trialtree.TrialTree.update_trial`.

### Saving and loading

```python
import ethograph as eto

dt = eto.open("trials.nc")       # load
dt.save("trials.nc")             # save
dt = eto.from_datasets([ds1, ds2])  # build from datasets
```
