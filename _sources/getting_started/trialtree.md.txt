(target-trialtree-getting-started)=
# TrialTree

{class}`~ethograph.io.trialtree.TrialTree` is the core data structure in ethograph — a wrapper around {class}`xarray.DataTree` that stores one {class}`xarray.Dataset` per trial.

Each child node holds one trial's data (timeseries, features, attributes), and the tree as a whole carries session-level metadata (media paths, stream offsets, alignment). You get trial access by ID or index, iteration, in-place and structural mutation, filtering, and continuous-mode lazy slicing — all while preserving xarray semantics.

For the {class}`xarray.Dataset` structure expected inside each trial, see {doc}`data_requirements`.

```{seealso}
For usage examples and the full API — creating, accessing, iterating, modifying, filtering, saving — see {doc}`../api/trialtree`.
```
