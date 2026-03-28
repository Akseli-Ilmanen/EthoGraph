# Quick start

This page gives a quick overview of what ethograph can do.

## Loading data

```python
import ethograph as eto

# Open a NetCDF dataset
dt = eto.open("my_data.nc")

# Inspect trials
print(dt.trials)

# Access a single trial
ds = dt.trial("trial_01")
```

## Launching the GUI

From the command line:

```bash
ethograph launch
```

Or from Python:

```python
import ethograph as eto
from ethograph.cli import launch

launch()
```

The GUI lets you:

- Load NetCDF datasets with multi-trial behavioural data
- View synchronized video and audio
- Browse time-series features as line plots, heatmaps, or spectrograms
- Create and edit interval-based labels
- Detect changepoints in kinematic and audio data
- Visualise spike-sorted neural data alongside behaviour

See the {doc}`../user_guide/index` for the full GUI guide.
