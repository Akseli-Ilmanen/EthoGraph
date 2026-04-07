from ethograph.io.time_model import TimeRange, RestrictionWindow, TrialVideoBounds, compute_trial_video_bounds
print("time_model: OK")

from ethograph.io.plot_sources import FileSource, PlotSource, XarraySource, WindowedBuffer
print("modality: OK")

from ethograph.gui.widgets_data import DataWidget
print("widgets_data: OK")

from ethograph.gui.widgets_ephys import EphysWidget
print("widgets_ephys: OK")

import ethograph as eto
dt = eto.from_continuous.__doc__
print("eto.from_continuous: OK")

print("\nAll imports clean!")
