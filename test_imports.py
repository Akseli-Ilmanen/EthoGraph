from ethograph.gui.plots_timeseriessource import TimeRange, RestrictionWindow, TrialAlignment, compute_trial_alignment
print("plots_timeseriessource: OK")

from ethograph.gui.modality import FileSource, ModalitySource, XarraySource, WindowedBuffer
print("modality: OK")

from ethograph.gui.widgets_data import DataWidget
print("widgets_data: OK")

from ethograph.gui.widgets_ephys import EphysWidget
print("widgets_ephys: OK")

import ethograph as eto
dt = eto.from_continuous.__doc__
print("eto.from_continuous: OK")

print("\nAll imports clean!")
