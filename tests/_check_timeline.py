"""Quick standalone check for timeline tests — run directly, not via pytest."""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from ethograph.gui.wizard_multi_timeline import _normalize_trial_key
from ethograph.gui.wizard_overview import WizardState

# _normalize_trial_key
assert _normalize_trial_key(3) == 3
assert _normalize_trial_key("42") == 42
assert _normalize_trial_key("trial_A") == "trial_A"
assert _normalize_trial_key(None) is None
assert _normalize_trial_key("  7  ") == 7
print("_normalize_trial_key: OK")

from qtpy.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

from ethograph.gui.wizard_multi_timeline import TimelinePage

# xx.csv round-trip
df = pd.read_csv(Path(__file__).parents[1] / "data" / "xx.csv")
state = WizardState()
state.trial_table = df
state.files_aligned_to_trials = False
page = TimelinePage()
page.populate_from_state(state)
app.processEvents()
expected = df["stop_time"].max()
assert abs(page._total_duration - expected) < 1e-6, f"{page._total_duration} != {expected}"
print(f"populate_from_state (xx.csv, {len(df)} trials): OK — total_duration={page._total_duration:.2f}s")

# NaN stop_time regression
df_nan = pd.DataFrame({"start_time": [0.0, 5.0], "stop_time": [float("nan"), 10.0], "trial": [1, 2]})
state2 = WizardState()
state2.trial_table = df_nan
state2.files_aligned_to_trials = False
page2 = TimelinePage()
page2.populate_from_state(state2)
app.processEvents()
assert math.isfinite(page2._total_duration), f"expected finite, got {page2._total_duration}"
assert abs(page2._total_duration - 10.0) < 1e-6
print("NaN stop_time regression: OK")

# All-NaN — total_duration stays at constructor default (1.0)
df_allnan = pd.DataFrame({"start_time": [0.0, 5.0], "stop_time": [float("nan"), float("nan")], "trial": [1, 2]})
state3 = WizardState()
state3.trial_table = df_allnan
state3.files_aligned_to_trials = False
page3 = TimelinePage()
page3.populate_from_state(state3)
app.processEvents()
assert page3._total_duration == 1.0, f"expected 1.0, got {page3._total_duration}"
print("All-NaN stop_time: OK")

# Empty trial table
df_empty = pd.DataFrame(columns=["start_time", "stop_time", "trial"])
state4 = WizardState()
state4.trial_table = df_empty
state4.files_aligned_to_trials = False
page4 = TimelinePage()
page4.populate_from_state(state4)
app.processEvents()
print("Empty trial table: OK")

# None trial table
state5 = WizardState()
state5.trial_table = None
state5.files_aligned_to_trials = False
page5 = TimelinePage()
page5.populate_from_state(state5)
app.processEvents()
print("None trial table: OK")

print("\nAll checks passed.")
