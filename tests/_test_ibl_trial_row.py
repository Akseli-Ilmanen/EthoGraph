"""Verify _trial_row works for IBL NWB (no trial column, index-based)."""
from ethograph.io.nwb_alignment import NWBAlignment

sio = NWBAlignment(r"C:\Users\aksel\Desktop\ibl\ibl.nwb")
print(f"has trial col: {'trial' in sio.trials_df.columns}")

for trial_id in [1, 2, 3, 533]:
    row = sio._trial_row(trial_id)
    start = sio.start_time(trial_id)
    stop = sio.stop_time(trial_id)
    duration = stop - start if stop is not None else None
    print(f"  trial={trial_id}: start={start:.3f}, stop={stop}, dur={duration}")

sio.close()
