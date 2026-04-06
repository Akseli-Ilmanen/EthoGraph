"""Quick debug script to check trials_ep on IBL NWB file."""
from ethograph.io.nwb_alignment import NWBAlignment

sio = NWBAlignment(r"C:\Users\aksel\Desktop\ibl\ibl.nwb")
df = sio.trials_df
print(f"trials_df: {len(df)} rows")
print(f"columns (first 5): {list(df.columns)[:5]}")
print(f"has start_time: {'start_time' in df.columns}")
print(f"has stop_time: {'stop_time' in df.columns}")
print(f"has trial col: {'trial' in df.columns}")

ep = sio.trials_ep
print(f"trials_ep is None: {ep is None}")
if ep is not None:
    print(f"len(trials_ep): {len(ep)}")
    print(f"first 3 intervals: {ep[:3]}")
else:
    print("BUG: trials_ep is None despite having start/stop times!")
sio.close()
