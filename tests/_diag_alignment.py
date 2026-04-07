"""Quick diagnostic: check trial alignment from remote DANDI NWB."""
import warnings
warnings.filterwarnings("ignore")

from ethograph.utils.dandi import open_nwb_dandi

nwb, io, h5, rf = open_nwb_dandi("000409", "196aa923-79c8-4524-a1b2-344fc30d8cb2")
df = nwb.trials.to_dataframe()
print("Columns:", list(df.columns))
print("Index (first 5):", df.index[:5].tolist())
has_trial_col = "trial" in df.columns
print("Has 'trial' column:", has_trial_col)
if has_trial_col:
    print("trial values (first 5):", df["trial"].iloc[:5].tolist())
print("start_time (first 3):", df["start_time"].iloc[:3].tolist())
print("stop_time (first 3):", df["stop_time"].iloc[:3].tolist())

from ethograph.io.nwb_alignment import NWBAlignment
sio = NWBAlignment.from_nwb_object(nwb)
print()
print("trials_df shape:", sio.trials_df.shape)
print("start_time(1):", sio.start_time(1))
print("stop_time(1):", sio.stop_time(1))
print("start_time(2):", sio.start_time(2))
print("stop_time(2):", sio.stop_time(2))
print("cameras:", sio.cameras)
print("mics:", sio.mics)
