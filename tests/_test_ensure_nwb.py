"""Quick smoke test for _ensure_alignment_nwb."""
from pathlib import Path
import tempfile
import pandas as pd
import xarray as xr
import numpy as np
from ethograph.io.trialtree import TrialTree, _SETTINGS_DIR, _NWB_FILENAME
from ethograph.utils.nwb import build_nwb_from_trial_table

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)

    ds = xr.Dataset(
        {"x": (["time"], np.zeros(10))},
        coords={"time": np.arange(10) / 10.0},
    )
    ds.attrs["trial"] = 1
    dt = TrialTree(name=None, children={"trial_1": xr.DataTree(dataset=ds)})

    # Create alignment NWB in dir A
    dir_a = tmp / "media"
    dir_a.mkdir()
    nwb_a = dir_a / _SETTINGS_DIR / _NWB_FILENAME
    nwb_a.parent.mkdir(parents=True)
    trial_table = pd.DataFrame([{"trial": 1, "start_time": 0.0}])
    build_nwb_from_trial_table(trial_table, stream_rates={"video": 30.0, "pose": 30.0}, output_path=nwb_a)
    dt._nwb_path = str(nwb_a)

    # Call _ensure_alignment_nwb for dir B
    dir_b = tmp / "output"
    dir_b.mkdir()
    dt._ensure_alignment_nwb(dir_b)

    nwb_b = dir_b / _SETTINGS_DIR / _NWB_FILENAME
    print("NWB copied:", nwb_b.exists())
    print("dt._nwb_path updated:", dt._nwb_path == str(nwb_b))

    # Verify same-dir is a no-op
    size_before = nwb_b.stat().st_size
    dt._ensure_alignment_nwb(dir_b)
    print("No-op on same dir:", nwb_b.stat().st_size == size_before)
