"""Demo: split BirdPark into 3 x 20s trials using from_continuous.

Run:  python examples/demo_from_continuous.py

Creates a continuous TrialTree + alignment NWB, then prints
trial timing and video offsets to verify everything works.
"""

from pathlib import Path

import pandas as pd
import ethograph as eto
from ethograph.utils.nwb import build_nwb_from_trial_table

BIRDPARK_DIR = Path.home() / ".ethograph" / "example_data" / "BirdPark"
NC_FILE = BIRDPARK_DIR / "copExpBP08_trim.nc"
VIDEO = "BP_2021-05-25_08-12-51_655154_0380000.mp4"
AUDIO = "BP_2021-05-25_08-12-51_655154_0380000.wav"


def main():
    # 1. Load the original single-trial dataset
    orig_dt = eto.open(str(NC_FILE))
    ds = orig_dt.itrial(0)
    fps = float(ds.attrs["fps"])
    total_time = float(ds.time.values[-1])
    print(f"Original dataset: {total_time:.1f}s, fps={fps:.2f}")
    print(f"  variables: {list(ds.data_vars)}")
    print()

    # 2. Define 3 x 20s trial epochs
    epochs = pd.DataFrame({
        "trial": [1, 2, 3],
        "start_time": [0.0, 20.0, 40.0],
        "stop_time": [20.0, 40.0, 60.0],
    })

    # 3. Create continuous TrialTree (no data copying — slices on demand)
    dt = eto.from_continuous(ds, epochs)
    print(f"Continuous TrialTree: {len(dt.trials)} trials")
    print(f"  trials: {dt.trials}")
    print()

    # 4. Inspect each trial — time should start at 0
    for trial_id in dt.trials:
        trial_ds = dt.trial(trial_id)
        t0 = float(trial_ds.time.values[0])
        t1 = float(trial_ds.time.values[-1])
        n = len(trial_ds.time)
        print(f"  Trial {trial_id}: time={t0:.3f}–{t1:.3f}s ({n} samples)")

    # 5. Create alignment NWB (session-wide video + audio)
    trial_table = epochs.copy()
    trial_table["video_cam-1"] = VIDEO
    trial_table["audio_mic-1"] = AUDIO

    nwb_path = BIRDPARK_DIR / ".ethograph" / "alignment_demo.nwb"
    build_nwb_from_trial_table(trial_table, stream_rates={"video": fps, "pose": fps}, output_path=nwb_path)
    dt.nwb_path = str(nwb_path)
    print(f"\nAlignment NWB: {nwb_path}")

    # 6. Verify session_io works
    print(f"  cameras: {dt.cameras}")
    print(f"  mics:    {dt.mics}")
    print()

    # 7. Show video/audio offsets per trial
    print("Video/audio offsets (trial-relative):")
    for trial_id in dt.trials:
        v_off = dt.stream_offset_for_trial(trial_id, "video", "cam-1")
        a_off = dt.stream_offset_for_trial(trial_id, "audio", "mic-1")
        t_start = dt.start_time(trial_id)
        t_stop = dt.stop_time(trial_id)
        print(f"  Trial {trial_id}: session={t_start:.0f}–{t_stop:.0f}s  "
              f"video_offset={v_off:.1f}s  audio_offset={a_off:.1f}s")

    # Cleanup demo NWB (close session_io handle first on Windows)
    dt.session_io.close()
    dt.__dict__.pop("session_io", None)
    nwb_path.unlink(missing_ok=True)
    print("\nDone! (cleaned up demo NWB)")


if __name__ == "__main__":
    main()
