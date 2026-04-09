"""Quick inspection of the freshly-built Moll2025_pynapple alignment."""
from ethograph.io.nwb_alignment import make_nwb_alignment

nwb_path = r"C:\Users\aksel\.ethograph\example_data\Moll2025_pynapple\.ethograph\alignment.nwb"
sio = make_nwb_alignment(nwb_path)
print("cameras:", sio.cameras)
print("mics:", sio.mics)
for trial in [1, 2]:
    print(f"  trial {trial}: start={sio.start_time(trial)}, stop={sio.stop_time(trial)}")
    vid = sio.get_media(trial, "video", "cam-1")
    pose = sio.get_media(trial, "pose", "cam-1")
    print(f"    video={vid}, pose={pose}")
    offset = sio.stream_offset_for_trial(trial, "video", "cam-1")
    rate = sio.get_stream_rate("video", "cam-1")
    print(f"    offset={offset}, rate={rate}")
