
import av
import dask
import dask.array as da
import napari
import numpy as np

url = (
    "https://api.dandiarchive.org/api/assets/66bd0842-61da-48d1-9bf5-60f67508fafb/download"
)



container = av.open(url)
stream = container.streams.video[0]
stream.thread_type = "AUTO"

fps = float(stream.average_rate)
H, W = stream.height, stream.width
print(f"Video: {W}x{H} @ {fps:.1f} fps")

# Decode first N frames into a (N, H, W, 3) array — napari needs
# that leading axis to show a slider
N = 300
frames = []
for i, frame in enumerate(container.decode(video=0)):
    if i >= N:
        break
    frames.append(frame.to_ndarray(format="rgb24"))

data = np.stack(frames)  # (N, H, W, 3)

viewer = napari.Viewer()
viewer.add_image(data, rgb=True, name="remote_video")
napari.run()