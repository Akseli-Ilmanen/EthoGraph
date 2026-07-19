"""Ad-hoc check: draw pose points for one frame directly onto a video frame.

Bypasses the GUI entirely. Interpretation:
- Points offset here too  -> offset is in the data (pose tracked on a
  cropped/downsampled/other variant of this video file).
- Points align here but not in the GUI -> bug in the GUI overlay pipeline.

Usage:
    python tests/_test_pose_overlay_alignment.py VIDEO POSE_FILE SOURCE_SOFTWARE [FRAME]

Example:
    python tests/_test_pose_overlay_alignment.py cam1.mp4 cam1DLC.h5 DeepLabCut 500
"""

import sys

import av
import matplotlib.pyplot as plt
import numpy as np
from movement.io import load_dataset


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    video_path, pose_path, software = sys.argv[1:4]
    frame_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                img = frame.to_ndarray(format="rgb24")
                break
        else:
            raise SystemExit(f"Video has fewer than {frame_idx + 1} frames")

    ds = load_dataset(pose_path, software, fps=None)
    pos = ds.position.isel(time=frame_idx)
    xs = pos.sel(space="x").values.ravel()
    ys = pos.sel(space="y").values.ravel()

    h, w = img.shape[:2]
    print(f"video frame:  {w} x {h}  (w x h)")
    print(f"pose n_frames: {ds.sizes['time']}")
    print(
        f"pose x range (all frames): {np.nanmin(ds.position.sel(space='x').values):.1f}"
        f" .. {np.nanmax(ds.position.sel(space='x').values):.1f}"
    )
    print(
        f"pose y range (all frames): {np.nanmin(ds.position.sel(space='y').values):.1f}"
        f" .. {np.nanmax(ds.position.sel(space='y').values):.1f}"
    )

    # imshow uses origin="upper" (y down) — the same convention pose files use,
    # so this is a direct data-space comparison with no flip involved.
    plt.imshow(img)
    plt.scatter(xs, ys, s=40, facecolors="none", edgecolors="red", linewidths=1.5)
    for name, x, y in zip(
        np.repeat(ds.coords["keypoints"].values, ds.sizes.get("individuals", 1)), xs, ys
    ):
        if not (np.isnan(x) or np.isnan(y)):
            plt.annotate(str(name), (x, y), color="yellow", fontsize=7)
    plt.title(f"frame {frame_idx} — points should sit on the animal")
    plt.show()


if __name__ == "__main__":
    main()
