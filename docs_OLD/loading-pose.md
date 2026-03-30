# From a pose file

Use this path if you have pose estimation output from DeepLabCut, SLEAP, LightningPose, or any other tracker that produces `.h5` or `.csv` files — with or without a matching video.

The **Create dialog** handles single recordings (one trial) without any scripting. For multiple trials or multiple cameras, see [Multi-trial setup](loading-script.md).

---

## Steps

--8<-- "snippets/launch-single.md"

1. Under **Single trial**, select: **1) Generate from pose file (DeepLabCut, SLEAP, ...)**
2. Click **Next →** — the dialog opens
3. Set **Source software** (DeepLabCut, SLEAP, LightningPose, …) and **Pose file** (`.h5` or `.csv`)
4. Optionally set **Video file** — frame rate is auto-detected from the video
5. Set **Output path** for the generated `trials.nc`
6. Click **Generate .nc file**
7. The I/O widget auto-populates → click **Load**

---

## What the loader computes

Beyond raw position, the loader automatically computes kinematic features for each keypoint:

- Velocity, acceleration, speed

These appear in the Feature dropdown and are useful for identifying movement onset/offset. See [Kinematic changepoints](changepoints.md#kinematic-changepoints).

---

## Dialog fields

| Field | Notes |
|-------|-------|
| **Source software** | DeepLabCut, SLEAP, LightningPose, … |
| **Pose file** | `.h5` or `.csv` |
| **Video file** | Optional — frame rate auto-detected |
| **Frame rate** | Auto-detected from video; set manually if no video |
| **Video onset in pose** | Time offset (s) where video starts relative to pose stream |
| **Output path** | Where to save the generated `trials.nc` |

---

## Data requirements

| Attribute | Value |
|-----------|-------|
| `attrs["fps"]` | Auto-detected from video, or set in the dialog |
| `coords["individuals"]` | From pose file |
| `coords["keypoints"]` | From pose file |
| `coords["space"]` | `["x", "y"]` (or `["x", "y", "z"]` for 3D) |
| `attrs["type"]` on each feature | `"features"` |

For the full xarray Dataset structure see [Data Format Requirements](data-requirements.md).
