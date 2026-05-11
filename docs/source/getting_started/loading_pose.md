(target-loading-pose)=
# From a pose file

Use this path if you have pose estimation output from DeepLabCut, SLEAP, LightningPose, or any other tracker that produces `.h5` or `.csv` files — with or without a matching video.

The **data wizard** handles single recordings (one trial) without any scripting. For multiple trials or multiple cameras, see {doc}`multi_trial`.

---

## Steps

```{tip}
{doc}`Install EthoGraph <../getting_started/installation>` if you haven't already, then launch via shortcut or:
`conda activate ethograph && ethograph launch`

In the **I/O widget**, click **Create with own data** — the wizard opens.
```

1. Under **Single trial**, select: **1) Generate from pose file (DeepLabCut, SLEAP, ...)**
2. Click **Next** — the dialog opens
3. Set **Source software** (DeepLabCut, SLEAP, LightningPose, ...) and **Pose file** (`.h5` or `.csv`)
4. Optionally set **Video file** — frame rate is auto-detected from the video
5. Set **Output path** for the generated `session.nc`
6. Click **Generate .nc file**
7. The I/O widget auto-populates -> click **Load**

---

## What the loader computes

Beyond raw position, the loader automatically computes kinematic features for each keypoint:

- Velocity, acceleration, speed

These appear in the Feature dropdown and are useful for identifying movement onset/offset. See {ref}`Kinematic changepoints <target-kinematic-changepoints>`.

---

## Dialog fields

| Field | Notes |
|-------|-------|
| **Source software** | DeepLabCut, SLEAP, LightningPose, ... |
| **Pose file** | `.h5` or `.csv` |
| **Video file** | Optional |
| **Frame rate** | |
| **Output path** | |
