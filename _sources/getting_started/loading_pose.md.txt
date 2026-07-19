(target-loading-pose)=
# From a pose file

Use this path if you have pose estimation output from DeepLabCut, SLEAP, LightningPose, or any other tracker that produces `.h5` or `.csv` files — with or without a matching video.

Single recordings (one trial) load by **drag & drop** — no scripting, no wizard. For multiple trials or multiple cameras, see {doc}`multi_trial`.

---

## Load it — drag & drop

```{tip}
{doc}`Install EthoGraph <../getting_started/installation>` if you haven't already, then launch via shortcut or:
`conda activate ethograph && ethograph launch`
```

1. On the start page, drag your **pose file** (and optionally a **video**) onto the **Drag & drop** zone.
2. Click **Load**.

That's it. A follow-up popup appears **only** when something can't be read from the files:

- **Source software** — asked when the extension is ambiguous (`.h5` / `.csv` could be several trackers). A `.slp` is always SLEAP, so no question is needed.
- Frame rate is read from the video automatically; no prompt.

Drop several videos and pose files together (multi-camera) and EthoGraph asks you to order them so each pose file is paired with its camera.

---

## What the loader computes

Beyond raw position, the loader automatically computes kinematic features for each keypoint:

- Velocity, acceleration, speed

These appear in the Feature dropdown and are useful for identifying movement onset/offset. See {ref}`Kinematic changepoints <target-kinematic-changepoints>`.
