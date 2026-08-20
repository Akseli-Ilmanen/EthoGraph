(target-calibration)=
# 4. Calibrate — optional, for trajectories in cm

Labelled and filled keypoints live in pixel coordinates of the source video.
If a few fixed physical landmarks in the scene — arena corners, a doorway, a
perch — have known real-world positions, the **Calibrate** tab turns them into
a pixel→cm map, and the export can then produce trajectories in your own cm
frame instead of pixels.

## Landmarks are not keypoints

A landmark belongs to the scene, not to an animal: it has no individual, it
never moves, and it must never join the fill span or the exported `keypoint`
dimension. So landmarks live in their own table, drawn as **diamonds** on the
video where every keypoint is a circle — and they cannot collide with the
detector's `corner_N` tag keypoints, which are body-marker corners, not scene
corners.

## The workflow

1. **Name the landmarks and give them cm coordinates** — type them into the
   table, or **Load coordinates…** from a file (see below).
2. **Click each landmark on the video**, on a few different frames. Opening the
   tab hands the pointer to calibration clicking: a click places the selected
   landmark and moves on to the next one this frame still lacks; clicking an
   existing diamond drags it; `Backspace` removes the selected landmark's click
   on this frame; `Shift`+left-drag still pans. The clicks of one landmark are
   **averaged** — the camera is assumed static for the session, so a few clicks
   on different frames simply cancel out click jitter. The smaller hollow
   diamond is that running mean: what the fit will actually use.
3. **Pick "cm (calibrated)"** in the export's *Coordinate space* box, which
   enables itself as soon as three landmarks are *ready* (cm coordinates plus
   at least one click).

The **Clicked frames** table below the landmarks mirrors the points table: one
row per frame carrying a click, one column per landmark, the playhead's row
highlighted. Clicking a row seeks the video there, clicking a landmark's cell
also makes it the one the next canvas click places, and right-click removes a
single click or a whole frame's worth.

Three landmarks fit an **affine** map — right for a top-down camera. Four or
more fit a **homography**, the exact model for any camera viewing a flat
plane, which also corrects the foreshortening of an angled view; extra
landmarks are averaged into the fit, so more is better. Either way the fitted
matrix is derived data, recomputed from the table whenever it is needed.

```{note}
A single camera calibrates a **plane**. Use landmarks roughly level with the
plane the animal moves on: the cm output means "where this point sits on the
calibration plane", and a point well above it (a head high off the ground,
say) is displaced along the camera ray, more so the more the camera is tilted.
For that reason a `z` column in a coordinates file is ignored.
```

## Loading coordinates from a file

The world layout is the stable part — the same arena serves many sessions —
while the clicked pixels drift as the camera is nudged between sessions.
**Load coordinates…** reads the cm side from a table so only the clicking is
per-session; existing landmarks keep their clicks, and re-clicking is what
absorbs the drift. Two layouts are understood:

- **One row per landmark**: columns `name` (or `landmark`), `x`, `y` — and
  optionally `z`, which is ignored.
- **One row per session**: a `session` column plus `{landmark}_x` /
  `{landmark}_y` (and ignored `_z`) columns. You are asked which session row
  to use.

## Where it is stored, and what the export writes

The landmark table — cm coordinates and clicks — is user intent, so it is
saved in `<video>.keypoints.json` beside your labels, exactly like the
detector's assignment table. A cm export maps `position` (and everything
derived from it: velocity and acceleration come out in cm/s and cm/s², the
heading is rotated into the same frame) through the fit, and records
`attrs["space_unit"] = "cm"` and the matrix itself as `attrs["pixels_to_cm"]`.
The y-flip changes meaning in cm: it mirrors your **world frame's** y axis
(y → −y), composed after the calibration — never the image's pixels, which the
fit was not made from. Untick it if your landmark coordinates already have y
pointing the way you want plots to read. The video overlay always stays in
pixels — it is drawn on the pixels.
