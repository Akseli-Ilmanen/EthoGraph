# Changepoints

Changepoints mark transitions in time-series data — moments where the signal changes character. EthoGraph uses changepoints in two ways:

1. **Visual guides** — displayed on plots to help you see where behaviours start and stop
2. **Label correction** — snapping hand-drawn label boundaries to the nearest changepoint for consistency

There are three detection families, each suited to different data types.

---

## Kinematic Changepoints

For body-tracking features (speed, acceleration, joint angles, etc.) sampled at moderate rates (typically 30–120 Hz). These are stored as dense binary arrays sharing the feature's time dimension.

### Troughs (local minima)

Finds local minima in the signal using `scipy.signal.find_peaks` applied to the negated signal. Troughs in speed often correspond to moments where an animal pauses or reverses direction.

Parameters are passed directly to [`scipy.signal.find_peaks`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html): `height`, `distance`, `prominence`, `width`, etc.

### Turning points

A custom algorithm that identifies the boundaries of peak regions rather than the peaks themselves. The idea is that behaviour transitions happen not at the peak of a movement, but where the animal starts or stops accelerating.

![turning points illustration](media/changepoints2.png)

The algorithm works in four steps:

1. **Compute the gradient** of the signal and find all indices where `|gradient| < threshold` — these are candidate turning points (near-stationary regions, shown as light green dots).
2. **Find peaks** in the original signal using `scipy.signal.find_peaks` with `prominence` and `width` parameters (red triangles).
3. **For each peak**, select the closest candidate turning point to its left and right (green circles). These define the boundaries of the peak region.
4. **Filter by `max_value`**: any candidate turning point where the signal exceeds `max_value` is discarded (purple dashed line). This prevents selecting turning points on high plateaus.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 1.0 | Maximum absolute gradient to qualify as a turning point. Lower = only very flat regions. |
| `max_value` | None | Discard turning points where signal exceeds this value. |
| `prominence` | 0.5 | Minimum peak prominence (passed to `find_peaks`). |
| `distance` | 2 | Minimum distance between peaks (passed to `find_peaks`). |

All kinematic changepoints also include NaN-boundary markers — transitions between valid data and NaN gaps are automatically added as changepoints.

### Usage

1. Select a feature in the Data Controls (e.g. `speed`)
2. Open the **Kinematic CPs** panel
3. Choose a method (`troughs` or `turning_points`)
4. Click **Configure...** to adjust parameters
5. Click **Detect**

The changepoints are stored in the dataset as `{feature}_{method}` (e.g. `speed_troughs`) and persist when you save.

---

## Audio Changepoints

For audio data or high-sample-rate periodic signals. Audio changepoints are stored as onset/offset time pairs (in seconds) rather than dense binary arrays — a dense array at 44 kHz would be impractically large.

Four methods are available, drawn from two libraries:

### VocalPy methods

Reference: [VocalPy documentation](https://vocalpy.readthedocs.io/)

**Mean-squared energy** (`meansquared`): Computes a smoothed energy envelope via mean-squared amplitude, then thresholds to find vocal segments. Simple and fast.

**AVA** (`ava`): The segmentation method from the Animal Vocalization Analysis pipeline. Uses a spectrogram-based approach with multiple threshold levels. Automatically computes spectrogram range if not provided.

### VocalSeg methods

Reference: [VocalSeg (Sainburg et al., 2020)](https://github.com/timsainb/vocalization-segmentation)

**Dynamic thresholding** (`vocalseg`): Adaptive threshold segmentation that adjusts to local spectral energy. Good for signals with varying background noise.

**Continuity filtering** (`continuity`): Extends dynamic thresholding with temporal continuity constraints to merge fragmented detections.

### Usage

1. Select **Audio Waveform** as the feature (or any feature — the method will use the appropriate sample rate)
2. Open the **Audio CPs** panel
3. Choose a method and click **Configure...** to adjust parameters
4. Click **Detect**

Detected onsets and offsets are drawn as vertical lines on the plot and stored in the dataset as `audio_cp_onsets` / `audio_cp_offsets`.

---

## Ruptures

Reference: [Ruptures (Truong et al., 2020)](https://centre-borelli.github.io/ruptures-docs)

General-purpose changepoint detection using the `ruptures` library. Five search methods are available:

| Method | Description |
|--------|-------------|
| **Pelt** | Penalty-based, fast. Good when the number of changepoints is unknown. |
| **Binseg** | Binary segmentation. Fast recursive splitting. |
| **BottomUp** | Bottom-up merging of segments. |
| **Window** | Sliding window approach. |
| **Dynp** | Dynamic programming. Optimal but slow on long signals. |

Each method supports a cost model (`l2`, `l1`, `rbf`, etc.) and parameters like `min_size`, `jump`, and either `pen` (penalty) or `n_bkps` (fixed number of breakpoints).

**Note:** Ruptures detection has not been tested as extensively as the kinematic and audio methods. Results may vary — check visually and adjust parameters as needed. Ruptures cannot be applied directly to raw audio waveforms (too large); use a derived feature or the Audio CPs panel instead.

### Usage

1. Select a feature in Data Controls
2. Open the **Ruptures** panel
3. Choose a method and click **Configure...**
4. Click **Detect**

---

## Changepoint Correction

Once changepoints are detected, they can be used to refine label boundaries. The correction pipeline snaps hand-drawn label edges to nearby changepoints, producing more consistent annotations.

### How it works

The correction runs four steps in sequence:

1. **Purge short intervals** — remove labels shorter than the minimum duration
2. **Stitch gaps** — merge adjacent same-label intervals separated by a small gap
3. **Snap boundaries** — move each label's start/end to the nearest changepoint, constrained by maximum expansion/shrink limits
4. **Purge short intervals** (again) — snapping may create new short intervals

### Parameters

| Parameter | Description |
|-----------|-------------|
| **Min label length** | Labels shorter than this (in samples) are removed. |
| **Stitch gap** | Maximum gap (samples) between same-label segments to merge. |
| **Max expansion** | How far a boundary can move outward toward a changepoint. |
| **Max shrink** | How far a boundary can move inward toward a changepoint. |
| **Per-label thresholds** | Override the global min label length for specific label classes. |

### Automatic vs manual correction

- **Checkbox "Changepoint correction"**: When enabled, label boundaries are snapped to changepoints as you create them. When disabled, only a minimal cleanup (min length = 2) is applied.
- **Single Trial**: Applies the full correction pipeline to the current trial's labels.
- **All Trials**: Applies correction to every trial. The dataset is marked as corrected to prevent double-application.
- **Undo**: Reverts the last correction (single or all trials).

### Saving parameters

Click **Save** to write correction parameters to `configs/changepoint_settings.yaml`. Click **Load** to restore them. Parameters are also auto-loaded on startup if the file exists.
