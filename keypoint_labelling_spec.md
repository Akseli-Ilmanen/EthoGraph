# Keypoint labelling + CoTracker3 fill — implementation spec

Label a handful of frames, let a pretrained point tracker fill the rest. No
training, no GPU required, no DeepLabCut dependency.

**Scope**: single video, 2D, one or more individuals. Multi-camera is out of
scope for v1.

---

## 1. User flow

1. **Tools ▸ Keypoint labelling…** opens the labelling dialog.
2. User types keypoint names (or loads a skeleton template) and, for more than
   one animal, adds individuals — one shared schema, one instance per
   individual, as in SLEAP.
3. Labelling mode activates on the video canvas. Click to place the active
   keypoint, `Tab` cycles to the next, `Backspace` deletes the one under the
   cursor. Navigate frames with the existing playhead.
4. Sidebar counter shows `frames labelled: 12 / recommended ~20`.
5. **Fill remaining frames** runs CoTracker3 between consecutive labelled
   frames.
6. Filled points render through the normal pose overlay. Low-confidence points
   are hidden by the existing confidence threshold spinbox.
7. User corrects the worst frames and presses Fill again — corrected frames
   become new anchors.
8. **Export** writes a movement-compatible dataset (and optionally DLC
   `CollectedData` h5 — see §7).

Labelling a few keypoints across *many* frames beats labelling all keypoints on
one frame — the annotation-strategy ablation in Pan et al. (CV4Animals 2025)
found ~54 vs ~45 δ_avg. The dialog should say so.

---

## 2. New modules

Everything goes in `gui/`, following the existing `pose_*` prefix. No new
package.

```
ethograph/
  gui/
    pose_annotate.py           # KeypointStore + store_to_movement_ds
    pose_fill.py               # SplineBackend, CoTrackerBackend
    dialog_pose_labelling.py   # keypoint names, backend choice, fill + progress
    pose_edit_mixin.py         # canvas click/drag editing
```

Why `gui/` and not `io/` or `skeleton/`:

- `gui/pose_convert.py`, `pose_overlay.py` and `pose_render.py` already own the
  pose pipeline, and `pose_render.py` already mixes data logic with display
  (`PoseRenderData`, `pose_render_to_movement_ds`). Annotation state and fill
  backends sit in exactly that band. Colocating them keeps one pose story in
  one place.
- `io/` is session/file plumbing — loaders, NWB, trialtree, time models. Live
  annotation state is not that. If a `CollectedData` writer ever lands, `io/`
  is arguably its home, but not the store.
- `skeleton/` is the static keypoint *schema* (templates, config, shapes,
  renderers). Per-frame coordinates are a different thing, and mixing them
  muddies a currently clean package.
- **Not** a new `labelling/` package: `labels/` already exists and means
  behavioural annotation (boris, crowsetta, intervals, tsv_store). Two packages
  one letter apart meaning different things is a trap.

`dialog_pose_labelling.py` matches the existing `dialog_pose_video_matcher.py`
and `dialog_skeleton_editor.py` naming; `pose_edit_mixin.py` matches
`label_drawing_mixin.py`.

Export lives in `pose_annotate.py` rather than its own module, mirroring
`pose_render_to_movement_ds` in `pose_render.py`.

Nothing existing moves. `top_bar.py`, `widgets_data.py`, `pygfx_video.py` and
`pose_render.py` get small additions only.

---

## 3. Data model

`pose_annotate.py` owns all state. The GUI never mutates arrays directly.

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class KeypointStoreError(Exception):
    """Base for keypoint store failures."""


class UnknownKeypointError(KeypointStoreError):
    """Raised when a keypoint name is not in the store's schema."""


@dataclass
class KeypointStore:
    keypoint_names: list[str]
    n_frames: int
    individual_names: list[str] = field(default_factory=lambda: ["individual_0"])
    anchors: dict[int, np.ndarray] = field(default_factory=dict)
    filled: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def set_point(self, frame: int, keypoint: str, xy: tuple[float, float], individual: str | None = None) -> None: ...
    def clear_point(self, frame: int, keypoint: str, individual: str | None = None) -> None: ...
    def anchor_frames(self) -> list[int]: ...
    def positions(self, frame: int) -> np.ndarray: ...
    def positions_for(self, frame: int, individual: str | None = None) -> np.ndarray: ...
    def flat_anchors(self) -> dict[int, np.ndarray]: ...
    def undo(self) -> None: ...
```

Two levels, like SLEAP's skeleton/instance split: one keypoint schema shared by
every individual, and each individual an instance of it on a given frame.
``individual=None`` means the first (usually only) one, so single-individual
labelling reads exactly as it did before the axis existed.

Conventions worth fixing now, because they cause silent bugs later:

- Anchors are `(n_individuals, n_keypoints, 2)` in **`(x, y)` pixel coordinates
  of the source video**, `NaN` where unlabelled. Note
  `pose_convert.poses_ds_to_points` emits `(track_id, frame, y, x)` — the axis
  swap lives in `pose_annotate.py` and nowhere else.
- `filled` is `(n_frames, n_individuals, n_keypoints, 2)`; anchor frames are
  copied through verbatim, never overwritten by the model.
- `confidence` is `(n_frames, n_individuals, n_keypoints)` in `[0, 1]`; anchors
  are `1.0`.
- Names are singular wherever they name a thing — dataset dims `keypoint` /
  `individual`, matching movement ≥0.17 and the rest of the codebase. Plural is
  for Python containers and counts only.

A frame counts as an anchor if **any** point on it is labelled. Partially
labelled anchors are normal (the user labels the beak on some frames and the
tail on others) — the backend must handle per-point anchor sets, not a
single shared frame list.

Backends stay hierarchy-blind: `flat_anchors()` hands them one row per
`(individual, keypoint)` pair and `set_fill_from_flat()` restores the shape, so
§4 needs nothing for multi-individual beyond more rows.

---

## 4. Fill backends

One protocol, three implementations, selected in the dialog. This is the
important seam: it lets you ship without torch and add it later.

```python
from typing import Protocol

import numpy as np


class FillBackend(Protocol):
    name: str
    requires_video: bool

    def fill(
        self,
        anchors: dict[int, np.ndarray],
        n_frames: int,
        frames: np.ndarray | None,
        progress: Callable[[float], bool],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (positions, confidence) for every frame."""
```

`progress` returns `False` when the user cancels; backends check it between
gaps and bail out.

### 4.1 SplineBackend (default, zero new dependencies)

`scipy.interpolate.PchipInterpolator` per keypoint over its own anchor frames.
Ignores pixels entirely. At 1-in-10 density with smooth motion this is a
genuinely strong baseline, and it is the yardstick every other backend must
beat. Confidence decays with distance from the nearest anchor.

### 4.2 CoTrackerBackend (`ethograph[co-tracker]`)

For each consecutive anchor pair, track forward from the left anchor and
backward from the right, then blend linearly by position within the gap.
Because gaps are ~10 frames, drift is bounded and no test-time optimisation is
needed — that is a GPU-only technique and is deliberately not implemented.

```python
import numpy as np
import torch

from cotracker.predictor import CoTrackerPredictor


class CoTrackerBackend:
    name = "CoTracker3"
    requires_video = True

    def __init__(self, predictor: CoTrackerPredictor, device: str = "cpu", max_side: int = 512):
        self._predictor = predictor
        self._device = device
        self._max_side = max_side

    def fill(self, anchors, n_frames, frames, progress):
        filled = np.full((n_frames, len(next(iter(anchors.values()))), 2), np.nan, np.float32)
        confidence = np.zeros(filled.shape[:2], np.float32)
        for index, points in anchors.items():
            filled[index], confidence[index] = points, 1.0

        gaps = list(zip(sorted(anchors), sorted(anchors)[1:]))
        for done, (start, end) in enumerate(gaps):
            if not progress(done / len(gaps)):
                break
            clip = self._to_tensor(frames[start : end + 1])
            forward, visible_fwd = self._track(clip, anchors[start], 0)
            backward, visible_bwd = self._track(clip, anchors[end], end - start)
            weight = np.linspace(0.0, 1.0, end - start + 1)[:, None, None]
            filled[start : end + 1] = forward * (1 - weight) + backward * weight
            disagreement = np.linalg.norm(forward - backward, axis=-1)
            confidence[start : end + 1] = np.minimum(visible_fwd, visible_bwd) * np.exp(
                -disagreement / 10.0
            )
        return filled, confidence
```

`_track` builds `(t, x, y)` queries, calls the predictor, and rescales
coordinates back from `max_side`. The predictor returns `(tracks, visibility)`
— use `visibility` as the primary confidence and forward/backward disagreement
as a secondary signal; frames where both are poor are the ones to show the user
first.

Downscaling to a 512px longest side is the single biggest CPU speedup and costs
almost nothing in accuracy at this anchor density. Cost is dominated by frame
feature extraction, not by point count, so tracking 20 keypoints costs about
what 3 do.

### 4.3 OpticalFlowBackend (optional, `opencv-python-headless`)

`cv2.calcOpticalFlowPyrLK` forward and backward, same blending. Real-time on
CPU, useful as a fallback where torch cannot be installed. Use
**`opencv-python-headless`** — plain `opencv-python` ships Qt plugins that
conflict with PyQt6.

---

## 5. Canvas editing

The only genuinely new interaction code. `pygfx_video.py` already registers a
`pointer_down` handler that emits `clicked`; extend rather than replace it, so
panel-focus behaviour is preserved.

```python
def _on_pointer_down(self, event=None) -> None:
    self.clicked.emit()
    if self._label_mode is not None and event is not None:
        self._label_mode.handle_click(*self._screen_to_image(event.x, event.y))
```

Needed pieces:

- `_screen_to_image(x, y)` — unproject canvas coords to texture pixels via the
  pygfx camera. Must work under zoom/pan, which the controller already supports.
- Hit test — nearest existing point within a radius, for select/drag/delete.
- `pointer_move` + `pointer_up` for dragging.
- A distinct visual state for anchors vs filled points. `pose_overlay` already
  colours per track; add a marker-size or outline distinction so anchors are
  obviously different from predictions.

`_StaticImagePlot` gets the same handler so labelling works in pose-only mode.

---

## 6. GUI wiring

**Tools menu** (`top_bar.py::_build_tools_menu`) — one entry, following the
existing `_popup_section` pattern:

```python
menu.addAction("Keypoint labelling…", self._open_keypoint_labelling)
```

**Pose section** (`widgets_data.py::_create_pose_section`) — a new
`QGroupBox("Label keypoints")` below the existing "Create / edit skeleton…"
button, containing: an editable keypoint-name list, a **Start labelling**
toggle, an anchor-count label, a backend combo, and **Fill remaining frames**.
Reuse `dialog_select_template.py` to seed names from a skeleton template.

**Fill execution** — wrap the backend in `BusyProgressDialog.execute` so the UI
stays responsive and cancel works. If exposing backend parameters (max_side,
blend width), use `FunctionSpec` from `dialog_function_params.py` rather than
hand-rolling a form.

**app_state** — add `labelling_keypoints: list[str]`,
`labelling_individuals: list[str]` and `labelling_backend: str` alongside the
existing `keypoints` field. Anchors are
project data, not settings: persist them to a sidecar file next to the video,
not into app_state.

---

## 7. Export

Build a movement dataset and everything downstream (overlay, filtering,
kinematics, NWB) works unchanged:

```python
def store_to_movement_ds(store: KeypointStore, fps: float) -> xr.Dataset: ...
```

Dims `(time, space, keypoint, individual)` — singular, as movement ≥0.17 names
them — with `confidence` populated from the store. More than one individual
also switches the DLC export to the multi-animal layout (which carries DLC's
own `individuals` column level). Feed it through the existing `PoseRenderData` path — filled
keypoints should be indistinguishable from imported DLC predictions.

**Optional, drop if not wanted**: a second exporter writing DeepLabCut's
`CollectedData_<scorer>.h5` — a single file (their naming convention, not a
directory) that a DLC user can train from elsewhere. Pure interop; it keeps
EthoGraph out of the training business, which is where the support burden
lives. Nothing else in the design depends on it.

---

## 8. Dependencies

```toml
[project.optional-dependencies]

# CoTracker3 point-tracking backend for pose keypoint fill (CPU is fine).
# torch is declared here because this backend is unusable without it; on Linux
# pip resolves to the multi-GB CUDA build by default, so document the CPU index.
co-tracker = [
  "torch>=2.0",
  "cotracker @ git+https://github.com/facebookresearch/co-tracker.git@<PIN_COMMIT>",
]
```

Installed as `pip install ethograph[co-tracker]` / `uv sync --extra co-tracker`.

- Do **not** add to `[gui]`. Base install stays light and the spline backend
  covers the default path.
- Note the divergence from the existing `[model]` extra, which deliberately
  omits torch so users install a CUDA-matched build first. That is the right
  call there (training needs a GPU); here CPU is the expected case, so
  declaring torch keeps the one-line install working. If that inconsistency
  grates, drop torch from this extra too and document it alongside `[model]`.
- CPU wheel index, for the docs:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
  Windows PyPI wheels are already CPU-only, so this only bites Linux users.
  For a CUDA build under uv, add an `[[tool.uv.index]]` entry with
  `explicit = true` and a `[tool.uv.sources]` pin for torch.
- Do **not** use `torch.hub.load`: it needs GitHub reachable at runtime, can
  prompt interactively (hangs the Qt event loop), and tracks a moving branch.
- Import `cotracker` lazily inside `pose_fill.py`, guarded, so the dialog can
  grey out the backend with an install hint instead of crashing.

**Licence**: CoTracker's README states the majority of the project is CC-BY-NC
(the Apache-2.0 mention covers vendored TAP-Vid/LocoTrack code, not CoTracker
itself). CC-BY-NC is not OSI open-source and some institutions reject it on
review. Verify against the repo's LICENSE file, state it in the docs, and keep
it out of the default install. If it blocks adoption, TAPIR/LocoTrack/TAPNext
are Apache-2.0 and slot into the same backend protocol — TAPNext++ (CVPR 2026)
currently leads TAP-Vid at much lower latency, which matters more than accuracy
for CPU use.

---

## 9. Tests

```
tests/test_unit/
  test_pose_annotate.py      # set/clear/undo, partial anchors, NaN handling,
                             # store -> movement ds round-trip, x/y axis order
  test_pose_fill.py          # protocol conformance, anchor preservation
```

Backend tests use a fake predictor returning known tracks — no torch needed in
CI. Assert the invariant that matters: **anchor frames are returned exactly as
labelled**, for every backend.

For accuracy, build a held-out harness: take a densely labelled video, keep
every Nth frame as anchors, fill, and measure median pixel error against the
rest. That answers "is CoTracker worth the dependency over the spline" with
data instead of argument. BADJA is the established public benchmark for exactly
this task (keypoints on ~1 in 5 frames, evaluated as keypoint-transfer
accuracy).

---

## 10. Build order

1. `KeypointStore` + tests. No GUI.
2. `SplineBackend` + `store_to_movement_ds`. Feature is now end-to-end useful with zero
   new dependencies.
3. Canvas editing + dialog. The bulk of the work.
4. `CoTrackerBackend` behind the extra.
5. Held-out harness; decide whether the extra earns its place as the default.

Steps 1–3 ship a complete feature. Step 4 is an upgrade, not a prerequisite —
if torch proves too heavy for the teaching context, the spline path stands on
its own.
