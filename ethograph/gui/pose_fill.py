"""Fill backends: turn a handful of labelled frames into every frame.

One protocol, three implementations, chosen in the labelling dialog:

- :class:`SplineBackend` — monotone cubic interpolation, no new dependencies.
  Ignores pixels entirely and is the yardstick the others must beat.
- :class:`OpticalFlowBackend` — Lucas-Kanade forward/backward (``opencv-python-headless``).
- :class:`CoTrackerBackend` — CoTracker3 point tracking (``ethograph[co-tracker]``).

All three share the same invariant, asserted by the tests: **anchor frames come
back exactly as they were labelled.** Pixel-based backends seed missing points
from a spline pre-pass, so partially labelled anchors (beak on some frames, tail
on others) work without a shared frame list.

Backends track independent *points* and know nothing about the individual /
keypoint hierarchy above them: :meth:`~ethograph.gui.pose_annotate.KeypointStore.flat_anchors`
flattens one row per ``(individual, keypoint)`` pair before the fill, and
``set_fill_from_flat`` restores the shape afterwards. Multi-individual labelling
therefore needs nothing from this module beyond more rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import PchipInterpolator

#: ``progress(fraction) -> keep_going``; backends bail out when it returns False.
Progress = Callable[[float], bool]

#: Frames over which spline confidence decays to 1/e away from an anchor.
CONFIDENCE_DECAY_FRAMES = 10.0

#: Pixels of forward/backward disagreement that costs a factor 1/e of confidence.
DISAGREEMENT_SCALE = 10.0


@runtime_checkable
class FillBackend(Protocol):
    """Fills every frame from a sparse set of labelled ones."""

    name: str
    requires_video: bool

    def fill(
        self,
        anchors: dict[int, np.ndarray],
        n_frames: int,
        frames: object | None,
        progress: Progress,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positions, confidence)`` for every frame.

        ``anchors`` maps frame index to an ``(n_points, 2)`` array of
        ``(x, y)`` with ``NaN`` for unlabelled points. ``frames`` is a frame
        source indexable by frame index and slice (see :class:`VideoFrameSource`);
        it is ``None`` for backends with ``requires_video = False``.
        """


def _no_progress(_fraction: float) -> bool:
    return True


def _n_points(anchors: dict[int, np.ndarray]) -> int:
    return len(next(iter(anchors.values())))


def _apply_anchors(filled: np.ndarray, confidence: np.ndarray, anchors: dict[int, np.ndarray]) -> None:
    """Copy anchors through verbatim — the invariant every backend must hold."""
    n_frames = filled.shape[0]
    for frame, points in anchors.items():
        if not 0 <= frame < n_frames:
            continue
        labelled = ~np.isnan(points[:, 0])
        filled[frame][labelled] = points[labelled]
        confidence[frame][labelled] = 1.0


def _empty(n_frames: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((n_frames, n_points, 2), np.nan, dtype=np.float64),
        np.zeros((n_frames, n_points), dtype=np.float64),
    )


# ----------------------------------------------------------------------
# Spline
# ----------------------------------------------------------------------


class SplineBackend:
    """Per-point monotone cubic (PCHIP) interpolation over its own anchors.

    Outside the anchored span the nearest anchor value is held rather than
    extrapolated — a cubic run past its data produces confident nonsense.
    Confidence decays exponentially with distance from the nearest anchor.
    """

    name = "Spline"
    requires_video = False

    def __init__(self, decay_frames: float = CONFIDENCE_DECAY_FRAMES):
        self._decay = float(decay_frames)

    def fill(self, anchors, n_frames, frames=None, progress: Progress = _no_progress):
        if not anchors:
            raise ValueError("No labelled frames — label at least one frame before filling.")
        n_points = _n_points(anchors)
        filled, confidence = _empty(n_frames, n_points)
        grid = np.arange(n_frames, dtype=np.float64)

        for k in range(n_points):
            if not progress(k / n_points):
                break
            labelled = sorted(f for f, points in anchors.items() if not np.isnan(points[k, 0]))
            labelled = [f for f in labelled if 0 <= f < n_frames]
            if not labelled:
                continue
            xy = np.array([anchors[f][k] for f in labelled], dtype=np.float64)
            if len(labelled) == 1:
                filled[:, k, :] = xy[0]
            else:
                knots = np.asarray(labelled, dtype=np.float64)
                clamped = np.clip(grid, knots[0], knots[-1])
                filled[:, k, 0] = PchipInterpolator(knots, xy[:, 0])(clamped)
                filled[:, k, 1] = PchipInterpolator(knots, xy[:, 1])(clamped)
            distance = np.min(np.abs(grid[:, None] - np.asarray(labelled)[None, :]), axis=1)
            confidence[:, k] = np.exp(-distance / self._decay)

        _apply_anchors(filled, confidence, anchors)
        return filled, confidence


# ----------------------------------------------------------------------
# Shared gap machinery for the pixel-based backends
# ----------------------------------------------------------------------


def _seeded_endpoints(anchors: dict[int, np.ndarray], seed: np.ndarray, frame: int) -> np.ndarray:
    """Anchor row at *frame*, with unlabelled points taken from the spline seed."""
    points = seed[frame].copy()
    labelled = ~np.isnan(anchors[frame][:, 0])
    points[labelled] = anchors[frame][labelled]
    return points


def _blend(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    """Linear crossfade from the left track to the right one across a gap."""
    weight = np.linspace(0.0, 1.0, forward.shape[0])[:, None, None]
    return forward * (1.0 - weight) + backward * weight


class _GapBackend:
    """Base for backends that track each gap between consecutive anchor frames.

    A spline pre-pass provides positions for points that are missing on a
    gap's endpoints and covers the span before the first / after the last
    anchor, where there is nothing to track between.
    """

    name = "gap"
    requires_video = True

    def fill(self, anchors, n_frames, frames=None, progress: Progress = _no_progress):
        if not anchors:
            raise ValueError("No labelled frames — label at least one frame before filling.")
        if frames is None:
            raise ValueError(f"The {self.name} backend needs video frames.")

        seed, seed_confidence = SplineBackend().fill(anchors, n_frames, None, _no_progress)
        filled, confidence = seed.copy(), seed_confidence.copy()

        anchor_frames = [f for f in sorted(anchors) if 0 <= f < n_frames]
        gaps = list(zip(anchor_frames, anchor_frames[1:]))
        for done, (start, end) in enumerate(gaps):
            if not progress(done / len(gaps)):
                break
            if end - start < 2:
                continue
            clip = np.asarray(frames[start : end + 1])
            scale = float(getattr(frames, "scale", 1.0))
            left = _seeded_endpoints(anchors, seed, start) / scale
            right = _seeded_endpoints(anchors, seed, end) / scale

            forward, visible_forward = self._track(clip, left, 0)
            backward, visible_backward = self._track(clip, right, end - start)
            forward, backward = forward * scale, backward * scale

            filled[start : end + 1] = _blend(forward, backward)
            disagreement = np.linalg.norm(forward - backward, axis=-1)
            confidence[start : end + 1] = np.minimum(visible_forward, visible_backward) * np.exp(
                -disagreement / DISAGREEMENT_SCALE
            )

        _apply_anchors(filled, confidence, anchors)
        return filled, confidence

    def _track(self, clip: np.ndarray, points: np.ndarray, query_frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Track *points* (given at *query_frame*) across every frame of *clip*.

        Returns ``(positions (T, N, 2), visibility (T, N))`` in clip pixels.
        """
        raise NotImplementedError


# ----------------------------------------------------------------------
# Optical flow
# ----------------------------------------------------------------------


class OpticalFlowBackend(_GapBackend):
    """Lucas-Kanade pyramidal flow, tracked forward and backward per gap.

    Real-time on CPU and a useful fallback where torch cannot be installed.
    Requires ``opencv-python-headless`` — plain ``opencv-python`` ships Qt
    plugins that conflict with PyQt6.
    """

    name = "Optical flow"
    requires_video = True

    def __init__(self, window: int = 21, levels: int = 3):
        self._window = int(window)
        self._levels = int(levels)

    def _track(self, clip, points, query_frame):
        import cv2

        n = len(points)
        positions = np.full((len(clip), n, 2), np.nan, dtype=np.float32)
        visibility = np.zeros((len(clip), n), dtype=np.float64)
        positions[query_frame] = points
        visibility[query_frame] = 1.0

        gray = [cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY) for f in clip]
        params = dict(
            winSize=(self._window, self._window),
            maxLevel=self._levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        for direction in (1, -1):
            current = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
            alive = np.ones(n, dtype=bool)
            index = query_frame
            while 0 <= index + direction < len(clip):
                nxt = index + direction
                moved, status, _ = cv2.calcOpticalFlowPyrLK(gray[index], gray[nxt], current, None, **params)
                status = status.reshape(-1).astype(bool) & ~np.any(np.isnan(moved.reshape(-1, 2)), axis=1)
                alive &= status
                current = np.where(status[:, None, None], moved, current)
                positions[nxt] = current.reshape(-1, 2)
                visibility[nxt] = alive.astype(np.float64)
                index = nxt
        return positions.astype(np.float64), visibility


# ----------------------------------------------------------------------
# CoTracker3
# ----------------------------------------------------------------------


class CoTrackerBackend(_GapBackend):
    """CoTracker3 point tracking, forward from the left anchor and backward
    from the right, blended linearly across the gap.

    Because gaps are short (~10 frames at the recommended anchor density)
    drift is bounded and no test-time optimisation is needed — that is a
    GPU-only technique and is deliberately not implemented. Cost is dominated
    by frame feature extraction rather than point count, so tracking 20
    keypoints costs about what 3 do.
    """

    name = "CoTracker3"
    requires_video = True

    def __init__(self, predictor, device: str | None = None):
        self._predictor = predictor
        self._device = device or resolve_device()

    def _track(self, clip, points, query_frame):
        import torch

        video = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2).float()[None]
        video = video.to(self._device)
        queries = np.column_stack([np.full(len(points), query_frame, dtype=np.float32), points.astype(np.float32)])
        queries = torch.from_numpy(queries)[None].to(self._device)

        with torch.no_grad():
            tracks, visibility = self._predictor(
                video,
                queries=queries,
                backward_tracking=query_frame > 0,
            )
        return (
            tracks[0].cpu().numpy().astype(np.float64),
            visibility[0].cpu().numpy().astype(np.float64),
        )


def resolve_device(preferred: str | None = None) -> str:
    """Pick the best available torch device: CUDA, then Apple MPS, then CPU.

    CPU is the *expected* case for this feature, not the only one — a user with
    a GPU should get it without configuring anything, so nothing here hardcodes
    ``"cpu"``. Pass *preferred* to force a device; it is honoured only if torch
    reports it as usable, otherwise this falls back and the caller runs on CPU
    rather than crashing mid-fill.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    available = ["cpu"]
    if torch.cuda.is_available():
        available.insert(0, "cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        available.insert(0 if "cuda" not in available else 1, "mps")

    if preferred:
        root = preferred.split(":")[0]
        return preferred if root in available else available[0]
    return available[0]


#: CoTracker3 offline weights (~97 MB). Fetched once into the checkpoint dir so
#: installing the backend stays a single pip command — the model itself has no
#: PyPI release and cannot ship weights through the dependency resolver.
COTRACKER_CHECKPOINT_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
COTRACKER_CHECKPOINT_NAME = "scaled_offline.pth"

#: Pinned so the install is reproducible — the repo has no PyPI release, and an
#: unpinned branch is exactly the moving target we avoid ``torch.hub`` for.
COTRACKER_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"

#: The single install command. Kept here so the GUI hint, the docs and the error
#: messages cannot drift apart.
#: One explicit command — there is no ``[co-tracker]`` extra. cotracker has no
#: PyPI release and declares no dependencies (not even torch), so both halves
#: have to be named anyway; ``--torch-backend=auto`` picks up a GPU, which the
#: CPU-only Windows wheels on PyPI would otherwise silently ignore.
COTRACKER_INSTALL_HINT = (
    "uv pip install --torch-backend=auto torch "
    f'"cotracker @ git+https://github.com/facebookresearch/co-tracker.git@{COTRACKER_COMMIT}"'
)


def cotracker_checkpoint_dir() -> Path:
    """Where CoTracker3 weights are expected: ``~/.ethograph/models/cotracker``."""
    from ethograph.utils.paths import ethograph_home

    return ethograph_home() / "models" / "cotracker"


def download_cotracker_checkpoint(progress: Progress | None = None) -> Path:
    """Fetch the CoTracker3 weights into the checkpoint dir.

    A plain HTTPS download — deliberately *not* ``torch.hub.load``, which needs
    GitHub reachable, can prompt interactively (hanging the Qt event loop) and
    tracks a moving branch. Downloads to a ``.part`` file and renames only on
    success, so an interrupted fetch never leaves weights that load as garbage.
    """
    import urllib.request

    directory = cotracker_checkpoint_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / COTRACKER_CHECKPOINT_NAME
    partial = target.with_suffix(target.suffix + ".part")

    with urllib.request.urlopen(COTRACKER_CHECKPOINT_URL) as response:  # noqa: S310 - fixed https URL
        total = int(response.headers.get("content-length") or 0)
        done = 0
        with open(partial, "wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                done += len(chunk)
                if progress is not None and not progress(done / total if total else 0.0):
                    partial.unlink(missing_ok=True)
                    raise RuntimeError("Checkpoint download cancelled.")
    partial.replace(target)
    return target


def find_cotracker_checkpoint(explicit: str | Path | None = None) -> Path | None:
    """Locate CoTracker3 weights, or ``None`` if they are not downloaded."""
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    directory = cotracker_checkpoint_dir()
    if not directory.is_dir():
        return None
    # Prefer the offline (whole-clip) model — gaps here are short clips.
    for pattern in ("scaled_offline.pth", "*offline*.pth", "*.pth"):
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_cotracker_predictor(
    checkpoint: str | Path | None = None,
    device: str | None = None,
    progress: Progress | None = None,
) -> object:
    """Construct a ``CoTrackerPredictor`` on the best available device.

    Weights are downloaded on first use if absent, so installing the backend
    stays one pip command. Never ``torch.hub.load``: hub needs GitHub reachable,
    can prompt interactively (hanging the Qt event loop) and tracks a moving
    branch.

    Never passes ``checkpoint=None`` through to ``CoTrackerPredictor`` — that
    builds an *unloaded* network which returns confident nonsense with no error.
    """
    from cotracker.predictor import CoTrackerPredictor

    resolved = find_cotracker_checkpoint(checkpoint)
    if resolved is None:
        if checkpoint is not None:
            raise FileNotFoundError(f"No CoTracker3 checkpoint at {checkpoint}")
        resolved = download_cotracker_checkpoint(progress)

    predictor = CoTrackerPredictor(checkpoint=str(resolved))
    return predictor.to(resolve_device(device))


# ----------------------------------------------------------------------
# Availability
# ----------------------------------------------------------------------


@dataclass
class BackendInfo:
    key: str
    label: str
    available: bool
    hint: str = ""


def _module_available(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def available_backends() -> list[BackendInfo]:
    """Describe every backend so the dialog can grey out the missing ones."""
    # torch and cotracker install together but resolve separately (cotracker has
    # no PyPI release), so report whichever is actually missing. Weights are NOT
    # a precondition — they download on first use.
    installed = _module_available("torch") and _module_available("cotracker")
    if not installed:
        cotracker_hint = COTRACKER_INSTALL_HINT
    elif find_cotracker_checkpoint() is None:
        cotracker_hint = "~97 MB of weights will download on first use"
    else:
        cotracker_hint = ""
    # Showing the resolved device makes it obvious whether the GPU was picked
    # up — a silent CPU fallback on a CUDA machine is the confusing case.
    label = f"CoTracker3 ({resolve_device()})" if installed else "CoTracker3"
    return [
        BackendInfo("spline", "Spline (no extra dependencies)", True),
        BackendInfo(
            "flow",
            "Optical flow (OpenCV)",
            _module_available("cv2"),
            "pip install opencv-python-headless",
        ),
        BackendInfo("cotracker", label, installed, cotracker_hint),
    ]


def build_backend(
    key: str,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    progress: Progress | None = None,
) -> FillBackend:
    """Instantiate a backend by key, importing heavy dependencies only now.

    ``device=None`` auto-detects (CUDA → MPS → CPU) via :func:`resolve_device`.
    ``progress`` reports the one-time CoTracker weight download; call this from
    inside the progress dialog so a ~97 MB fetch is visible and cancellable.
    """
    if key == "spline":
        return SplineBackend()
    if key == "flow":
        return OpticalFlowBackend()
    if key == "cotracker":
        resolved = resolve_device(device)
        predictor = load_cotracker_predictor(checkpoint, resolved, progress)
        return CoTrackerBackend(predictor, device=resolved)
    raise ValueError(f"Unknown fill backend {key!r}")


# ----------------------------------------------------------------------
# Frame source
# ----------------------------------------------------------------------


class VideoFrameSource:
    """Lazily decoded RGB frames, indexable by frame index and slice.

    Decoding the whole video into memory is not viable for the videos this GUI
    targets, so frames are decoded on demand with PyAV. Access is assumed to be
    broadly forward (gaps are visited in order); a backward request re-seeks.

    ``max_side`` downscales during decode — the single biggest CPU speedup for
    the tracking backends, and near-free in accuracy at this anchor density.
    :attr:`scale` converts decoded pixels back to source pixels.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        n_frames: int,
        max_side: int | None = None,
        start_frame: int = 0,
    ):
        import av

        if fps <= 0:
            raise ValueError("fps must be positive — read it from the video, do not default it.")
        self._path = str(path)
        self._fps = float(fps)
        self._n_frames = int(n_frames)
        #: Video frame that index 0 of this source maps to — lets callers work
        #: in trial frames while decoding stays in video frames.
        self._start_frame = int(start_frame)
        self._container = av.open(self._path)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"

        width, height = self._stream.codec_context.width, self._stream.codec_context.height
        longest = max(width, height)
        if max_side and longest > max_side:
            self.scale = longest / float(max_side)
            self._size = (
                max(2, round(width / self.scale / 2) * 2),
                max(2, round(height / self.scale / 2) * 2),
            )
        else:
            self.scale = 1.0
            self._size = (width, height)

    def __len__(self) -> int:
        return self._n_frames

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._n_frames)
            if step != 1:
                raise ValueError("VideoFrameSource only supports contiguous slices.")
            return self._decode(start, stop)
        return self._decode(int(key), int(key) + 1)[0]

    def _decode(self, start: int, stop: int) -> np.ndarray:
        start, stop = start + self._start_frame, stop + self._start_frame
        decoded: dict[int, np.ndarray] = {}
        self._seek(start)
        for frame in self._container.decode(self._stream):
            index = self._frame_index(frame)
            if index >= stop:
                break
            if index >= start:
                decoded[index] = frame.reformat(width=self._size[0], height=self._size[1], format="rgb24").to_ndarray()
        if not decoded:
            raise ValueError(f"No frames decoded for [{start}, {stop}) in {self._path}")

        # Timestamp rounding can skip an index; hold the previous frame so the
        # returned clip always has one entry per requested frame.
        out, previous = [], decoded[min(decoded)]
        for index in range(start, stop):
            previous = decoded.get(index, previous)
            out.append(previous)
        return np.stack(out)

    def _seek(self, frame_index: int) -> None:
        target = int(frame_index / self._fps / float(self._stream.time_base))
        offset = int(self._stream.start_time or 0)
        self._container.seek(max(0, target + offset), stream=self._stream, backward=True)

    def _frame_index(self, frame) -> int:
        pts = frame.pts if frame.pts is not None else 0
        pts -= int(self._stream.start_time or 0)
        return int(round(float(pts * self._stream.time_base) * self._fps))

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> VideoFrameSource:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
