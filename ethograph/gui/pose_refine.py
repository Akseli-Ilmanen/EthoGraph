"""Test-time refinement: fit CoTracker3's query features to *this* video.

Out of the box CoTracker3 tracks a point by the appearance it had on the query
frame. That is why it drifts onto the wrong leg, the wrong wing, the other
animal: nothing tells it what a *beak* looks like across the recording, only
what one patch looked like on one frame.

:class:`QueryFeatureRefinement` implements the fix from Pan et al. (2025),
"Animal Pose Labeling Using General-Purpose Point Trackers" (arXiv:2506.03868,
reference implementation https://github.com/Zhuoyang-Pan/PosePAL): freeze the
whole network and optimise **only the query-point feature embedding** against
the frames the user labelled. The paper reports 49.6 -> 67.5 delta_avg on
DAVIS-Animals from that one change, and its ablation shows tuning this embedding
beats tuning the feature extractor (54.1) or the whole network (66.3) at a third
of the cost.

Three deliberate departures from the reference implementation:

**No forked CoTracker.** PosePAL passes precomputed pyramids into a patched
``forward``. Upstream's ``forward`` takes no such argument, so instead of
vendoring a CC-BY-NC fork this module wraps :meth:`get_track_feat` at runtime
and adds a trainable **residual** to what it returns. Same parameterisation,
and the paper's L1 pull-back to the original features falls out as ``|delta|``.

**Windows, not whole videos.** The paper fits one 100-frame clip in one pass.
Behaviour recordings are minutes long, so the fit iterates over short windows
spanning the labelled frames — bounded memory, and every labelled frame gets
used regardless of video length.

**Sparse supervision.** Their benchmark videos are labelled on every frame; here
only a handful are, so ``valids`` masks the loss down to the (frame, point)
pairs a human actually placed. That is what ``reduce_masked_mean`` inside
``sequence_loss`` is for.

The refinement is *per video and per point row*: ``delta[level, s, n, c]`` is
learned for flat point ``n``, so a schema edit invalidates it (see
:attr:`signature`).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ethograph.gui.pose_fill import (
    DISAGREEMENT_SCALE,
    CoTrackerBackend,
    Progress,
    SplineBackend,
    _no_progress,
    resolve_device,
)

#: Optimisation steps. PosePAL's own default is 500; the paper's evaluation uses
#: 1000. 500 converges in well under the paper's 3 minutes on a mid-range GPU and
#: keeps the fit inside a tolerable modal wait — this is a labelling loop, not a
#: benchmark run.
REFINE_STEPS = 500

#: Adam, decayed linearly across the fit (paper: 1e-3 -> 1e-5).
REFINE_LR = 1e-3
REFINE_FINAL_LR = 1e-5

#: Weight of the L1 pull-back towards the unrefined features (paper: 0.01).
REFINE_REG_WEIGHT = 0.01

#: Frames per training window. Must stay under the model's own 60-frame
#: attention window, and is the main lever on fit memory.
REFINE_WINDOW_FRAMES = 48

#: Consecutive steps spent on one window before moving to the next. The frame
#: features are cached per window, so switching costs a CNN pass over the whole
#: window — visiting each window in a short run amortises that away.
REFINE_STEPS_PER_WINDOW = 5

#: Update iterations inside the tracker during the fit. Inference uses 6; the
#: reference implementation trains with 4, which is a third less backprop for a
#: loss that already weights the last iterations most (gamma = 0.8).
REFINE_TRAIN_ITERS = 4

#: A window has to hold at least this many labelled frames to teach anything:
#: with one, the only supervised frame is the query frame itself.
MIN_WINDOW_ANCHORS = 2

#: Support points CoTracker's predictor appends to every query set. They are the
#: model's own context, not ours, so they are never given a delta.
SUPPORT_GRID_SIZE = 6


def _windows(anchor_frames: list[int], n_frames: int, length: int) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` windows, each holding >= 2 labelled frames.

    One window is anchored at each labelled frame, which puts every labelled
    frame in at least one window whenever its neighbour is within reach. Anchors
    separated by more than *length* frames yield no window between them: there is
    nothing to learn from a query with no second observation to hit.
    """
    windows: list[tuple[int, int]] = []
    for frame in anchor_frames:
        start = max(0, min(frame, n_frames - length))
        stop = min(n_frames, start + length)
        if sum(start <= f < stop for f in anchor_frames) < MIN_WINDOW_ANCHORS:
            continue
        if (start, stop) not in windows:
            windows.append((start, stop))
    return windows


@dataclass
class RefinementConfig:
    steps: int = REFINE_STEPS
    lr: float = REFINE_LR
    final_lr: float = REFINE_FINAL_LR
    reg_weight: float = REFINE_REG_WEIGHT
    window_frames: int = REFINE_WINDOW_FRAMES
    steps_per_window: int = REFINE_STEPS_PER_WINDOW
    train_iters: int = REFINE_TRAIN_ITERS


class _CachedEncoder:
    """Returns the frame features of the current window instead of recomputing.

    ``CoTrackerThreeOffline.forward`` runs its feature CNN over every frame on
    every call. During a fit the window is identical for several hundred calls,
    and the CNN is frozen, so its output is computed once per window and held.
    ``key = None`` disables the cache, which is the state inference runs in.
    """

    def __init__(self, inner):
        self._inner = inner
        self.key: object | None = None
        self._cached = None

    def __call__(self, x):
        import torch

        if self.key is not None and self._cached is not None and self._cached[0] == self.key:
            return self._cached[1]
        with torch.no_grad():
            out = self._inner(x)
        if self.key is not None:
            self._cached = (self.key, out)
        return out


class QueryFeatureRefinement:
    """A trainable residual on CoTracker3's per-point query features.

    Holds one ``(support, n_points, channels)`` delta per correlation-pyramid
    level. :meth:`fit` optimises it against the labelled frames; :meth:`applied`
    installs it around any call that goes through the model, inference included.
    """

    def __init__(self, predictor, n_points: int, device: str | None = None, config: RefinementConfig | None = None):
        self._predictor = predictor
        self._model = predictor.model
        self._device = device or resolve_device()
        self._n_points = int(n_points)
        self._config = config or RefinementConfig()
        self._delta = None
        self._levels: dict[tuple[int, int], int] = {}
        self.signature: str = ""

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._delta is not None

    @property
    def n_points(self) -> int:
        return self._n_points

    def matches(self, signature: str) -> bool:
        """Is the current fit still the one *signature* describes?"""
        return self.fitted and self.signature == signature

    def clear(self) -> None:
        self._delta = None
        self.signature = ""

    def save(self, path: str | Path) -> None:
        import torch

        if not self.fitted:
            raise ValueError("Nothing to save — fit the refinement first.")
        torch.save(
            {"delta": self._delta.detach().cpu(), "signature": self.signature, "n_points": self._n_points},
            str(path),
        )

    def load(self, path: str | Path, signature: str) -> bool:
        """Restore a saved fit, or return ``False`` if it no longer applies.

        A stale sidecar is a normal state (the user labelled more frames, or
        edited the schema since), so this reports rather than raises.
        """
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if state.get("signature") != signature or int(state.get("n_points", -1)) != self._n_points:
            return False
        delta = state["delta"]
        self._delta = delta.to(self._device).requires_grad_(True)
        self.signature = signature
        return True

    # ------------------------------------------------------------------
    # Applying the delta
    # ------------------------------------------------------------------

    def _level(self, fmaps) -> int | None:
        """Pyramid level of *fmaps*, identified by its spatial shape.

        ``forward`` calls ``get_track_feat`` once per level, each at half the
        previous resolution, and the model always sees the same input size — so
        first-seen order maps shapes to levels for the life of the object. Keying
        on shape rather than call order survives the chunked feature path, which
        can enter ``forward`` more than once per clip.
        """
        shape = (int(fmaps.shape[-2]), int(fmaps.shape[-1]))
        if shape not in self._levels:
            self._levels[shape] = len(self._levels)
        level = self._levels[shape]
        return level if level < self._delta.shape[0] else None

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Wrap ``get_track_feat`` so every tracked point carries its delta."""
        if not self.fitted:
            yield
            return

        model = self._model
        original = model.get_track_feat

        def patched(fmaps, queried_frames, queried_coords, support_radius=0):
            feat, support = original(fmaps, queried_frames, queried_coords, support_radius)
            level = self._level(fmaps)
            if level is None:
                return feat, support
            n = min(self._n_points, support.shape[2])
            delta = self._delta[level][:, :n].to(support.dtype)
            support = support.clone()
            # Only our own points are refined; the trailing rows are the
            # predictor's support grid, which belongs to the model.
            support[:, :, :n] = support[:, :, :n] + delta
            middle = support.shape[1] // 2
            return support[:, middle][:, None], support

        model.get_track_feat = patched
        try:
            yield
        finally:
            model.get_track_feat = original

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        anchors: dict[int, np.ndarray],
        n_frames: int,
        frames,
        signature: str = "",
        progress: Progress = _no_progress,
    ) -> bool:
        """Optimise the delta against the labelled frames.

        Returns ``False`` if the user cancelled, in which case any previous fit
        is left untouched — a half-optimised embedding is confidently wrong in
        exactly the way an unloaded checkpoint is.
        """
        import torch

        anchor_frames = sorted(f for f in anchors if 0 <= f < n_frames)
        windows = _windows(anchor_frames, n_frames, self._config.window_frames)
        if not windows:
            raise ValueError(
                "Refinement needs at least two labelled frames within "
                f"{self._config.window_frames} frames of each other."
            )

        previous, previous_signature = self._delta, self.signature
        seed, _ = SplineBackend().fill(anchors, n_frames, None, _no_progress)
        scale = float(getattr(frames, "scale", 1.0))
        clips = [self._clip(frames, start, stop, scale) for start, stop in windows]

        self._model.requires_grad_(False)
        self._delta = self._new_delta(clips[0])
        optimizer = torch.optim.Adam([self._delta], lr=self._config.lr)

        encoder = _CachedEncoder(self._model.fnet)
        self._model.fnet = encoder
        try:
            for step in range(self._config.steps):
                if not progress(step / self._config.steps):
                    self._delta, self.signature = previous, previous_signature
                    return False
                index = (step // self._config.steps_per_window) % len(windows)
                encoder.key = index
                for group in optimizer.param_groups:
                    group["lr"] = self._lr(step)
                loss = self._step(clips[index], windows[index], anchors, seed, scale)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        finally:
            self._model.fnet = encoder._inner

        self.signature = signature
        return True

    def _lr(self, step: int) -> float:
        fraction = step / max(1, self._config.steps - 1)
        return self._config.lr + (self._config.final_lr - self._config.lr) * fraction

    def _clip(self, frames, start: int, stop: int, scale: float):
        """Decode one window and resize it to the model's own resolution."""
        import torch
        import torch.nn.functional as F

        clip = np.ascontiguousarray(np.asarray(frames[start:stop]))
        video = torch.from_numpy(clip).permute(0, 3, 1, 2).float()
        self._source_size = (video.shape[-2], video.shape[-1])
        video = F.interpolate(video, tuple(self._interp_shape()), mode="bilinear", align_corners=True)
        return video[None].to(self._device)

    def _interp_shape(self) -> tuple[int, int]:
        return tuple(self._predictor.interp_shape)

    def _to_model(self, points: np.ndarray, scale: float) -> np.ndarray:
        """Source-video pixels -> the resized frames the model actually sees."""
        height, width = self._source_size
        interp_h, interp_w = self._interp_shape()
        factor = np.array([(interp_w - 1) / (width - 1), (interp_h - 1) / (height - 1)])
        return points / scale * factor

    def _new_delta(self, clip):
        """Zero delta shaped from one real ``get_track_feat`` call per level."""
        import torch

        shapes: list[tuple[int, int]] = []
        original = self._model.get_track_feat

        def probe(fmaps, queried_frames, queried_coords, support_radius=0):
            feat, support = original(fmaps, queried_frames, queried_coords, support_radius)
            shapes.append((support.shape[1], support.shape[-1]))
            return feat, support

        queries = torch.zeros((1, self._n_points, 3), device=self._device)
        self._model.get_track_feat = probe
        try:
            with torch.no_grad():
                self._model.forward(video=clip[:, :2], queries=queries, iters=1)
        finally:
            self._model.get_track_feat = original

        support, channels = shapes[0]
        delta = torch.zeros((len(shapes), support, self._n_points, channels), device=self._device)
        return delta.requires_grad_(True)

    def _targets(self, window: tuple[int, int], anchors, seed, scale):
        """Queries, ground-truth trajectories and the validity mask for a window.

        Every point gets a query row so the delta's point axis keeps its meaning;
        a point with no label inside the window is queried from the spline seed
        and masked out of the loss entirely.
        """
        import torch
        from cotracker.models.core.model_utils import get_points_on_a_grid

        start, stop = window
        length = stop - start
        n = self._n_points

        queries = np.zeros((n, 3), dtype=np.float32)
        traj = np.zeros((length, n, 2), dtype=np.float32)
        valid = np.zeros((length, n), dtype=np.float32)

        for point in range(n):
            labelled = [f for f in range(start, stop) if f in anchors and not np.isnan(anchors[f][point, 0])]
            query_frame = labelled[0] if labelled else start
            source = anchors[query_frame][point] if labelled else seed[query_frame, point]
            queries[point] = (query_frame - start, *self._to_model(np.asarray(source), scale))
            traj[:, point] = self._to_model(seed[start:stop, point], scale)
            for frame in labelled:
                traj[frame - start, point] = self._to_model(anchors[frame][point], scale)
                valid[frame - start, point] = 1.0

        grid = get_points_on_a_grid(SUPPORT_GRID_SIZE, self._interp_shape(), device=self._device)
        grid = torch.cat([torch.zeros_like(grid[:, :, :1]), grid], dim=2)
        queries_t = torch.cat([torch.from_numpy(queries)[None].to(self._device), grid], dim=1)

        # The support grid is tracked alongside our points but never supervised.
        pad = grid.shape[1]
        traj_t = torch.cat(
            [torch.from_numpy(traj)[None].to(self._device), torch.zeros((1, length, pad, 2), device=self._device)],
            dim=2,
        )
        valid_t = torch.cat(
            [torch.from_numpy(valid)[None].to(self._device), torch.zeros((1, length, pad), device=self._device)],
            dim=2,
        )
        return queries_t, traj_t, valid_t

    def _step(self, clip, window, anchors, seed, scale):
        from cotracker.models.core.cotracker.losses import sequence_loss

        queries, traj, valid = self._targets(window, anchors, seed, scale)
        with self.applied():
            *_, train_data = self._model.forward(
                video=clip,
                queries=queries,
                iters=self._config.train_iters,
                is_train=True,
            )
        coord_predictions, *_ = train_data
        track_loss = sequence_loss(coord_predictions, [traj], [valid], gamma=0.8, add_huber_loss=True).mean()
        return track_loss + self._config.reg_weight * self._delta.abs().mean()


class CoTrackerRefinedBackend(CoTrackerBackend):
    """CoTracker3 with its query features fitted to this video's labels.

    Everything the plain backend does still happens — per-gap forward/backward
    tracking, anchors copied through verbatim — the tracker simply knows what
    the user's keypoints look like *here*. The fit runs once and is reused: a
    correction should cost a forward pass, not another optimisation.
    """

    name = "CoTracker3 (refined)"
    requires_video = True

    def __init__(
        self,
        predictor,
        refinement: QueryFeatureRefinement,
        device: str | None = None,
        disagreement_px: float = DISAGREEMENT_SCALE,
    ):
        super().__init__(predictor, device=device, disagreement_px=disagreement_px)
        self.refinement = refinement
        #: Set by the GUI to name the phase in a progress dialog. Refinement is
        #: the one backend where "filling" would be a lie for most of the wait.
        self.on_stage: Callable[[str], None] | None = None
        #: Identifies the labels the fit must match; the GUI supplies it.
        self.signature: str = ""

    def fill(self, anchors, n_frames, frames=None, progress: Progress = _no_progress):
        if frames is None:
            raise ValueError(f"The {self.name} backend needs video frames.")
        if not self.refinement.matches(self.signature):
            self._stage("Learning your keypoints in this video…")
            if not self.refinement.fit(anchors, n_frames, frames, self.signature, progress):
                raise KeyboardInterrupt("Refinement cancelled.")
        self._stage("Filling frames…")
        with self.refinement.applied():
            return super().fill(anchors, n_frames, frames, progress)

    def _stage(self, text: str) -> None:
        if self.on_stage is not None:
            self.on_stage(text)
