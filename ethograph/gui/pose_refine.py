"""Test-time refinement: fit CoTracker3's query features to *this* video.

Out of the box CoTracker3 tracks a point by the appearance it had on its query
frame. That is why it drifts onto the wrong leg, the wrong wing, the other
animal: nothing tells it what a *beak* looks like across the recording, only
what one patch looked like on one frame.

:class:`QueryFeatureRefinement` implements the fix from Pan et al. (2025),
"Animal Pose Labeling Using General-Purpose Point Trackers" (arXiv:2506.03868;
reference implementation https://github.com/Zhuoyang-Pan/PosePAL): freeze the
whole network and optimise **only the query-point feature embedding** against
the frames the user labelled. The paper reports 49.6 -> 67.5 delta_avg on
DAVIS-Animals from that one change, and its ablation shows tuning this embedding
beats tuning the feature extractor (54.1) or the entire network (66.3), at a
third of the cost of the latter.

``tests/_test_posepal_parity.py`` runs their ``tto`` and this module's :meth:`fit`
on one clip, one set of weights and one set of labels, and is what the claims
below are checked against rather than argued from.

Four deliberate departures from the reference implementation:

**No forked CoTracker.** PosePAL injects its features through a ``forward``
patched to accept precomputed pyramids, which upstream's ``forward`` has no
argument for. Rather than vendoring a CC-BY-NC fork, this module wraps
:meth:`get_track_feat` at runtime and **substitutes** the learned features for
what it returns. The optimised tensor is therefore the same object the paper
optimises — one absolute support-window feature per point, initialised (as in
PosePAL's ``get_kp_feats``) to the **mean over every frame the user labelled that
point on**, with the paper's L1 pull-back towards that initial value. A residual
added to whatever the current call sampled would be a different thing: its base
changes with the query frame, so the correction learned during the fit would be
applied on top of an appearance it was never fitted against.

**Windows, not whole videos.** The paper fits one 100-frame clip in a single
pass. Behaviour recordings run for minutes, so the fit iterates over short
windows spanning the labelled frames: bounded memory, and every labelled frame
is used however long the video is.

**Sparse supervision.** Their benchmark videos are labelled on every frame; here
only a handful are, so ``valids`` masks the loss down to the (frame, point)
pairs a human actually placed — which is what ``reduce_masked_mean`` inside
``sequence_loss`` is there for.

**Pixels the network was trained on.** PosePAL's ``extract_features`` runs the
feature CNN on ``process_video``'s raw 0..255 tensor, while ``forward`` — which
this module goes through, as does every other caller — maps the video to
``[-1, 1]`` first. Building the pyramid inside ``forward`` is therefore not only
less code than copying their 40-line pyramid builder, it is the corrected input
range: the parity harness matches their initial features to ~1.5e-5 against a
normalised rebuild, and to only ~5e-3 (a tenth of the features' own magnitude)
against their code as written.

Everything torch- and cotracker-specific lives in this module, which is imported
lazily by :func:`ethograph.gui.pose_fill.build_backend` and never at GUI start.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from cotracker.models.core.cotracker.losses import sequence_loss
from cotracker.models.core.model_utils import get_points_on_a_grid

from ethograph.gui.pose_fill import (
    DISAGREEMENT_SCALE,
    POSEPAL_LABEL,
    Progress,
    SplineBackend,
    _CoTrackerTracking,
    no_progress,
    resolve_device,
)

#: Optimisation steps. PosePAL's own default is 500; the paper's evaluation uses
#: 1000. 500 converges well inside the paper's ~3 minutes on a mid-range GPU and
#: keeps the fit to a tolerable wait — this is a labelling loop, not a benchmark.
REFINE_STEPS = 500

#: Adam, decayed linearly across the fit. PosePAL's default is a constant 1e-4
#: (its scheduler is commented out), which is what the start is matched to: the
#: features being optimised come off an L2-normalised feature map, so a step size
#: an order of magnitude larger walks them off the unit sphere they live on.
REFINE_LR = 1e-4
REFINE_FINAL_LR = 1e-5

#: Weight of the L1 pull-back towards the unrefined features (paper: 0.01).
REFINE_REG_WEIGHT = 0.01

#: Frames per training window. Stays under the model's own 60-frame attention
#: window, and is the main lever on how much memory the fit needs.
REFINE_WINDOW_FRAMES = 48

#: Consecutive steps spent on one window before moving to the next. Frame
#: features are cached per window, so a switch costs a CNN pass over the whole
#: window; a short run on each visit amortises that away.
REFINE_STEPS_PER_WINDOW = 5

#: Tracker update iterations during the fit. Inference uses 6; the reference
#: implementation trains with 4 — a third less backprop for a loss that already
#: weights the last iterations most (gamma = 0.8).
REFINE_TRAIN_ITERS = 4

#: A window must hold at least this many labelled frames to teach anything: with
#: one, the only supervised frame is the query frame itself.
MIN_WINDOW_ANCHORS = 2

#: Most windows any one fit trains on, spread evenly over those available. One
#: window per labelled frame would decode (and hold) unboundedly much for a
#: densely labelled recording, and the paper fits a single clip — a dozen spans
#: sampled across the video is already far more coverage than that.
MAX_TRAINING_WINDOWS = 12

#: Context points CoTracker's predictor appends to every query set. They are the
#: model's own scaffolding, not the user's keypoints, so they never get a feature.
SUPPORT_GRID_SIZE = 6


@contextmanager
def _patched_track_feat(model, replacement) -> Iterator[None]:
    """Shadow ``model.get_track_feat`` for the duration of a block.

    Restoring by deleting the instance attribute rather than reassigning the
    bound method leaves the model exactly as it was found — no shadowing
    attribute, and no reference from the model back to this object.
    """
    model.get_track_feat = replacement
    try:
        yield
    finally:
        del model.get_track_feat


def training_windows(
    anchor_frames: list[int],
    n_frames: int,
    length: int,
    limit: int = MAX_TRAINING_WINDOWS,
) -> list[tuple[int, int]]:
    """Contiguous ``[start, stop)`` windows, each holding >= 2 labelled frames.

    One window is anchored at each labelled frame, so every labelled frame is
    covered whenever a neighbour is within reach. Anchors more than *length*
    apart contribute no window between them — a query with no second observation
    to hit teaches nothing.

    More than *limit* windows are thinned evenly rather than truncated, so a
    capped fit still spans the whole video instead of its opening minutes.
    """
    windows: list[tuple[int, int]] = []
    for frame in anchor_frames:
        start = max(0, min(frame, n_frames - length))
        stop = min(n_frames, start + length)
        if sum(start <= f < stop for f in anchor_frames) < MIN_WINDOW_ANCHORS:
            continue
        if (start, stop) not in windows:
            windows.append((start, stop))
    if limit and len(windows) > limit:
        keep = np.linspace(0, len(windows) - 1, limit).round().astype(int)
        windows = [windows[i] for i in dict.fromkeys(keep.tolist())]
    return windows


@dataclass
class RefinementConfig:
    """Optimiser settings. Deliberately not exposed in the GUI — unlike the
    disagreement tolerance, none of these depend on the footage."""

    steps: int = REFINE_STEPS
    lr: float = REFINE_LR
    final_lr: float = REFINE_FINAL_LR
    reg_weight: float = REFINE_REG_WEIGHT
    window_frames: int = REFINE_WINDOW_FRAMES
    steps_per_window: int = REFINE_STEPS_PER_WINDOW
    train_iters: int = REFINE_TRAIN_ITERS


class _CachedEncoder(torch.nn.Module):
    """Returns the current window's frame features instead of recomputing them.

    ``CoTrackerThreeOffline.forward`` runs its feature CNN over every frame on
    every call. During a fit the window is unchanged for several hundred calls
    and the CNN is frozen, so its output is computed once per window and held.
    ``key = None`` disables caching, which is the state inference runs in.
    """

    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.key: object | None = None
        self._cached: tuple[object, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.key is not None and self._cached is not None and self._cached[0] == self.key:
            return self._cached[1]
        with torch.no_grad():
            out = self.inner(x)
        if self.key is not None:
            self._cached = (self.key, out)
        return out


class QueryFeatureRefinement:
    """CoTracker3's per-point query features, learned for one video.

    Holds one ``(support, n_points, channels)`` feature tensor per
    correlation-pyramid level, plus the mask of which point rows it actually
    learned anything for. :meth:`fit` optimises it against the labelled frames;
    :meth:`applied` installs it around anything that goes through the model,
    inference included.

    The fit is specific to one video *and* one set of point rows, so
    :attr:`signature` — supplied by the caller — is what says whether it still
    applies; a schema edit or a new label changes it.
    """

    def __init__(
        self,
        predictor,
        n_points: int,
        device: str | None = None,
        config: RefinementConfig | None = None,
    ):
        self._predictor = predictor
        self._model = predictor.model
        self._device = device or resolve_device()
        self._n_points = int(n_points)
        self._config = config or RefinementConfig()
        self._features: torch.Tensor | None = None
        #: The initial (unoptimised) features, target of the L1 pull-back.
        self._initial: torch.Tensor | None = None
        #: Point rows with at least one label, hence a feature worth using.
        self._learned: torch.Tensor | None = None
        #: Flat point rows the *current* caller's query list stands for, set by
        #: :meth:`set_rows` — ``None`` means "all of them, in order".
        self._rows: torch.Tensor | None = None
        self._levels: dict[tuple[int, int], int] = {}
        self._source_size: tuple[int, int] | None = None
        self.signature: str = ""
        #: Labelled frames the current fit was made from — what the GUI reports.
        self.n_anchor_frames: int = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self._features is not None

    @property
    def n_points(self) -> int:
        return self._n_points

    def matches(self, signature: str) -> bool:
        """Is the current fit the one *signature* describes?"""
        return self.fitted and self.signature == signature

    def clear(self) -> None:
        self._features = self._initial = self._learned = None
        self.signature = ""
        self.n_anchor_frames = 0

    def save(self, path: str | Path) -> None:
        if self._features is None or self._learned is None:
            raise ValueError("Nothing to save — fit the refinement first.")
        torch.save(
            {
                "features": self._features.detach().cpu(),
                "learned": self._learned.cpu(),
                "signature": self.signature,
                "n_points": self._n_points,
                "n_anchor_frames": self.n_anchor_frames,
            },
            str(path),
        )

    def load(self, path: str | Path, signature: str) -> bool:
        """Restore a saved fit, reporting whether it still applies.

        A stale sidecar is an ordinary state — the user labelled more frames or
        edited the schema since — so this returns ``False`` rather than raising.
        A sidecar written before the features were learned absolutely holds a
        residual instead, which means nothing here: it has no ``features`` key
        and so reads as a miss.
        """
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if state.get("signature") != signature or int(state.get("n_points", -1)) != self._n_points:
            return False
        if "features" not in state or "learned" not in state:
            return False
        self._features = state["features"].to(self._device).requires_grad_(True)
        self._initial = self._features.detach().clone()
        self._learned = state["learned"].to(self._device)
        self.signature = signature
        self.n_anchor_frames = int(state.get("n_anchor_frames", 0))
        return True

    # ------------------------------------------------------------------
    # Applying the learned features
    # ------------------------------------------------------------------

    def set_rows(self, rows: np.ndarray | None) -> None:
        """Say which flat point rows the next query list stands for.

        The gap backends drop point rows they have no seed for, so the query list
        the tracker sees is compressed and query ``i`` is point ``rows[i]``.
        Without this the wrong keypoint's feature would be handed to every point
        after the first missing one — silently, and only for the schemas where
        some point is never labelled.
        """
        self._rows = None if rows is None else torch.as_tensor(np.asarray(rows), dtype=torch.long, device=self._device)

    def _level(self, fmaps: torch.Tensor) -> int:
        """Pyramid level of *fmaps*, identified by its spatial shape.

        ``forward`` calls ``get_track_feat`` once per level, each at half the
        previous resolution, and the model always sees one input size (the
        predictor resizes for it) — so first-seen order maps shapes to levels for
        the life of this object. Keying on shape rather than call order survives
        the chunked feature path, which can re-enter ``forward`` per clip.
        """
        shape = (int(fmaps.shape[-2]), int(fmaps.shape[-1]))
        if shape not in self._levels:
            self._levels[shape] = len(self._levels)
        return self._levels[shape]

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Wrap ``get_track_feat`` so every tracked point gets its own feature."""
        if self._features is None:
            yield
            return

        original = self._model.get_track_feat

        def patched(fmaps, queried_frames, queried_coords, support_radius=0):
            feat, support = original(fmaps, queried_frames, queried_coords, support_radius)
            level = self._level(fmaps)
            if level >= self._features.shape[0]:
                return feat, support
            learned, mask = self._for_query(level)
            n = min(learned.shape[1], support.shape[2])
            support = support.clone()
            # Substituted, not added: the learned feature *is* the point's
            # appearance template, independent of the frame this call sampled.
            # Rows the fit never saw a label for keep the model's own feature.
            support[:, :, :n] = torch.where(
                mask[:n].view(1, 1, n, 1),
                learned[:, :n].unsqueeze(0).to(support.dtype),
                support[:, :, :n],
            )
            # The track feature is the centre of the support window, so it has to
            # be re-read from the refined tensor rather than passed through.
            middle = support.shape[1] // 2
            return support[:, middle][:, None], support

        try:
            with _patched_track_feat(self._model, patched):
                yield
        finally:
            self._rows = None

    def _for_query(self, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This level's features and learned-mask, in query-list order."""
        assert self._features is not None and self._learned is not None
        learned, mask = self._features[level], self._learned
        if self._rows is None:
            return learned, mask
        return learned[:, self._rows], mask[self._rows]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        anchors: dict[int, np.ndarray],
        n_frames: int,
        frames,
        signature: str = "",
        progress: Progress = no_progress,
    ) -> bool:
        """Optimise the query features against the labelled frames.

        Returns ``False`` if the user cancelled, leaving any previous fit
        untouched — a half-optimised embedding is confidently wrong in exactly
        the way an unloaded checkpoint is.
        """
        anchor_frames = sorted(f for f in anchors if 0 <= f < n_frames)
        windows = training_windows(anchor_frames, n_frames, self._config.window_frames)
        if not windows:
            raise ValueError(
                "Refinement needs at least two labelled frames within "
                f"{self._config.window_frames} frames of each other."
            )

        previous = (self._features, self._initial, self._learned, self.signature, self.n_anchor_frames)
        seed, _ = SplineBackend().fill(anchors, n_frames, None, no_progress)
        scale = float(getattr(frames, "scale", 1.0))
        # Decoded once, held as uint8 on the CPU: a dozen windows of float frames
        # at model resolution would be gigabytes, on the device, for nothing.
        clips = [np.ascontiguousarray(np.asarray(frames[start:stop])) for start, stop in windows]
        self._source_size = (int(clips[0].shape[1]), int(clips[0].shape[2]))

        self._model.requires_grad_(False)
        self._rows = None

        encoder = _CachedEncoder(self._model.fnet)
        self._model.fnet = encoder
        try:
            if not self._initialise(clips, windows, anchors, scale, progress):
                self._restore(previous)
                return False
            assert self._features is not None
            optimizer = torch.optim.Adam([self._features], lr=self._config.lr)
            current, clip = 0, self._to_device(clips[0])
            encoder.key = 0
            for step in range(self._config.steps):
                if not progress(step / self._config.steps):
                    self._restore(previous)
                    return False
                index = (step // self._config.steps_per_window) % len(windows)
                if index != current:
                    # One window on the device at a time; the encoder's cached
                    # features are keyed to it and recomputed on the switch.
                    current, clip = index, self._to_device(clips[index])
                    encoder.key = index
                for group in optimizer.param_groups:
                    group["lr"] = self._lr(step)
                loss = self._step(clip, windows[index], anchors, seed, scale)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        finally:
            self._model.fnet = encoder.inner

        self.signature = signature
        self.n_anchor_frames = len(anchor_frames)
        return True

    def _restore(self, previous: tuple) -> None:
        """Put back the fit a cancelled one replaced, whole."""
        self._features, self._initial, self._learned, self.signature, self.n_anchor_frames = previous

    def _initialise(
        self,
        clips: list[np.ndarray],
        windows: list[tuple[int, int]],
        anchors: dict[int, np.ndarray],
        scale: float,
        progress: Progress,
    ) -> bool:
        """Seed the features with the mean appearance over the user's labels.

        PosePAL's ``get_kp_feats``: a point's template is the average of the
        support windows sampled at **every frame the user labelled it on**, not
        the patch from one query frame. That average is where the optimisation
        starts and what its L1 term pulls back towards, so the fit begins from
        something already better than stock CoTracker rather than having to
        rediscover it.

        The average is taken per pyramid level, each read from that level's own
        feature maps — PosePAL computes level 0 only and reuses it for all four,
        which upstream's ``forward`` never does. Each ``(point, frame)`` pair
        counts once, from the first window that contains it, so a frame shared by
        two windows does not weigh double.
        """
        totals: dict[int, torch.Tensor] = {}
        counts = torch.zeros(self._n_points, device=self._device)
        seen: set[tuple[int, int]] = set()
        original = self._model.get_track_feat

        def collector(row_index: torch.Tensor):
            def collect(fmaps, queried_frames, queried_coords, support_radius=0):
                feat, support = original(fmaps, queried_frames, queried_coords, support_radius)
                level = self._level(fmaps)
                if level not in totals:
                    totals[level] = torch.zeros(
                        (support.shape[1], self._n_points, support.shape[-1]),
                        device=self._device,
                        dtype=torch.float32,
                    )
                totals[level].index_add_(1, row_index, support[0].float())
                return feat, support

            return collect

        for window, clip in zip(windows, clips):
            # One forward per window, so cancelling stays responsive here too.
            if not progress(0.0):
                return False
            rows, queries = self._label_queries(window, anchors, scale, seen)
            if not len(rows):
                continue
            row_index = torch.as_tensor(rows, dtype=torch.long, device=self._device)
            with _patched_track_feat(self._model, collector(row_index)), torch.no_grad():
                # The queries *are* the labelled (frame, point) pairs, so what
                # this samples is exactly the set of features to average.
                self._model.forward(video=self._to_device(clip), queries=queries, iters=1)
            counts.index_add_(0, row_index, torch.ones_like(row_index, dtype=torch.float32))

        if not totals:
            raise ValueError("No labelled points fell inside a training window.")

        learned = counts > 0
        divisor = counts.clamp(min=1.0).view(1, -1, 1)
        features = torch.stack([totals[level] / divisor for level in sorted(totals)])
        self._features = features.requires_grad_(True)
        self._initial = features.detach().clone()
        self._learned = learned
        return True

    def _label_queries(
        self,
        window: tuple[int, int],
        anchors: dict[int, np.ndarray],
        scale: float,
        seen: set[tuple[int, int]],
    ) -> tuple[np.ndarray, torch.Tensor]:
        """One query row per labelled ``(point, frame)`` pair in *window*."""
        start, stop = window
        rows: list[int] = []
        queries: list[tuple[float, float, float]] = []
        for frame in range(start, stop):
            if frame not in anchors:
                continue
            for point in range(self._n_points):
                if np.isnan(anchors[frame][point, 0]) or (point, frame) in seen:
                    continue
                seen.add((point, frame))
                rows.append(point)
                queries.append((float(frame - start), *self._to_model(anchors[frame][point], scale)))
        return (
            np.asarray(rows, dtype=np.int64),
            torch.tensor(queries, dtype=torch.float32, device=self._device)[None],
        )

    def _lr(self, step: int) -> float:
        fraction = step / max(1, self._config.steps - 1)
        return self._config.lr + (self._config.final_lr - self._config.lr) * fraction

    def _to_device(self, clip: np.ndarray) -> torch.Tensor:
        """One decoded window as the model wants it: on the device, resized."""
        video = torch.from_numpy(clip).permute(0, 3, 1, 2).float().to(self._device)
        video = F.interpolate(video, self._interp_shape(), mode="bilinear", align_corners=True)
        return video[None]

    def _interp_shape(self) -> tuple[int, int]:
        height, width = self._predictor.interp_shape
        return int(height), int(width)

    def _to_model(self, points: np.ndarray, scale: float) -> np.ndarray:
        """Source-video pixels -> the resized frames the model actually sees."""
        assert self._source_size is not None
        height, width = self._source_size
        interp_h, interp_w = self._interp_shape()
        factor = np.array([(interp_w - 1) / (width - 1), (interp_h - 1) / (height - 1)])
        return np.asarray(points, dtype=np.float64) / scale * factor

    def _targets(
        self,
        window: tuple[int, int],
        anchors: dict[int, np.ndarray],
        seed: np.ndarray,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Queries, target trajectories and the validity mask for one window.

        Every point gets a query row so the features' point axis keeps its
        meaning across windows; a point with no label inside the window is queried
        from the spline seed and masked out of the loss entirely.
        """
        start, stop = window
        length = stop - start
        n = self._n_points

        queries = np.zeros((n, 3), dtype=np.float32)
        trajectory = np.zeros((length, n, 2), dtype=np.float32)
        valid = np.zeros((length, n), dtype=np.float32)

        for point in range(n):
            labelled = [f for f in range(start, stop) if f in anchors and not np.isnan(anchors[f][point, 0])]
            query_frame = labelled[0] if labelled else start
            source = anchors[query_frame][point] if labelled else seed[query_frame, point]
            queries[point] = (query_frame - start, *self._to_model(source, scale))
            trajectory[:, point] = self._to_model(seed[start:stop, point], scale)
            for frame in labelled:
                trajectory[frame - start, point] = self._to_model(anchors[frame][point], scale)
                valid[frame - start, point] = 1.0

        # A point labelled nowhere in this video has no seed to be queried from,
        # and a NaN query poisons the whole forward pass — every point, hence the
        # loss, hence the features. `valid` already excludes these points from the
        # loss, so any finite placeholder does; they simply train on nothing (and
        # `_learned` keeps their untrained feature out of inference entirely).
        np.nan_to_num(queries, copy=False)
        np.nan_to_num(trajectory, copy=False)

        grid = get_points_on_a_grid(SUPPORT_GRID_SIZE, self._interp_shape(), device=self._device)
        grid = torch.cat([torch.zeros_like(grid[:, :, :1]), grid], dim=2)
        pad = grid.shape[1]

        return (
            torch.cat([torch.from_numpy(queries)[None].to(self._device), grid], dim=1),
            torch.cat(
                [
                    torch.from_numpy(trajectory)[None].to(self._device),
                    torch.zeros((1, length, pad, 2), device=self._device),
                ],
                dim=2,
            ),
            torch.cat(
                [
                    torch.from_numpy(valid)[None].to(self._device),
                    torch.zeros((1, length, pad), device=self._device),
                ],
                dim=2,
            ),
        )

    def _step(
        self,
        clip: torch.Tensor,
        window: tuple[int, int],
        anchors: dict[int, np.ndarray],
        seed: np.ndarray,
        scale: float,
    ) -> torch.Tensor:
        queries, trajectory, valid = self._targets(window, anchors, seed, scale)
        with self.applied():
            *_, train_data = self._model.forward(
                video=clip,
                queries=queries,
                iters=self._config.train_iters,
                is_train=True,
            )
        coord_predictions, *_ = train_data
        track_loss = sequence_loss(
            coord_predictions,
            [trajectory],
            [valid],
            gamma=0.8,
            add_huber_loss=True,
        ).mean()
        assert self._features is not None and self._initial is not None
        # The paper's L1 pull-back: stay near the appearance the labels averaged
        # to unless the tracking loss really pays for moving away from it.
        pull_back = (self._features - self._initial).abs().mean()
        return track_loss + self._config.reg_weight * pull_back


class PosePALBackend(_CoTrackerTracking):
    """CoTracker3 with its query features fitted to this video's labels.

    The method of Pan et al. 2025, and the only form of CoTracker3 the GUI
    offers: plain tracking is what happens on the way, not an alternative to
    choose. Per-gap forward/backward tracking and verbatim anchors are inherited
    from :class:`~ethograph.gui.pose_fill._CoTrackerTracking` — the tracker
    simply knows what the user's keypoints look like *here*. The fit runs once
    per label set and is then reused, so correcting a point costs a forward pass
    and not another optimisation; :attr:`signature` is what decides that the fit
    has gone stale.
    """

    name = POSEPAL_LABEL
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
        #: Set by the GUI to name the phase in a progress dialog. This is the one
        #: backend where "Filling frames" would be a lie for most of the wait.
        self.on_stage: Callable[[str], None] | None = None
        #: Identifies the labels the fit has to match; the GUI supplies it.
        self.signature: str = ""

    def fill(self, anchors, n_frames, frames=None, progress: Progress = no_progress):
        if frames is None:
            raise ValueError(f"The {self.name} backend needs video frames.")
        if not self.refinement.matches(self.signature):
            self._stage("Learning your keypoints in this video…")
            if not self.refinement.fit(anchors, n_frames, frames, self.signature, progress):
                # Same contract as cancelling any other fill: anchors survive and
                # what is returned is the plain interpolation, never a partial fit.
                return SplineBackend().fill(anchors, n_frames, None, no_progress)
        self._stage("Filling frames…")
        with self.refinement.applied():
            return super().fill(anchors, n_frames, frames, progress)

    def _on_rows(self, rows: np.ndarray) -> None:
        """The features are per point row, and a gap tracks only the seeded ones."""
        self.refinement.set_rows(rows)

    def _stage(self, text: str) -> None:
        if self.on_stage is not None:
            self.on_stage(text)
