"""Which frames to label: pick diverse or high-motion frames, not neighbours.

Labelling frames 100, 101, 102… is close to wasted effort — consecutive frames
are near-identical, so they teach a tracker almost nothing new. Both established
labelling GUIs solve this the same way, and this module follows them:

- **DeepLabCut** (``frameselectiontools.KmeansbasedFrameselection``) downsamples
  the video to ~30 px wide, treats each frame as a vector, mean-centres, runs
  MiniBatchKMeans with ``n_clusters = numframes2pick``, and takes **one frame per
  cluster** — so the chosen frames "look different, i.e. different postures".
- **SLEAP** (``FeatureSuggestionPipeline``) does image features (raw / HOG /
  BRISK bag-of-features) → PCA → k-means → sample per cluster, and separately
  offers motion-driven methods (``velocity``, ``max_point_displacement``) that
  threshold per-frame displacement to find frames worth proofreading.

Three methods are offered here:

``uniform``
    Evenly spaced. The honest baseline — for a short clip of one behaviour it is
    hard to beat, and it needs no decoding.
``diverse``
    DeepLabCut's k-means on downscaled grayscale frames: one frame per cluster,
    so distinct postures are covered rather than whatever the animal did most.
``motion``
    Frames with the largest change from the previous frame — where the action
    is. This is the same signal as EthoGraph's ``extract_video_motion`` (mean
    absolute luma difference). If the clip holds fewer *distinct* moving moments
    than you asked for, the remaining slots are filled with the next
    best-scoring frames, so the requested count is always returned; the highest-
    motion moments are simply taken first.
``uncertain``
    Frames the last fill was least sure about. Available only after a fill, and
    the method that actually matches a tracker-based workflow — see below.

Matching the method to the backend
----------------------------------
The four methods are **not interchangeable**, and the fill backend decides which
one helps:

===============  ==========================================  ====================
Fill backend     Fails when                                  Suggest with
===============  ==========================================  ====================
``spline``       the trajectory turns sharply between        ``motion`` (a proxy
                 anchors — it never looks at pixels          for curvature), or
                                                             ``uniform``
``flow``         displacement exceeds Lucas-Kanade's         ``motion``, then
                 pyramid capture range; occlusion            ``uncertain``
``cotracker``    occlusion, target leaves frame              ``uncertain``
===============  ==========================================  ====================

``diverse`` suits **none** of them, which is worth stating plainly because it is
DeepLabCut's default and the obvious thing to copy. Its premise is that labels
are *training data*, so the model needs varied appearance to generalise from.
Every fill backend here is frozen or purely geometric — none learns from your
labels — so a visually distinct frame where tracking already succeeded buys
nothing. ``diverse`` earns its place for the DeepLabCut ``CollectedData`` export
(``store_to_dlc_h5``), where the labels really are training data.

Why ``uncertain`` is the right criterion for tracker fill
---------------------------------------------------------
DeepLabCut and SLEAP select frames to *train* a pose model, so redundancy is the
enemy and visual diversity is the goal. A point tracker is not trained here at
all: CoTracker3 takes queries of ``(t, x, y)`` — **one query frame per point** —
and propagates them, so extra labelled frames exist only to reset accumulated
drift. The frames worth labelling are therefore the ones where tracking *fails*
(occlusion, motion blur, the animal leaving frame), which is not the same set as
the visually diverse ones. ``uncertain`` ranks by the fill's own confidence —
forward/backward disagreement and visibility — closing the label → fill →
correct-the-worst → fill loop. It is the analogue of SLEAP's ``prediction_score``.

One thing neither GUI enforces, and which matters for the stated goal: SLEAP's
velocity method returns *every* frame over the threshold, so a single fast bout
can supply a run of consecutive frames. Every method here passes through
:func:`enforce_min_gap`, which keeps the best-scoring frame in any neighbourhood
and drops the rest — so a burst of motion contributes one frame, not thirty.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

#: Frames decoded to score a video. Above this the candidate grid is strided —
#: clustering 2000 thumbnails is plenty to characterise any clip.
MAX_CANDIDATES = 2000

#: Longest side of the thumbnails used for scoring. DeepLabCut uses ~30 px wide;
#: a little more detail costs nothing at this frame count.
FEATURE_MAX_SIDE = 64

#: Fraction of the mean suggestion spacing enforced as a minimum gap. At 1/4,
#: asking for 20 frames over 2000 keeps suggestions at least 25 frames apart.
MIN_GAP_FRACTION = 0.25

METHODS = ("uniform", "diverse", "motion", "uncertain")


def default_min_gap(n_frames: int, count: int) -> int:
    """Minimum spacing between suggestions, derived from the request.

    Scales with the video and the number of frames asked for rather than being
    a fixed number of frames or seconds — a fixed gap would be meaningless
    across clips of different length and frame rate.
    """
    if count <= 0:
        return 1
    return max(1, int(n_frames / count * MIN_GAP_FRACTION))


def enforce_min_gap(frames: Sequence[int], min_gap: int, count: int) -> list[int]:
    """Greedily keep frames in priority order, dropping any within *min_gap*.

    *frames* must already be ordered best-first; the result is sorted by frame
    index. This is what stops one burst of motion from filling the whole budget.
    """
    kept: list[int] = []
    for frame in frames:
        if len(kept) >= count:
            break
        if all(abs(frame - k) >= min_gap for k in kept):
            kept.append(frame)
    return sorted(kept)


def _candidate_indices(n_frames: int, exclude: set[int]) -> np.ndarray:
    """Frame indices to score, strided so long videos stay tractable."""
    step = max(1, int(np.ceil(n_frames / MAX_CANDIDATES)))
    candidates = np.arange(0, n_frames, step)
    if exclude:
        candidates = np.array([c for c in candidates if int(c) not in exclude], dtype=int)
    return candidates


def _thumbnails(frames, indices: np.ndarray, progress: Callable[[float], bool] | None) -> np.ndarray:
    """``(n, d)`` mean-centred grayscale vectors, one row per candidate frame.

    Grayscale by averaging channels — DeepLabCut's choice, and colour rarely
    distinguishes postures.
    """
    rows = []
    total = max(len(indices), 1)
    for position, index in enumerate(indices):
        if progress is not None and not progress(position / total):
            break
        image = np.asarray(frames[int(index)], dtype=np.float32)
        if image.ndim == 3:
            image = image.mean(axis=2)
        rows.append(image.reshape(-1))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    data = np.asarray(rows, dtype=np.float32)
    return data - data.mean(axis=0)


def suggest_uniform(count: int, n_frames: int, exclude: set[int] | None = None) -> list[int]:
    """Evenly spaced frames — no decoding, no dependencies."""
    exclude = exclude or set()
    available = [f for f in range(n_frames) if f not in exclude]
    if not available or count <= 0:
        return []
    if count >= len(available):
        return available
    picks = np.linspace(0, len(available) - 1, count).round().astype(int)
    return sorted({available[p] for p in picks})


def _suggest_diverse(features: np.ndarray, indices: np.ndarray, count: int) -> list[int]:
    """One frame per k-means cluster (DeepLabCut's strategy).

    ``n_clusters = count`` so each cluster contributes exactly one frame; the
    frame nearest its centroid is chosen rather than a random member, which
    makes the result deterministic and picks the most representative posture.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = min(count, len(indices))
    if n_clusters < 2:
        return [int(indices[0])] if len(indices) else []

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=0)
    labels = kmeans.fit_predict(features)

    picks: list[int] = []
    for cluster in range(n_clusters):
        members = np.flatnonzero(labels == cluster)
        if not len(members):
            continue
        distances = np.linalg.norm(features[members] - kmeans.cluster_centers_[cluster], axis=1)
        picks.append(int(indices[members[int(np.argmin(distances))]]))
    return sorted(set(picks))


def _motion_scores(features: np.ndarray) -> np.ndarray:
    """Mean absolute change from the previous candidate frame."""
    if len(features) < 2:
        return np.zeros(len(features), dtype=np.float32)
    difference = np.abs(np.diff(features, axis=0)).mean(axis=1)
    # Align to the later frame of each pair: motion is attributed to where the
    # animal has moved TO, and the first candidate gets the first score.
    return np.concatenate([difference[:1], difference])


def frame_confidence(confidence: np.ndarray) -> np.ndarray:
    """Reduce a fill's confidence array to one score per frame.

    Averages over the trailing (point) axes rather than taking the minimum: a
    structurally absent point — an asymmetric schema leaves those at zero
    forever — would otherwise pin every frame to the same worst score.
    """
    array = np.asarray(confidence, dtype=np.float64)
    if array.ndim == 1:
        return array
    return np.nanmean(array.reshape(len(array), -1), axis=1)


def suggest_uncertain(
    confidence: np.ndarray,
    count: int,
    exclude: set[int] | None = None,
    min_gap: int | None = None,
) -> list[int]:
    """Frames the fill was least confident about, worst first."""
    exclude = set(exclude or ())
    scores = frame_confidence(confidence)
    n_frames = len(scores)
    if count <= 0 or not n_frames:
        return []
    order = np.argsort(scores, kind="stable")
    ranked = [int(f) for f in order if int(f) not in exclude]
    gap = default_min_gap(n_frames, count) if min_gap is None else int(min_gap)
    return enforce_min_gap(ranked, gap, count)


def suggest_frames(
    method: str,
    count: int,
    n_frames: int,
    frames=None,
    exclude: set[int] | None = None,
    min_gap: int | None = None,
    progress: Callable[[float], bool] | None = None,
    confidence: np.ndarray | None = None,
) -> list[int]:
    """Suggest *count* frames to label.

    Parameters
    ----------
    method
        One of :data:`METHODS`. ``diverse`` and ``motion`` need *frames*.
    frames
        Frame source indexable by frame index (see
        :class:`~ethograph.gui.pose_fill.VideoFrameSource`). Open it with a
        small ``max_side`` — only thumbnails are needed.
    exclude
        Frames already labelled; never suggested again (SLEAP's
        ``filter_unique_suggestions``).
    min_gap
        Minimum spacing; defaults to :func:`default_min_gap`.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown suggestion method {method!r}; expected one of {METHODS}")
    exclude = set(exclude or ())
    if count <= 0 or n_frames <= 0:
        return []
    if method == "uniform":
        return suggest_uniform(count, n_frames, exclude)
    if method == "uncertain":
        if confidence is None:
            raise ValueError("The 'uncertain' method needs a fill to have run first.")
        return suggest_uncertain(confidence, count, exclude, min_gap)
    if frames is None:
        raise ValueError(f"The {method!r} method needs video frames.")

    indices = _candidate_indices(n_frames, exclude)
    if not len(indices):
        return []
    features = _thumbnails(frames, indices, progress)
    if not len(features):
        return []
    indices = indices[: len(features)]

    gap = default_min_gap(n_frames, count) if min_gap is None else int(min_gap)
    if method == "diverse":
        # Cluster picks are already spread across posture space; the gap only
        # breaks ties between near-identical neighbours.
        return enforce_min_gap(_suggest_diverse(features, indices, count), gap, count)

    scores = _motion_scores(features)
    ranked = [int(indices[i]) for i in np.argsort(scores)[::-1]]
    return enforce_min_gap(ranked, gap, count)
