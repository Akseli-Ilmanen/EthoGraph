"""Keypoint annotation state + export.

:class:`KeypointStore` owns every coordinate the user labels and everything a
fill backend produces. The GUI never mutates the arrays directly — it calls
``set_point`` / ``clear_point`` / ``undo`` and reads back through ``positions``.

Hierarchy
---------
The store is two-level, like SLEAP's skeleton/instance split: one keypoint
schema (the skeleton) is shared by every individual, and each individual is an
instance of that schema on a given frame. Anchors are therefore
``(n_individuals, n_keypoints, 2)`` per frame, and the active target the canvas
writes to is an ``(individual, keypoint)`` pair. Single-individual labelling is
just the ``n_individuals == 1`` case — every method takes ``individual=None``
to mean "the first (usually only) one". ``n_individuals == 0`` is legal too: it
is the state right after deleting the last individual, and nothing can be
labelled until one is added back.

Asymmetric schemas
------------------
With ``shared_keypoints = False`` each individual carries its own subset of the
keypoint schema (label the wings of the flying bird only, say). ``keypoint_names``
stays the union across individuals — the arrays remain rectangular over it, so
backends, exports and the overlay are unchanged — and :attr:`keypoint_sets` says
which pairs actually exist. Points outside an individual's set are permanently
``NaN`` and :meth:`set_point` refuses them.

Coordinate convention
---------------------
Everything here is ``(x, y)`` in **pixel coordinates of the source video**,
``NaN`` where unlabelled. Note that :func:`~ethograph.gui.pose_convert.poses_ds_to_points`
emits ``(track_id, frame, y, x)`` — the axis swap lives in this module (in
:func:`store_to_movement_ds`) and nowhere else.

Naming follows the rest of the codebase (and movement ≥0.17): the *dimension*
of a poses dataset is singular — ``keypoint``, ``individual`` — even though it
holds many of them. Plural names here are Python containers and counts only.

A frame counts as an anchor if *any* keypoint of *any* individual is labelled.
Partially labelled anchors are normal: the user labels the beak on some frames
and the tail on others, so backends must handle per-point anchor sets rather
than a single shared frame list (see :meth:`KeypointStore.anchor_frames_for`).

Provenance: human vs fill
-------------------------
The split between what a person placed and what a backend produced is the
:attr:`~KeypointStore.anchors` / :attr:`~KeypointStore.filled` split, and a fill
never feeds the next fill: :meth:`~KeypointStore.flat_anchors` reads the anchors
alone, so re-filling is a pure function of the labels. :meth:`~KeypointStore.human_mask`
reports that provenance per point and :meth:`~KeypointStore.is_human` per
``(frame, individual)`` row — one human point anywhere in the row makes the row
human, which is the rule the dialog's ``Source`` column shows.

A filled point becomes human by being touched: clicking one on the canvas pins
it where the backend put it (see :meth:`~KeypointStore.promote_fill` for the
same thing over a whole frame). That is how a fill is "accepted" — there is no
third state between labelled and predicted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

#: Sidecar suffix appended to the video path to persist anchors.
SIDECAR_SUFFIX = ".keypoints.json"

#: Suggested number of labelled frames shown in the dialog's counter.
RECOMMENDED_ANCHORS = 20

#: Name given to the first individual when the user has not named any.
DEFAULT_INDIVIDUAL = "individual_0"


class KeypointStoreError(Exception):
    """Base for keypoint store failures."""


class UnknownKeypointError(KeypointStoreError):
    """Raised when a keypoint name is not in the store's schema."""


class UnknownIndividualError(KeypointStoreError):
    """Raised when an individual name is not in the store's schema."""


@dataclass
class KeypointStore:
    """Per-frame keypoint coordinates: user anchors + backend fill.

    Attributes
    ----------
    keypoint_names
        Keypoint schema, in display order — the union across individuals.
    n_frames
        Number of frames in the video being labelled.
    individual_names
        Individuals being labelled, in display order. May be empty.
    shared_keypoints
        ``True`` (the default) when every individual is an instance of the whole
        schema; ``False`` when each carries its own subset.
    keypoint_sets
        ``individual -> keypoints``, only meaningful (and only populated) when
        ``shared_keypoints`` is ``False``.
    anchors
        ``frame -> (n_individuals, n_keypoints, 2)`` array of user-placed
        ``(x, y)``, ``NaN`` where that point is unlabelled on that frame.
    filled
        ``(n_frames, n_individuals, n_keypoints, 2)`` backend output, or
        ``None`` before the first fill. Anchor frames are copied through
        verbatim.
    confidence
        ``(n_frames, n_individuals, n_keypoints)`` in ``[0, 1]``; anchors
        are ``1.0``.
    """

    keypoint_names: list[str]
    n_frames: int
    individual_names: list[str] = field(default_factory=lambda: [DEFAULT_INDIVIDUAL])
    shared_keypoints: bool = True
    keypoint_sets: dict[str, list[str]] = field(default_factory=dict)
    anchors: dict[int, np.ndarray] = field(default_factory=dict)
    filled: np.ndarray | None = None
    confidence: np.ndarray | None = None
    _history: list[tuple[int, int, int, np.ndarray | None]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.keypoint_names = [str(n) for n in self.keypoint_names]
        self.individual_names = [str(n) for n in self.individual_names]
        self.n_frames = int(self.n_frames)
        self._normalise_keypoint_sets()
        self._prune_unowned()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoint_names)

    @property
    def n_individuals(self) -> int:
        return len(self.individual_names)

    @property
    def n_points(self) -> int:
        """Rows in the flat point grid — one per ``(individual, keypoint)`` pair.

        This is the array size the fill backends see, so it counts pairs that
        an asymmetric schema leaves permanently unlabelled too; for the number
        of points the user can actually place, see :attr:`n_schema_points`.
        """
        return self.n_individuals * self.n_keypoints

    @property
    def n_schema_points(self) -> int:
        """Points the schema actually contains; ``== n_points`` when shared."""
        if self.shared_keypoints:
            return self.n_points
        return sum(len(self.keypoint_sets.get(name, ())) for name in self.individual_names)

    def keypoint_index(self, name: str) -> int:
        try:
            return self.keypoint_names.index(name)
        except ValueError:
            raise UnknownKeypointError(f"{name!r} is not in {self.keypoint_names}") from None

    def individual_index(self, name: str | None) -> int:
        """Index of *name*; ``None`` means the first (usually only) individual."""
        if not self.individual_names:
            raise UnknownIndividualError("There are no individuals — add one before labelling.")
        if name is None:
            return 0
        try:
            return self.individual_names.index(name)
        except ValueError:
            raise UnknownIndividualError(f"{name!r} is not in {self.individual_names}") from None

    def keypoints_for(self, individual: str | None = None) -> list[str]:
        """Keypoints belonging to *individual*, in schema order."""
        if self.shared_keypoints:
            return list(self.keypoint_names)
        return list(self.keypoint_sets.get(self.individual_names[self.individual_index(individual)], []))

    def has_keypoint(self, keypoint: str, individual: str | None = None) -> bool:
        """Whether *individual* carries *keypoint* under the current schema."""
        return keypoint in self.keypoints_for(individual)

    def keypoint_mask(self) -> np.ndarray:
        """``(n_individuals, n_keypoints)`` bool: which pairs exist."""
        mask = np.ones((self.n_individuals, self.n_keypoints), dtype=bool)
        if self.shared_keypoints:
            return mask
        for i, individual in enumerate(self.individual_names):
            owned = set(self.keypoint_sets.get(individual, ()))
            mask[i] = [name in owned for name in self.keypoint_names]
        return mask

    def set_keypoint_names(self, names: list[str]) -> None:
        """Replace the keypoint schema, carrying labelled points over by name."""
        self._reschema([str(n) for n in names], self.individual_names)

    def set_individual_names(self, names: list[str]) -> None:
        """Replace the individuals, carrying labelled points over by name."""
        self._reschema(self.keypoint_names, [str(n) for n in names])

    def set_shared_keypoints(self, shared: bool) -> None:
        """Switch between one shared schema and per-individual subsets.

        Neither direction touches the anchors: the arrays already span the union
        of keypoints, so turning sharing off simply gives every individual the
        whole schema to start from, and turning it back on re-admits the pairs
        that were excluded (they are ``NaN``, having never been labelled).
        """
        shared = bool(shared)
        if shared == self.shared_keypoints:
            return
        self.shared_keypoints = shared
        self.keypoint_sets = {} if shared else {name: list(self.keypoint_names) for name in self.individual_names}

    def set_keypoints_for(self, individual: str | None, names: list[str]) -> None:
        """Replace one individual's keypoints; names new to the union are added."""
        if self.shared_keypoints:
            raise KeypointStoreError(
                "Keypoints are shared by every individual — turn sharing off before editing one individual's set."
            )
        individual = self.individual_names[self.individual_index(individual)]
        names = list(dict.fromkeys(str(n) for n in names))
        union = self.keypoint_names + [name for name in names if name not in self.keypoint_names]
        if union != self.keypoint_names:
            self._reschema(union, self.individual_names)
        self.keypoint_sets[individual] = [name for name in self.keypoint_names if name in set(names)]
        self._prune_unowned()

    def _normalise_keypoint_sets(self) -> None:
        """Keep :attr:`keypoint_sets` in step with the individuals and the union."""
        if self.shared_keypoints:
            self.keypoint_sets = {}
            return
        normalised: dict[str, list[str]] = {}
        for individual in self.individual_names:
            owned = set(self.keypoint_sets.get(individual, self.keypoint_names))
            normalised[individual] = [name for name in self.keypoint_names if name in owned]
        self.keypoint_sets = normalised

    def _prune_unowned(self) -> None:
        """Drop anchors for pairs the schema no longer contains."""
        unowned = ~self.keypoint_mask()
        if not unowned.any():
            return
        pruned = False
        for frame in list(self.anchors):
            points = self.anchors[frame]
            if np.any(~np.isnan(points[unowned])):
                points[unowned] = np.nan
                pruned = True
                self._drop_if_empty(frame)
        if pruned:
            self._history.clear()
            self.clear_fill()

    def _reschema(self, keypoints: list[str], individuals: list[str]) -> None:
        """Re-index anchors onto a new schema; dropped names lose their points.

        Any existing fill is invalidated — its axes no longer match.
        """
        if keypoints == self.keypoint_names and individuals == self.individual_names:
            return
        old_kp = {n: i for i, n in enumerate(self.keypoint_names)}
        old_ind = {n: i for i, n in enumerate(self.individual_names)}
        remapped: dict[int, np.ndarray] = {}
        for frame, points in self.anchors.items():
            new_points = np.full((len(individuals), len(keypoints), 2), np.nan, dtype=np.float64)
            for i, individual in enumerate(individuals):
                if individual not in old_ind:
                    continue
                for k, keypoint in enumerate(keypoints):
                    if keypoint in old_kp:
                        new_points[i, k] = points[old_ind[individual], old_kp[keypoint]]
            if np.any(~np.isnan(new_points)):
                remapped[frame] = new_points
        self.keypoint_names = keypoints
        self.individual_names = individuals
        self.anchors = remapped
        self._history.clear()
        self.clear_fill()
        self._normalise_keypoint_sets()
        self._prune_unowned()

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def set_point(self, frame: int, keypoint: str, xy: tuple[float, float], individual: str | None = None) -> None:
        """Place *keypoint* of *individual* at ``xy`` on *frame*."""
        i, k = self.individual_index(individual), self.keypoint_index(keypoint)
        if not self.has_keypoint(keypoint, individual):
            raise UnknownKeypointError(f"{keypoint!r} is not a keypoint of {self.individual_names[i]!r}")
        frame = int(frame)
        points = self.anchors.get(frame)
        if points is None:
            points = np.full((self.n_individuals, self.n_keypoints, 2), np.nan, dtype=np.float64)
            self.anchors[frame] = points
        self._record(frame, i, k, points[i, k])
        points[i, k] = (float(xy[0]), float(xy[1]))

    def clear_point(self, frame: int, keypoint: str, individual: str | None = None) -> None:
        """Remove *keypoint* of *individual* from *frame*; drops empty anchors."""
        i, k = self.individual_index(individual), self.keypoint_index(keypoint)
        frame = int(frame)
        points = self.anchors.get(frame)
        if points is None:
            return
        self._record(frame, i, k, points[i, k])
        points[i, k] = np.nan
        self._drop_if_empty(frame)

    def clear_individual(self, frame: int, individual: str | None = None) -> None:
        """Remove every keypoint of *individual* on *frame* (one undo step each)."""
        i = self.individual_index(individual)
        points = self.anchors.get(int(frame))
        if points is None:
            return
        for k, keypoint in enumerate(self.keypoint_names):
            if not np.isnan(points[i, k, 0]):
                self.clear_point(frame, keypoint, self.individual_names[i])

    def predicted_mask(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints)`` bool: points the *backend* placed.

        Anchors are copied into :attr:`filled` verbatim, so "has a filled
        position" is not the same question as "was predicted"; this excludes
        them, and is therefore the complement of :meth:`human_mask` over the
        points that exist at all.
        """
        if self.filled is None or not 0 <= int(frame) < self.n_frames:
            return np.zeros((self.n_individuals, self.n_keypoints), dtype=bool)
        return ~np.isnan(self.filled[int(frame), :, :, 0]) & ~self.human_mask(frame)

    def clear_fill_for(self, frame: int, individual: str | None = None) -> int:
        """Discard the predicted points of *frame*; returns how many went.

        The counterpart of :meth:`promote_fill`: where that keeps a prediction,
        this throws one away — for the frames where the animal is occluded or
        out of shot and the backend placed a position anyway. **Labels are left
        alone**, so a row strips back to what the user actually placed.

        The fill is derived data (it is never persisted), so a later fill will
        produce these points again unless the reason they were wrong — a missing
        anchor nearby — has been fixed.
        """
        frame = int(frame)
        predicted = self.predicted_mask(frame)
        if not predicted.any():
            return 0
        if individual is not None:
            keep = np.zeros_like(predicted)
            keep[self.individual_index(individual)] = predicted[self.individual_index(individual)]
            predicted = keep
        self.filled[frame][predicted] = np.nan
        self.confidence[frame][predicted] = np.nan
        return int(predicted.sum())

    def has_fill(self, frame: int, individual: str | None = None) -> bool:
        """Whether a backend produced any point for *individual* on *frame*."""
        predicted = self.predicted_mask(frame)
        if individual is not None:
            predicted = predicted[self.individual_index(individual)]
        return bool(predicted.any())

    def promote_fill(self, frame: int, individual: str | None = None) -> int:
        """Pin the filled points of *frame* as user labels; returns how many.

        "Accepting" a fill, in bulk: every filled point that is not already an
        anchor becomes one exactly where the backend put it, so the next fill
        treats it as ground truth rather than re-deriving it. Points already
        labelled are left alone — a human position always wins over a predicted
        one. Each promotion is its own undo step, as if it had been clicked.
        """
        frame = int(frame)
        if self.filled is None or not 0 <= frame < self.n_frames:
            return 0
        rows = range(self.n_individuals) if individual is None else [self.individual_index(individual)]
        owned = self.keypoint_mask()
        placed = self.anchor_positions(frame)
        promoted = 0
        for i in rows:
            for k, keypoint in enumerate(self.keypoint_names):
                if not owned[i, k] or not np.isnan(placed[i, k, 0]) or np.isnan(self.filled[frame, i, k, 0]):
                    continue
                self.set_point(frame, keypoint, tuple(self.filled[frame, i, k]), self.individual_names[i])
                promoted += 1
        return promoted

    def _drop_if_empty(self, frame: int) -> None:
        points = self.anchors.get(frame)
        if points is not None and not np.any(~np.isnan(points)):
            del self.anchors[frame]

    def _record(self, frame: int, i: int, k: int, previous: np.ndarray) -> None:
        self._history.append((frame, i, k, None if np.all(np.isnan(previous)) else previous.copy()))

    def undo(self) -> int | None:
        """Revert the last ``set_point`` / ``clear_point``; returns its frame.

        The frame is returned rather than remembered, because an undo can land
        anywhere — the caller redraws it, and there is nothing left to go stale.
        """
        if not self._history:
            return None
        frame, i, k, previous = self._history.pop()
        points = self.anchors.get(frame)
        if previous is None:
            if points is not None:
                points[i, k] = np.nan
                self._drop_if_empty(frame)
            return frame
        if points is None:
            points = np.full((self.n_individuals, self.n_keypoints, 2), np.nan, dtype=np.float64)
            self.anchors[frame] = points
        points[i, k] = previous
        return frame

    def clear_fill(self) -> None:
        self.filled = None
        self.confidence = None

    def set_fill(self, filled: np.ndarray, confidence: np.ndarray) -> None:
        """Store a backend result, re-asserting anchors over it.

        Backends are expected to return anchors verbatim; re-applying them here
        makes that invariant hold regardless of the backend.
        """
        filled = np.asarray(filled, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        expected = (self.n_frames, self.n_individuals, self.n_keypoints, 2)
        if filled.shape != expected:
            raise KeypointStoreError(f"filled has shape {filled.shape}, expected {expected}")
        if confidence.shape != expected[:3]:
            raise KeypointStoreError(f"confidence has shape {confidence.shape}, expected {expected[:3]}")
        for frame, points in self.anchors.items():
            if not 0 <= frame < self.n_frames:
                continue
            labelled = ~np.isnan(points[:, :, 0])
            filled[frame][labelled] = points[labelled]
            confidence[frame][labelled] = 1.0
        # Backends track the whole grid; pairs an asymmetric schema excludes are
        # tracked from nothing, so their output is meaningless — blank it out.
        unowned = ~self.keypoint_mask()
        filled[:, unowned] = np.nan
        confidence[:, unowned] = np.nan
        self.filled = filled
        self.confidence = confidence

    def set_fill_from_flat(self, filled: np.ndarray, confidence: np.ndarray) -> None:
        """Store a backend result given as flat points (see :meth:`flat_anchors`)."""
        shape = (self.n_frames, self.n_individuals, self.n_keypoints)
        self.set_fill(np.asarray(filled).reshape(*shape, 2), np.asarray(confidence).reshape(shape))

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def flat_anchors(self) -> dict[int, np.ndarray]:
        """Anchors as ``frame -> (n_points, 2)`` for the fill backends.

        Backends track independent points and neither know nor care which
        individual a point belongs to, so the hierarchy is flattened here and
        restored by :meth:`set_fill_from_flat`.
        """
        return {frame: points.reshape(-1, 2).copy() for frame, points in self.anchors.items()}

    def anchor_frames(self) -> list[int]:
        """Sorted frames carrying at least one labelled point."""
        return sorted(self.anchors)

    def anchor_frames_for(self, keypoint: str, individual: str | None = None) -> list[int]:
        """Sorted frames where this ``(individual, keypoint)`` is labelled."""
        i, k = self.individual_index(individual), self.keypoint_index(keypoint)
        return sorted(f for f, points in self.anchors.items() if not np.isnan(points[i, k, 0]))

    def labelled_points(
        self,
        individual: str | None = None,
        keypoint: str | None = None,
    ) -> list[tuple[int, str, str, float, float]]:
        """Every user-placed point as ``(frame, individual, keypoint, x, y)``.

        Sorted by frame, then by the schema order of individuals and keypoints —
        the order the dialog's table shows. Both filters are optional; passing
        neither returns everything.
        """
        out: list[tuple[int, str, str, float, float]] = []
        for frame in sorted(self.anchors):
            points = self.anchors[frame]
            for i, name in enumerate(self.individual_names):
                if individual is not None and name != individual:
                    continue
                for k, kp in enumerate(self.keypoint_names):
                    if keypoint is not None and kp != keypoint:
                        continue
                    x, y = points[i, k]
                    if not np.isnan(x):
                        out.append((frame, name, kp, float(x), float(y)))
        return out

    def anchor_frames_for_individual(self, individual: str | None = None) -> list[int]:
        """Sorted frames where any keypoint of *individual* is labelled."""
        i = self.individual_index(individual)
        return sorted(f for f, points in self.anchors.items() if np.any(~np.isnan(points[i, :, 0])))

    def positions(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints, 2)``: anchors over fill, else ``NaN``."""
        out = np.full((self.n_individuals, self.n_keypoints, 2), np.nan, dtype=np.float64)
        if self.filled is not None and 0 <= frame < self.n_frames:
            out[:] = self.filled[frame]
        points = self.anchors.get(int(frame))
        if points is not None:
            labelled = ~np.isnan(points[:, :, 0])
            out[labelled] = points[labelled]
        return out

    def positions_for(self, frame: int, individual: str | None = None) -> np.ndarray:
        """``(n_keypoints, 2)`` for one individual: anchors over fill."""
        return self.positions(frame)[self.individual_index(individual)]

    def anchor_positions(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints, 2)`` of user-placed points only."""
        points = self.anchors.get(int(frame))
        if points is None:
            return np.full((self.n_individuals, self.n_keypoints, 2), np.nan, dtype=np.float64)
        return points.copy()

    def anchor_positions_for(self, frame: int, individual: str | None = None) -> np.ndarray:
        """``(n_keypoints, 2)`` of user-placed points for one individual."""
        return self.anchor_positions(frame)[self.individual_index(individual)]

    def human_mask(self, frame: int) -> np.ndarray:
        """``(n_individuals, n_keypoints)`` bool: which points the user placed.

        The complement of this over :meth:`positions` is what a fill backend
        produced, so the two together say where every coordinate came from.
        """
        return ~np.isnan(self.anchor_positions(frame)[:, :, 0])

    def is_anchor(self, frame: int, keypoint: str, individual: str | None = None) -> bool:
        """Whether this ``(individual, keypoint)`` was placed by the user on *frame*."""
        i, k = self.individual_index(individual), self.keypoint_index(keypoint)
        points = self.anchors.get(int(frame))
        return points is not None and not bool(np.isnan(points[i, k, 0]))

    def is_human(self, frame: int, individual: str | None = None) -> bool:
        """Whether *individual* carries any user-placed point on *frame*.

        One labelled or corrected point is enough: a row the user has touched is
        theirs, even if a backend supplied the rest of its keypoints.
        """
        mask = self.human_mask(frame)
        return bool(mask[self.individual_index(individual)].any() if individual is not None else mask.any())

    def labelled_count(self, frame: int, individual: str | None = None) -> int:
        """Keypoints labelled on *frame* — for one individual, or all of them."""
        points = self.anchor_positions(frame)
        if individual is not None:
            points = points[self.individual_index(individual)][None]
        return int(np.count_nonzero(~np.isnan(points[..., 0])))

    def nearest(
        self,
        frame: int,
        xy: tuple[float, float],
        radius: float,
        include_fill: bool = False,
    ) -> tuple[str, str] | None:
        """``(individual, keypoint)`` of the point within *radius* px of ``xy``.

        Anchors only by default. With *include_fill* the filled points are
        candidates too, which is what makes a prediction grabbable on the canvas;
        deleting deliberately does not, since only an anchor can be removed.
        """
        points = self.positions(int(frame)) if include_fill else self.anchors.get(int(frame))
        if points is None or points.size == 0:
            return None
        distances = np.hypot(points[:, :, 0] - xy[0], points[:, :, 1] - xy[1])
        distances[np.isnan(distances)] = np.inf
        i, k = np.unravel_index(int(np.argmin(distances)), distances.shape)
        if distances[i, k] > radius:
            return None
        return self.individual_names[i], self.keypoint_names[k]

    # ------------------------------------------------------------------
    # Sidecar persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        payload = {
            "keypoint": list(self.keypoint_names),
            "individual": list(self.individual_names),
            "shared_keypoints": self.shared_keypoints,
            "n_frames": self.n_frames,
            "anchors": {str(f): points.tolist() for f, points in sorted(self.anchors.items())},
        }
        if not self.shared_keypoints:
            payload["keypoint_set"] = {name: list(kps) for name, kps in self.keypoint_sets.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> KeypointStore:
        """Rebuild a store from a sidecar payload, migrating older formats.

        Sidecars written before the individual/keypoint hierarchy used a
        ``"names"`` key and 2-D ``(n_keypoints, 2)`` anchor rows. Those load as
        one individual: the names stay keypoints, because reinterpreting them
        would silently change what the user labelled.
        """
        keypoints = payload.get("keypoint", payload.get("names"))
        if keypoints is None:
            raise KeypointStoreError("Sidecar has no keypoint names — expected a 'keypoint' or 'names' key.")
        keypoints = list(keypoints)
        # A missing key is a legacy sidecar (one individual); an empty list is a
        # user who deleted every individual, and must stay empty.
        saved_individuals = payload.get("individual")
        individuals = [DEFAULT_INDIVIDUAL] if saved_individuals is None else list(saved_individuals)
        expected = (len(individuals), len(keypoints), 2)

        anchors: dict[int, np.ndarray] = {}
        for frame, points in payload.get("anchors", {}).items():
            array = np.asarray(points, dtype=np.float64)
            if array.ndim == 2:  # legacy flat layout — one individual
                array = array[None, ...]
            if array.shape != expected:
                raise KeypointStoreError(
                    f"Sidecar anchor for frame {frame} has shape {array.shape}, expected {expected}."
                )
            anchors[int(frame)] = array

        # Unknown keys are ignored, so sidecars carrying the old
        # "last_labelled_frame" still load; it simply has no meaning any more.
        return cls(
            keypoint_names=keypoints,
            n_frames=int(payload["n_frames"]),
            individual_names=individuals,
            shared_keypoints=bool(payload.get("shared_keypoints", True)),
            keypoint_sets={str(k): list(v) for k, v in (payload.get("keypoint_set") or {}).items()},
            anchors=anchors,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> KeypointStore:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def sidecar_path(video_path: str | Path) -> Path:
    """Anchor sidecar location for *video_path* (``<video>.keypoints.json``)."""
    video = Path(video_path)
    return video.with_name(video.name + SIDECAR_SUFFIX)


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


def store_to_movement_ds(store: KeypointStore, fps: float, image_height: float | None = None) -> xr.Dataset:
    """Build a movement-format poses dataset from *store*.

    Dims are ``(time, space, keypoint, individual)`` — singular, as movement
    ≥0.17 and the rest of EthoGraph name them — so the result feeds the existing
    ``PoseRenderData`` path and everything downstream (overlay, filtering,
    kinematics, NWB) unchanged. Time is in seconds; ``fps`` must come from the
    video, never a default.

    ``image_height`` flips y to a **y-up** convention (``y_out = height - y``).
    Anchors are stored in image coordinates, where y grows *downward* from the
    top-left corner — the convention every pose file uses. Plots are y-up, so
    an unflipped trajectory comes out vertically mirrored: a keypoint at the top
    of the frame is drawn at the bottom of the plot. The canvas overlays handle
    this themselves (``y_world = img_height - y``) and DeepLabCut expects raw
    image coordinates, so the flip belongs only on the paths that hand data to a
    plot. Leave it ``None`` to keep image coordinates.
    """
    if fps <= 0:
        raise KeypointStoreError("fps must be positive — read it from the video, do not default it.")
    if image_height is not None and image_height <= 0:
        raise KeypointStoreError("image_height must be positive — read it from the video, do not default it.")

    n_ind, n_kp = store.n_individuals, store.n_keypoints
    position = np.full((store.n_frames, 2, n_kp, n_ind), np.nan, dtype=np.float64)
    confidence = np.full((store.n_frames, n_kp, n_ind), np.nan, dtype=np.float64)

    if store.filled is not None:
        # store: (time, individual, keypoint, space) -> ds: (time, space, keypoint, individual)
        position[:] = store.filled.transpose(0, 3, 2, 1)
    if store.confidence is not None:
        confidence[:] = store.confidence.transpose(0, 2, 1)

    for frame, points in store.anchors.items():
        if not 0 <= frame < store.n_frames:
            continue
        labelled = ~np.isnan(points[:, :, 0])
        for i, k in zip(*np.nonzero(labelled)):
            position[frame, :, k, i] = points[i, k]
            confidence[frame, k, i] = 1.0

    if image_height is not None:
        position[:, 1] = image_height - position[:, 1]

    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(position, dims=["time", "space", "keypoint", "individual"]),
            "confidence": xr.DataArray(confidence, dims=["time", "keypoint", "individual"]),
        },
        coords={
            "time": np.arange(store.n_frames) / fps,
            "space": ["x", "y"],
            "keypoint": list(store.keypoint_names),
            "individual": list(store.individual_names),
        },
        attrs={"ds_type": "poses", "fps": float(fps), "source_software": "ethograph"},
    )


def store_to_dlc_dataframe(store: KeypointStore, scorer: str, video_name: str) -> pd.DataFrame:
    """Anchor frames as a DeepLabCut ``CollectedData`` DataFrame.

    Only labelled frames are included — this is training data for DLC, not a
    prediction table. The index holds the image paths DLC expects; extracting
    those PNGs is DLC's job (``deeplabcut.extract_frames``), not EthoGraph's.

    With more than one individual the multi-animal layout is written, which
    carries DLC's own ``individuals`` column level. Those level names are part
    of the DLC format and are spelled the way DLC spells them.
    """
    frames = store.anchor_frames()
    if store.n_individuals > 1:
        columns = pd.MultiIndex.from_product(
            [[scorer], store.individual_names, store.keypoint_names, ["x", "y"]],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
    else:
        columns = pd.MultiIndex.from_product(
            [[scorer], store.keypoint_names, ["x", "y"]],
            names=["scorer", "bodyparts", "coords"],
        )
    index = pd.MultiIndex.from_tuples(
        [("labeled-data", video_name, f"img{frame:05d}.png") for frame in frames],
        names=[None, None, None],
    )
    values = np.full((len(frames), len(columns)), np.nan)
    for row, frame in enumerate(frames):
        values[row] = store.anchors[frame].reshape(-1)
    return pd.DataFrame(values, index=index, columns=columns)


def store_to_dlc_h5(store: KeypointStore, path: str | Path, scorer: str, video_name: str) -> Path:
    """Write ``CollectedData_<scorer>.h5`` (a single file, per DLC convention)."""
    df = store_to_dlc_dataframe(store, scorer, video_name)
    path = Path(path)
    df.to_hdf(path, key="df_with_missing", mode="w")
    return path


# ----------------------------------------------------------------------
# Derived kinematics, for inspecting a fill without leaving the GUI
# ----------------------------------------------------------------------

#: Quantities derivable from the labelled + filled keypoint trajectories.
KINEMATICS = ("velocity", "speed", "acceleration")

#: Prefix for the feature names injected into the GUI's dataset.
FEATURE_PREFIX = "keypoints"

#: The poses dataset's own time axis, renamed before it is merged into a trial.
#: It runs at the video frame rate over ``n_frames``, which almost never matches
#: the trial's ``time``; merging under the same name would outer-join the two
#: and pad every other feature with NaN. Any name containing "time" is treated
#: as a time coordinate by ``eto.get_time_coord``.
FEATURE_TIME_DIM = "time_keypoints"


def store_to_kinematics(ds: xr.Dataset, features: Sequence[str] = KINEMATICS) -> dict[str, xr.DataArray]:
    """Derive kinematics from a poses dataset's ``position``.

    Uses ``movement.kinematics``, which needs only ``time`` and ``space`` dims —
    the ``keypoint`` and ``individual`` axes pass straight through, so every
    keypoint of every individual is differentiated independently.

    Filled frames are included: the point of computing these in the GUI is to
    inspect what a fill produced, not only the frames that were labelled by hand.
    """
    from movement import kinematics

    unknown = set(features) - set(KINEMATICS)
    if unknown:
        raise KeypointStoreError(f"Unknown kinematic(s) {sorted(unknown)}; expected {list(KINEMATICS)}")

    computed: dict[str, xr.DataArray] = {"position": ds["position"]}
    for name in features:
        computed[name] = getattr(kinematics, f"compute_{name}")(ds["position"])
    return {
        f"{FEATURE_PREFIX}_{name}": array.rename({"time": FEATURE_TIME_DIM}).rename(f"{FEATURE_PREFIX}_{name}")
        for name, array in computed.items()
    }
