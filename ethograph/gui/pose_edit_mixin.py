"""Canvas keypoint editing: place, drag and delete anchors on a camera view.

:class:`KeypointLabelMode` is attached to a :class:`~ethograph.gui.pygfx_video.CameraView`
while the labelling dialog is open. It owns the pointer interaction and a small
pygfx overlay that draws the *anchors* of the current frame — deliberately
distinct from the pose overlay that shows filled predictions: anchors are drawn
as large markers with a hairline white outline around the active one, so a
labelled point is never confused with a model output.

Labelling is hierarchical (see :mod:`~ethograph.gui.pose_annotate`): the active
target is an ``(individual, keypoint)`` pair. The two axes of the hierarchy get
two *different* visual channels, so both stay readable at once: **shape encodes
the individual** (circle, triangle, square, …) and **colour encodes the
keypoint**. Encoding both with colour, as the pose display does, cannot show
which beak belongs to which animal. Individuals other than the active one are
dimmed, and the active point carries a thin white outline.

Editing is always available
--------------------------
There is no separate "edit" mode. A click on empty space places the active
keypoint; a click on an existing point selects it (switching to its individual
and keypoint) and drags it. ``Backspace`` / ``Delete`` remove the *selected*
point — the one the white outline is drawn around, falling back to whatever is
under the cursor — and ``Ctrl+Z`` undoes, always: correcting a mistake should
never require changing mode first.

*Filled* points are grabbable too, and grabbing one pins it as a label (see
:meth:`KeypointStore.promote_fill`) — that is how a prediction is accepted or
corrected. Deleting stays anchors-only: there is nothing to remove from a
prediction, and clearing the label under it simply hands the point back to the
next fill. The anchor overlay still draws labels *only*; predictions are the
ordinary pose overlay's job, so the two never look alike.

Two labelling modes, after napari-deeplabcut
--------------------------------------------
They differ only in what happens *after* a point is placed:

``"sequential"``
    Advance to the *first* keypoint this individual still lacks on this frame,
    in schema order — which is the left-to-right column order of the dialog's
    points table, so that table fills from the left. The playhead never moves on
    its own: you label a frame, then navigate yourself.
``"loop"``
    Keep the same keypoint and jump to the next frame. Sweeping one keypoint
    across many frames is what the fill backends actually want, since each
    keypoint is interpolated over its *own* anchor set.

``Tab`` cycles the keypoint and ``1``–``9`` select the individual in both modes;
the dialog owns the key handling.

While a mode is active, left-drag pans no longer — panning moves to ``Shift`` +
left-drag so the wheel/zoom controls keep working.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pygfx as gfx

from ethograph.gui.pose_annotate import KeypointStore
from ethograph.gui.pose_convert import sample_colormap

#: Sentinel position far outside any frame, used to hide unlabelled markers.
_OFFSCREEN = -1.0e6

_Z_ANCHORS = 4.0
_Z_ACTIVE = 4.2
_Z_TEXT = 4.5

#: Screen-pixel radius for hit-testing an existing anchor.
HIT_RADIUS_PX = 12.0

#: Default marker diameter in screen pixels; the dialog can override it.
_ANCHOR_SIZE = 16.0
#: The active point's ring is drawn this much larger than the markers.
_ACTIVE_SIZE_RATIO = 22.0 / 16.0
_ACTIVE_EDGE = (1.0, 1.0, 1.0, 1.0)
_IDLE_EDGE = (0.0, 0.0, 0.0, 0.8)

#: The active marker is an outline only — a transparent fill with a hairline
#: edge. pygfx's "ring" marker draws a filled donut whose thickness is fixed, so
#: it swamped the keypoint underneath; a circle with no fill leaves just the edge.
_ACTIVE_FILL = (0.0, 0.0, 0.0, 0.0)
_ACTIVE_EDGE_WIDTH = 1.0

#: Alpha applied to the individuals that are not currently being edited.
_INACTIVE_ALPHA = 0.35

#: Marker shape per individual, in display order — the individual axis of the
#: hierarchy. Colour is spent on keypoints instead, so the two never collide.
#: Every name here must exist in ``pygfx.utils.enums.MarkerShape``.
MARKER_SHAPES = (
    "circle",
    "triangle_up",
    "square",
    "diamond",
    "triangle_down",
    "plus",
    "cross",
    "heart",
    "spade",
)

#: Text stand-ins for :data:`MARKER_SHAPES`, so the dialog's tree can show the
#: same shape the canvas draws.
MARKER_GLYPHS = {
    "circle": "●",
    "triangle_up": "▲",
    "square": "■",
    "diamond": "◆",
    "triangle_down": "▼",
    "plus": "✚",
    "cross": "✖",
    "heart": "♥",
    "spade": "♠",
}

#: The interaction modes; see the module docstring.
#: Label every keypoint on one frame; the playhead never moves on its own.
SEQUENTIAL_MODE = "sequential"
#: Label one keypoint, then jump straight to the next frame.
LOOP_MODE = "loop"


def marker_for_individual(index: int) -> str:
    """pygfx marker shape for the *index*-th individual (wraps around)."""
    return MARKER_SHAPES[index % len(MARKER_SHAPES)]


def glyph_for_individual(index: int) -> str:
    """Unicode stand-in for :func:`marker_for_individual`."""
    return MARKER_GLYPHS[marker_for_individual(index)]


def keypoint_colors(n: int) -> np.ndarray:
    """``(n, 4)`` RGBA, one distinct colour per keypoint."""
    colors = sample_colormap(max(n, 1), "turbo")
    return np.array([c if len(c) == 4 else (*c, 1.0) for c in colors], dtype=np.float32)[:n]


class AnchorOverlay:
    """pygfx markers + labels for the anchors of one frame.

    One :class:`pygfx.Points` layer per individual — a marker *shape* is a
    material property, not a per-vertex one, so the shape-per-individual
    encoding needs one object each. Within a layer, vertices are coloured per
    keypoint. Keypoint names are drawn for the active individual only (with a
    full schema on several individuals the canvas is otherwise unreadable),
    while each individual's own name sits at the centroid of its labelled points.
    """

    def __init__(self, scene: gfx.Scene, keypoint_names: list[str], individual_names: list[str], img_height: float):
        self._scene = scene
        self._keypoint_names = list(keypoint_names)
        self._individual_names = list(individual_names)
        self._img_height = float(img_height)

        n_kp = len(self._keypoint_names)
        self._colors = keypoint_colors(n_kp)
        self._layers: list[gfx.Points] = []
        # No layers when there is nothing to draw: a zero-vertex geometry is not
        # worth defending against downstream.
        if n_kp:
            self._layers = [self._add_layer(i, n_kp) for i in range(len(self._individual_names))]

        # The active point is a separate single-vertex object so it can carry a
        # white outline; marker materials only expose one edge colour for all
        # vertices.
        self._active = gfx.Points(
            gfx.Geometry(positions=np.full((1, 3), _OFFSCREEN, dtype=np.float32)),
            gfx.PointsMarkerMaterial(
                size=_ANCHOR_SIZE * _ACTIVE_SIZE_RATIO,
                marker="circle",
                color=_ACTIVE_FILL,
                edge_width=_ACTIVE_EDGE_WIDTH,
                edge_color=_ACTIVE_EDGE,
            ),
        )
        self._active.local.z = _Z_ACTIVE
        scene.add(self._active)

        self._keypoint_texts = [self._add_text(name, 11) for name in self._keypoint_names]
        # The glyph repeats the marker shape, so the canvas itself says which
        # shape is which animal.
        self._individual_texts = (
            [self._add_text(f"{glyph_for_individual(i)} {name}", 13) for i, name in enumerate(self._individual_names)]
            if len(self._individual_names) > 1
            else []
        )

    def _add_layer(self, index: int, n_kp: int) -> gfx.Points:
        """One markers object for the *index*-th individual, in its own shape."""
        layer = gfx.Points(
            gfx.Geometry(
                positions=np.full((n_kp, 3), _OFFSCREEN, dtype=np.float32),
                colors=self._colors.copy(),
            ),
            gfx.PointsMarkerMaterial(
                size=_ANCHOR_SIZE,
                size_space="screen",  # constant on screen, unaffected by zoom
                marker=marker_for_individual(index),
                color_mode="vertex",
                edge_width=2.0,
                edge_color=_IDLE_EDGE,
            ),
        )
        layer.local.z = _Z_ANCHORS
        self._scene.add(layer)
        return layer

    def _add_text(self, name: str, size: int) -> gfx.Text:
        text = gfx.Text(
            text=str(name),
            font_size=size,
            screen_space=True,
            anchor="bottom-left",
            material=gfx.TextMaterial(color=gfx.Color(1.0, 1.0, 1.0)),
        )
        text.local.z = _Z_TEXT
        text.visible = False
        self._scene.add(text)
        return text

    def set_positions(self, positions: np.ndarray, active_individual: int, active_keypoint: int) -> None:
        """Draw ``(n_individuals, n_keypoints, 2)`` image-space anchors.

        ``NaN`` hides a marker. *active_individual* / *active_keypoint* index
        the pair the next click will write to.
        """
        n_ind, n_kp = positions.shape[0], positions.shape[1]
        if not self._layers or n_kp == 0 or n_ind == 0:
            self._active.visible = False
            for layer in self._layers:
                layer.visible = False
            for text in [*self._keypoint_texts, *self._individual_texts]:
                text.visible = False
            return

        world = positions.copy()
        world[:, :, 1] = self._img_height - world[:, :, 1]
        shown = ~np.isnan(positions[:, :, 0])

        for i, layer in enumerate(self._layers):
            layer.visible = True
            buffer = layer.geometry.positions
            buffer.data[:, 2] = 0.0
            buffer.data[:, :2] = np.where(shown[i][:, None], world[i], _OFFSCREEN)
            buffer.update_full()

            colors = self._colors.copy()
            if n_ind > 1 and i != active_individual:
                colors[:, 3] = _INACTIVE_ALPHA
            color_buffer = layer.geometry.colors
            color_buffer.data[:] = colors
            color_buffer.update_full()

        active_buffer = self._active.geometry.positions
        active_shown = shown[active_individual, active_keypoint] if n_kp else False
        active_buffer.data[0, :2] = world[active_individual, active_keypoint] if active_shown else _OFFSCREEN
        active_buffer.data[0, 2] = 0.0
        active_buffer.update_full()
        self._active.visible = bool(active_shown)

        for k, text in enumerate(self._keypoint_texts):
            text.visible = bool(shown[active_individual, k])
            if text.visible:
                x, y = world[active_individual, k]
                text.local.position = (float(x) + 4, float(y) + 4, _Z_TEXT)

        for i, text in enumerate(self._individual_texts):
            visible = bool(np.any(shown[i]))
            text.visible = visible
            if visible:
                centre = np.nanmean(world[i][shown[i]], axis=0)
                text.local.position = (float(centre[0]) + 6, float(centre[1]) - 14, _Z_TEXT)

    def set_point_size(self, size: float) -> None:
        """Resize every marker in place — no rebuild, so a spinbox can drive it."""
        for layer in self._layers:
            layer.material.size = float(size)
        self._active.material.size = float(size) * _ACTIVE_SIZE_RATIO

    def clear(self) -> None:
        for obj in [*self._layers, self._active, *self._keypoint_texts, *self._individual_texts]:
            self._scene.remove(obj)
        self._layers = []
        self._keypoint_texts = []
        self._individual_texts = []


class KeypointLabelMode:
    """Pointer-driven anchor editing on one camera view."""

    def __init__(
        self,
        view,
        store: KeypointStore,
        on_changed: Callable[[], None] | None = None,
        mode: str = SEQUENTIAL_MODE,
        on_advance_frame: Callable[[], None] | None = None,
        on_released: Callable[[], None] | None = None,
        point_size: float = _ANCHOR_SIZE,
    ):
        self.view = view
        self.store = store
        self.on_changed = on_changed or (lambda: None)
        #: Called by LOOP mode after a placement — the dialog owns navigation.
        self.on_advance_frame = on_advance_frame or (lambda: None)
        #: Called when the pointer is released, so work too heavy for every
        #: mouse move of a drag can happen once the edit has settled.
        self.on_released = on_released or (lambda: None)
        self.mode = mode
        self.frame = 0
        #: The active pair is held by name, not by index: with per-individual
        #: keypoint sets an index means different things on different branches,
        #: and with no individuals at all it means nothing.
        self._individual: str | None = None
        self._keypoint: str | None = None
        self._dragging: tuple[str, str] | None = None
        #: True once the current drag has written a point, so further motion
        #: collapses into that single undo step instead of popping an unrelated one.
        self._drag_recorded = False
        self._cursor: tuple[float, float] | None = None

        scene = view.scene()
        if scene is None:
            raise ValueError("Camera view has no scene to draw annotations on.")
        self._overlay = AnchorOverlay(scene, store.keypoint_names, store.individual_names, view.image_height())
        self.point_size = float(point_size)
        self._overlay.set_point_size(self.point_size)
        view.set_label_mode(self)
        self._sync_active()
        self.refresh()

    # ------------------------------------------------------------------
    # Active individual + keypoint
    # ------------------------------------------------------------------

    @property
    def active_individual(self) -> str | None:
        """The individual clicks write to, or ``None`` when there are none."""
        return self._individual

    @property
    def active_keypoints(self) -> list[str]:
        """The schema of the active individual — what ``Tab`` cycles through."""
        return [] if self._individual is None else self.store.keypoints_for(self._individual)

    @property
    def active_keypoint(self) -> str | None:
        return self._keypoint

    def _sync_active(self) -> None:
        """Keep the active pair pointing at something the schema still holds."""
        if self._individual not in self.store.individual_names:
            self._individual = self.store.individual_names[0] if self.store.individual_names else None
        keypoints = self.active_keypoints
        if self._keypoint not in keypoints:
            self._keypoint = keypoints[0] if keypoints else None

    def set_active(self, keypoint: str, individual: str | None = None) -> None:
        if individual is not None:
            self._individual = self.store.individual_names[self.store.individual_index(individual)]
        self._keypoint = self.store.keypoint_names[self.store.keypoint_index(keypoint)]
        self._sync_active()
        self.refresh()

    def set_active_individual(self, individual: str) -> None:
        self._individual = self.store.individual_names[self.store.individual_index(individual)]
        self._sync_active()
        self.refresh()

    def cycle(self, step: int = 1) -> None:
        """Move to the next/previous keypoint of the active individual."""
        keypoints = self.active_keypoints
        if not keypoints:
            return
        index = keypoints.index(self._keypoint) if self._keypoint in keypoints else 0
        self._keypoint = keypoints[(index + step) % len(keypoints)]
        self.refresh()

    def cycle_individual(self, step: int = 1) -> None:
        names = self.store.individual_names
        if not names:
            return
        index = names.index(self._individual) if self._individual in names else 0
        self._individual = names[(index + step) % len(names)]
        self._sync_active()
        self.refresh()

    def select_individual_by_number(self, number: int) -> bool:
        """Select the *number*-th individual (1-based, SLEAP's number keys)."""
        if not 1 <= number <= self.store.n_individuals:
            return False
        self._individual = self.store.individual_names[number - 1]
        self._sync_active()
        self.refresh()
        return True

    def _advance_to_unlabelled(self) -> None:
        """Move to the FIRST keypoint this individual still lacks on this frame.

        Ordered by the schema, which is exactly the left-to-right column order of
        the dialog's points table — so sequential labelling fills that table from
        the left, and jumping around out of order still comes back to the
        leftmost gap rather than continuing from wherever the last click landed.

        When the frame is complete the active keypoint stays put: there is
        nothing left to advance to, and clicking an existing point grabs it
        rather than overwriting, so nothing is at risk.
        """
        keypoints = self.active_keypoints
        if not keypoints:
            return
        placed = self.store.anchor_positions_for(self.frame, self._individual)[:, 0]
        for name in sorted(keypoints, key=self.store.keypoint_index):
            if np.isnan(placed[self.store.keypoint_index(name)]):
                self._keypoint = name
                return

    # ------------------------------------------------------------------
    # Pointer handling (called by CameraView)
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Switch what happens after a point is placed."""
        if mode not in (SEQUENTIAL_MODE, LOOP_MODE):
            raise ValueError(f"Unknown labelling mode {mode!r}")
        self.mode = mode
        self._dragging = None
        self._drag_recorded = False
        self.refresh()

    def handle_click(self, x: float, y: float) -> None:
        """Place the active keypoint, or grab an existing point to move it.

        Correcting is always available — clicking a point that is already there
        selects and drags it rather than dropping a second one on top. Filled
        points count as "there": grabbing one **pins it** as a label where the
        backend put it, so a prediction that is already right is accepted by
        clicking it and a wrong one is fixed by dragging it. Either way the next
        fill sees a human point, since fills are re-derived from labels alone.
        """
        self._cursor = (x, y)
        existing = self.store.nearest(self.frame, (x, y), self._hit_radius(), include_fill=True)
        if existing is not None:
            individual, keypoint = existing
            self._individual, self._keypoint = existing
            self._dragging = existing
            self._drag_recorded = False
            if not self.store.is_anchor(self.frame, keypoint, individual):
                position = self.store.positions_for(self.frame, individual)[self.store.keypoint_index(keypoint)]
                self.store.set_point(self.frame, keypoint, tuple(position), individual)
                # Recorded, so dragging on from here collapses into this one
                # undo step rather than popping something unrelated.
                self._drag_recorded = True
            self._changed()
            return

        keypoint, individual = self.active_keypoint, self._individual
        if keypoint is None or individual is None:
            return
        self.store.set_point(self.frame, keypoint, (x, y), individual)
        self._dragging = (individual, keypoint)
        self._drag_recorded = True
        if self.mode == LOOP_MODE:
            # Same keypoint, next frame. The dialog owns navigation, so it
            # decides whether "next" means the next suggestion or the next frame.
            self._changed()
            self.on_advance_frame()
            return
        self._advance_to_unlabelled()
        self._changed()

    def handle_move(self, x: float, y: float) -> None:
        self._cursor = (x, y)
        if self._dragging is None:
            return
        if self._drag_recorded:
            self.store.undo()  # collapse the whole drag into one undo step
        individual, keypoint = self._dragging
        self.store.set_point(self.frame, keypoint, (x, y), individual)
        self._drag_recorded = True
        self._changed()

    @property
    def dragging(self) -> bool:
        """Whether a point is currently being moved by the pointer."""
        return self._dragging is not None

    def handle_release(self, x: float, y: float) -> None:
        self._cursor = (x, y)
        self._dragging = None
        self._drag_recorded = False
        self.on_released()

    def delete_selected(self) -> bool:
        """What ``Backspace`` removes: the active point, else the one hovered.

        The active point is what the white outline is drawn around, so deleting
        it is what the canvas already promises. Falling back to the cursor keeps
        hover-and-delete working for points that are not the active one, and
        matters most when the active pair is unlabelled on this frame.
        """
        return self.delete_active() or self.delete_under_cursor()

    def delete_active(self) -> bool:
        """Delete the active ``(individual, keypoint)`` on this frame."""
        if self._individual is None or self._keypoint is None:
            return False
        placed = self.store.anchor_positions_for(self.frame, self._individual)
        if np.isnan(placed[self.store.keypoint_index(self._keypoint), 0]):
            return False
        self.store.clear_point(self.frame, self._keypoint, self._individual)
        self._changed()
        return True

    def delete_under_cursor(self) -> bool:
        """Delete the anchor nearest the cursor; ``True`` if one was removed."""
        if self._cursor is None:
            return False
        target = self.store.nearest(self.frame, self._cursor, self._hit_radius())
        if target is None:
            return False
        individual, keypoint = target
        self.store.clear_point(self.frame, keypoint, individual)
        self._changed()
        return True

    def set_point_size(self, size: float) -> None:
        """Change the marker size; also widens hit-testing (see _hit_radius)."""
        self.point_size = float(size)
        self._overlay.set_point_size(self.point_size)
        self.view.request_draw()

    def _hit_radius(self) -> float:
        """Grab radius in image units.

        Both terms are screen pixels — the markers are sized in screen space —
        converted through the current zoom, so the radius always matches what
        the user sees. It never shrinks below :data:`HIT_RADIUS_PX`, and grows
        with the marker so a click inside a large one selects it.
        """
        screen_px = max(HIT_RADIUS_PX, self.point_size / 2.0)
        return screen_px * self.view.image_units_per_pixel()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def set_frame(self, frame: int) -> None:
        self.frame = int(frame)
        self._dragging = None
        self._drag_recorded = False
        self.refresh()

    def refresh(self) -> None:
        self._sync_active()
        self._overlay.set_positions(
            self.store.anchor_positions(self.frame),
            self.store.individual_index(self._individual) if self._individual is not None else 0,
            self.store.keypoint_index(self._keypoint) if self._keypoint is not None else 0,
        )
        self.view.request_draw()

    def _changed(self) -> None:
        self.refresh()
        self.on_changed()

    def detach(self) -> None:
        self._overlay.clear()
        self.view.set_label_mode(None)
        self.view.request_draw()
