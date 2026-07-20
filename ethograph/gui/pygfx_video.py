"""Pygfx-backed camera view widgets (napari replacement for video display).

``CameraView`` hosts one camera's video via pynaviz's :class:`PlotVideo`
(shared-memory worker process decoding, pygfx rendering) plus a
:class:`~ethograph.gui.pose_overlay.PoseOverlay` in the same scene.

Seeking model:
- Programmatic seeks come from :class:`~ethograph.gui.video_sync.VideoSync`
  through ``seek_video_frame`` / ``seek_to_time`` and are *suppressed* — they
  do not re-emit ``time_changed``.
- User interaction on the canvas (scroll = frame step, arrow keys) triggers
  pynaviz sync events on the plot's renderer; the view forwards those as
  ``time_changed(video_time)`` so VideoSync can propagate to plots and other
  cameras.
"""

from __future__ import annotations

import queue
from typing import Optional

import numpy as np
import pygfx as gfx
from pynaviz.audiovideo import PlotVideo
from pynaviz.utils import RenderTriggerSource
from qtpy.QtCore import QEvent, Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from .pose_overlay import PoseOverlay


class _StaticImagePlot:
    """Minimal pygfx scene showing a single still image (no video worker)."""

    def __init__(self, img: np.ndarray, parent: QWidget):
        from rendercanvas.qt import RenderCanvas

        self.canvas = RenderCanvas(parent=parent)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.scene.add(gfx.Background.from_color("black"))

        data = np.asarray(img)
        if data.ndim == 2:
            data = np.repeat(data[:, :, None], 3, axis=2)
        data = data[::-1].astype("float32")
        if data.max() > 1.0:
            data = data / 255.0
        self.texture = gfx.Texture(data, dim=2)
        self.image = gfx.Image(
            gfx.Geometry(grid=self.texture),
            gfx.ImageBasicMaterial(clim=(0, 1)),
        )
        self.scene.add(self.image)
        self.camera = gfx.OrthographicCamera(maintain_aspect=True)
        self.camera.show_object(self.scene)
        self.controller = gfx.PanZoomController(self.camera, register_events=self.renderer)
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def request_draw(self):
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def close(self):
        try:
            self.canvas.close()
        except Exception:
            pass


class CameraView(QWidget):
    """One camera: pygfx video canvas + pose overlay + Qt label overlays."""

    time_changed = Signal(float)  # video-native time, from user interaction
    clicked = Signal()  # any mouse press in this camera view (for active-panel)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        # 2px margin so the active-panel green edge is visible around the video.
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        self.setLayout(layout)

        self._plot: Optional[PlotVideo] = None
        self._static: Optional[_StaticImagePlot] = None
        self._overlay: Optional[PoseOverlay] = None
        #: Set for static-image views: source file + fps of the pose shown on top.
        self.static_image_path: Optional[str] = None
        self.static_pose_fps: float = 0.0
        self._fps: float = 0.0
        self._time_offset: float = 0.0
        self._start_frame: int = 0
        self._end_frame: int = 0
        self._suppress_sync = False
        self.setMinimumSize(120, 90)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def time_offset(self) -> float:
        return self._time_offset

    @property
    def start_frame(self) -> int:
        return self._start_frame

    @property
    def n_frames(self) -> int:
        """Number of frames in the trial-clipped range."""
        return max(0, self._end_frame - self._start_frame)

    @property
    def has_video(self) -> bool:
        return self._plot is not None

    @property
    def plot(self) -> Optional[PlotVideo]:
        return self._plot

    def canvas_widget(self) -> QWidget | None:
        if self._plot is not None:
            return self._plot.canvas
        if self._static is not None:
            return self._static.canvas
        return None

    def set_video(
        self,
        video_path: str,
        fps: float,
        time_offset: float = 0.0,
        start_frame: int = 0,
        end_frame: int | None = None,
    ) -> None:
        """Load a video file. Frame indices used by callers are trial frames
        (0 = ``start_frame`` in the underlying video)."""
        self.clear()
        self._plot = PlotVideo(video=video_path, parent=self)
        self.layout().addWidget(self._plot.canvas)
        self._fit_image_to_canvas(self._plot)
        self._fps = float(fps)
        total = self._plot.data.shape[0]
        self._start_frame = max(0, int(start_frame))
        self._end_frame = int(end_frame) if end_frame is not None else total
        self._end_frame = min(self._end_frame, total)
        self._time_offset = float(time_offset)

        self._plot.renderer.add_event_handler(self._on_sync_event, "sync")
        # Route worker-thread frame updates into the pose overlay as well.
        original_update_extra = self._plot._update_extra_objects

        def _update_extra(frame_index, event_type=None):
            original_update_extra(frame_index, event_type)
            if self._overlay is not None:
                self._overlay.set_frame(frame_index - self._start_frame)

        self._plot._update_extra_objects = _update_extra
        # Clicks on the pygfx canvas go through the wgpu renderer, NOT Qt — so a
        # Qt event filter never sees them. Hook the renderer's pointer event to
        # emit `clicked` (used by the active-panel manager for the green edge).
        self._plot.renderer.add_event_handler(self._on_pointer_down, "pointer_down")
        self._install_click_filter(self._plot.canvas)

    def set_static_image(self, img: np.ndarray) -> None:
        """Show a still frame (pose-only mode, no video)."""
        self.clear()
        self._static = _StaticImagePlot(img, parent=self)
        self.layout().addWidget(self._static.canvas)
        self._static.renderer.add_event_handler(self._on_pointer_down, "pointer_down")
        self._install_click_filter(self._static.canvas)

    @staticmethod
    def _fit_image_to_canvas(plot) -> None:
        """Frame the exact video rectangle so it fills the canvas (up to the
        aspect-ratio letterbox). pygfx's ``show_object`` fits the bounding SPHERE
        (half-diagonal), which always leaves black padding; ``show_rect`` frames
        the image rectangle itself."""
        try:
            w, h = int(plot.texture.size[0]), int(plot.texture.size[1])
            plot.camera.show_rect(0, w, 0, h)
            plot.controller.renderer_request_draw()
        except Exception:  # noqa: BLE001 - framing is best-effort
            pass

    def _on_pointer_down(self, event=None) -> None:
        self.clicked.emit()

    def _install_click_filter(self, canvas) -> None:
        """Also catch Qt clicks (e.g. on the 2px margin around the canvas)."""
        try:
            canvas.installEventFilter(self)
        except (RuntimeError, AttributeError):
            pass

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            self.clicked.emit()
        return False

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    @property
    def overlay(self) -> PoseOverlay | None:
        return self._overlay

    def ensure_overlay(self) -> PoseOverlay | None:
        if self._overlay is not None:
            return self._overlay
        scene = None
        if self._plot is not None:
            scene = self._plot.scene
        elif self._static is not None:
            scene = self._static.scene
        if scene is None:
            return None
        self._overlay = PoseOverlay(scene)
        return self._overlay

    def image_height(self) -> float:
        if self._plot is not None:
            return float(self._plot.texture.size[1])
        if self._static is not None:
            return float(self._static.texture.size[1])
        return 0.0

    def clear_overlay(self) -> None:
        if self._overlay is not None:
            self._overlay.clear()
        self.request_draw()

    # ------------------------------------------------------------------
    # Seeking
    # ------------------------------------------------------------------

    def current_video_frame(self) -> int:
        if self._plot is None:
            return 0
        return int(self._plot.controller.frame_index)

    def seek_video_frame(self, video_frame: int, synchronous: bool = False) -> None:
        """Seek to an absolute video frame without re-emitting time_changed."""
        if self._plot is None:
            return
        total = self._plot.data.shape[0]
        video_frame = max(0, min(int(video_frame), total - 1))
        self._suppress_sync = True
        try:
            if synchronous or not hasattr(self._plot, "request_queue"):
                self._plot.controller.frame_index = video_frame
                self._plot._update_buffer(video_frame, RenderTriggerSource.SET_FRAME)
                self._plot.controller.renderer_request_draw()
            else:
                # Async path: hand the request to the decoder worker process.
                self._plot.frame_ready.clear()
                while not self._plot.request_queue.empty():
                    try:
                        self._plot.request_queue.get_nowait()
                    except queue.Empty:
                        break
                self._plot.request_queue.put((video_frame, None, RenderTriggerSource.UNKNOWN))
                self._plot.controller.frame_index = video_frame
        finally:
            self._suppress_sync = False

    def seek_trial_frame(self, trial_frame: int, synchronous: bool = False) -> None:
        self.seek_video_frame(trial_frame + self._start_frame, synchronous=synchronous)

    def seek_to_time(self, t_trial: float) -> None:
        """Seek from a trial-relative time (used for follower cameras)."""
        if self._plot is None or self._fps <= 0:
            self.set_overlay_time(t_trial)
            return
        video_t = t_trial - self._time_offset + self._start_frame / self._fps
        frame = int(round(video_t * self._fps))
        self.seek_video_frame(frame)

    def set_overlay_time(self, t_trial: float) -> None:
        """Animate the pose overlay on a static-image view from marker time.

        Video views ignore this — their overlay is driven by the decoder's
        frame updates; a static image has no frame clock, so the pose frame
        is derived from the trial time and the pose's own fps.
        """
        if self._static is None or self._overlay is None or self.static_pose_fps <= 0:
            return
        self._overlay.set_frame(int(round(t_trial * self.static_pose_fps)))
        self.request_draw()

    def request_draw(self) -> None:
        if self._plot is not None:
            self._plot.controller.renderer_request_draw()
        elif self._static is not None:
            self._static.request_draw()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_sync_event(self, event) -> None:
        """User interaction on the canvas changed the frame → propagate."""
        if self._suppress_sync:
            return
        kwargs = getattr(event, "kwargs", {})
        t = kwargs.get("current_time")
        if t is None and "cam_state" in kwargs:
            return  # camera pan/zoom, not a time change
        if t is not None:
            self.time_changed.emit(float(t))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        if self._overlay is not None:
            self._overlay.clear()
            self._overlay = None
        if self._plot is not None:
            self.layout().removeWidget(self._plot.canvas)
            self._plot.canvas.setParent(None)
            try:
                self._plot.close()
            except Exception:
                pass
            self._plot = None
        if self._static is not None:
            self.layout().removeWidget(self._static.canvas)
            self._static.canvas.setParent(None)
            self._static.close()
            self._static = None
        self.static_image_path = None
        self.static_pose_fps = 0.0
        self._fps = 0.0
        self._time_offset = 0.0
        self._start_frame = 0
        self._end_frame = 0

    def closeEvent(self, event):
        self.clear()
        super().closeEvent(event)
