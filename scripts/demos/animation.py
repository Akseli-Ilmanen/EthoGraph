# pip install napari-animation
# Open console in 'window' tab, and paste this in (one line)
from napari import current_viewer; from napari_animation import Animation; import numpy as np
viewer = current_viewer(); animation = Animation(viewer); layer = viewer.layers[0]; axis = 0
frame_indices = np.linspace(208, layer.data.shape[axis]-1, 500, dtype=int) # adjust num
[animation.capture_keyframe(steps=1) for idx in frame_indices if viewer.dims.set_current_step(axis, int(idx)) is None]
animation.animate("animation.mp4", canvas_only=True); print(f"Saved All frames")