from napari import current_viewer; from napari_animation import Animation; import numpy as np
viewer = current_viewer(); animation = Animation(viewer); layer = viewer.layers[0]; axis = 0
frame_indices = np.linspace(0, layer.data.shape[axis]-1, 1162, dtype=int) # adjust num
[animation.capture_keyframe(steps=1) for idx in frame_indices if viewer.dims.set_current_step(axis, int(idx)) is None]
animation.animate("animation.mp4", canvas_only=True); print(f"Saved All frames")