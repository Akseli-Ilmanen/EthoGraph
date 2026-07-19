"""Single-file helpers shared by the data-loading paths.

Single-trial data now loads through drag & drop on the cover page
(:mod:`ethograph.gui.cover_page`); the per-modality wizard dialogs that used to
live here were removed in favour of that flow. Only the fps probe remains, since
it is reused by the multi-trial wizard and the ``wizard_single_from_*`` builders.
"""

from typing import Optional


def get_video_fps(video_path: str) -> Optional[int]:
    """Read FPS from a video file, rounded to nearest integer (None on failure)."""
    from ethograph.gui.video_manager import probe_video

    try:
        fps = probe_video(video_path).fps
    except (OSError, ValueError, ZeroDivisionError):
        return None
    return round(fps) if fps else None
