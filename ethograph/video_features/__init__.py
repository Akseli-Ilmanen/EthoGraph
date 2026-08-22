"""Pretrained video features (S3D) for action segmentation.

Configured in seconds (:class:`S3DConfig`), resolved against the video's own
rate (:func:`plan_s3d`), extracted streaming (:func:`extract_s3d`) into a
DataArray on its own ``time_s3d`` axis.
"""

from ethograph.video_features.extract import FEATURE_DIM, TIME_DIM, extract_s3d
from ethograph.video_features.plan import MIN_STACK, S3DConfig, S3DPlan, plan_s3d
from ethograph.video_features.s3d import FULL_STAGE, S3D_STAGES

__all__ = [
    "FEATURE_DIM",
    "FULL_STAGE",
    "MIN_STACK",
    "S3DConfig",
    "S3DPlan",
    "S3D_STAGES",
    "TIME_DIM",
    "extract_s3d",
    "plan_s3d",
]
