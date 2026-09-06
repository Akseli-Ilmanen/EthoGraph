"""Frame-wise video features from any ``timm`` image model.

The frame-wise tier of the extractor registry: every frame is embedded on
its own by a pretrained image backbone — DINOv2 by default — with no
temporal window at all. Motion is not in the feature; a temporal model
downstream reads it off the sequence.

Nothing here is per model. ``timm.create_model(name, pretrained=True,
num_classes=0)`` returns the backbone's own pooled embedding (the CLS token
for a ViT, global average pooling for a convnet), and the model's
``pretrained_cfg`` says how it wants its input — size, interpolation,
normalisation — so a different backbone (DINOv3, ConvNeXt, EVA, …) is a
string swap in ``model_name``. Weights are downloaded by timm into its own
cache on first use.

``timm`` is an optional dependency (``pip install 'ethograph[timm]'``); this
module is imported only when the ``timm`` extractor is built.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import timm
import torch
import torch.nn.functional as F
import xarray as xr
from timm.data import resolve_data_config
from torch import nn

from ethograph.utils.device import resolve_device
from ethograph.video_features.base import CropBox, feature_dim, to_dataarray
from ethograph.video_features.frames import iter_frame_chunks, probe_video
from ethograph.video_features.plan import FramePlan, plan_frames

NAME = "timm"
FEATURE_DIM = feature_dim(NAME)

#: The default backbone: DINOv2 ViT-B/14 with register tokens, self-supervised
#: on LVD-142M (Apache 2.0). Base is the size the DINOv2 downstream literature
#: reports; the registers keep artifact tokens out of the pooled embedding.
DEFAULT_MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"

TimmPrecision = Literal["fp16", "fp32"]

_INTERPOLATION: dict[str, str] = {"bicubic": "bicubic", "bilinear": "bilinear"}


@dataclass(frozen=True)
class TimmConfig:
    """What the user decides.

    model_name
        Any ``timm`` model with pretrained weights.
    analysis_fps
        The rate the backbone sees; ``None`` = every frame. Frames are
        skipped, never interpolated up. The one cost lever: a ViT-B at its
        native 518 px is tens of GFLOPs per frame.
    batch / chunk
        Frames per forward pass; frames decoded per step.
    precision
        ``"fp16"`` autocasts on CUDA; ``"fp32"`` reproduces bit-exactly.
    """

    model_name: str = DEFAULT_MODEL
    analysis_fps: float | None = None
    batch: int = 32
    chunk: int = 128
    precision: TimmPrecision = "fp16"


@dataclass(frozen=True)
class DataConfig:
    """How the backbone wants its input, read off ``pretrained_cfg``."""

    side: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    crop_pct: float
    interpolation: str

    @classmethod
    def of(cls, model: nn.Module) -> DataConfig:
        cfg: dict[str, Any] = resolve_data_config({}, model=model)
        _, h, w = cfg["input_size"]
        if h != w:
            raise ValueError(f"timm model wants a {w}x{h} input; only square inputs are supported here")
        interpolation = _INTERPOLATION.get(str(cfg["interpolation"]))
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation {cfg['interpolation']!r} in the model's data config")
        return cls(
            side=int(h),
            mean=tuple(float(m) for m in cfg["mean"]),  # type: ignore[arg-type]
            std=tuple(float(s) for s in cfg["std"]),  # type: ignore[arg-type]
            crop_pct=float(cfg["crop_pct"]),
            interpolation=interpolation,
        )


def load_timm(model_name: str, device: torch.device) -> nn.Module:
    """The pretrained backbone with its classifier removed, in eval mode."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()
    return model.to(device)


def preprocess(frames: np.ndarray, data: DataConfig, device: torch.device) -> torch.Tensor:
    """``(N, H, W, 3)`` uint8 → ``(N, 3, side, side)`` normalised float32.

    timm's eval transform: the shorter side is resized to ``side / crop_pct``,
    the centre ``side`` square is cropped, then mean/std normalisation.
    """
    x = torch.from_numpy(np.ascontiguousarray(frames)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    h, w = x.shape[-2:]
    target = int(round(data.side / data.crop_pct))
    scale = target / min(h, w)
    size = (max(target, int(round(h * scale))), max(target, int(round(w * scale))))
    x = F.interpolate(x, size=size, mode=data.interpolation, align_corners=False)
    i = int(round((size[0] - data.side) / 2.0))
    j = int(round((size[1] - data.side) / 2.0))
    x = x[..., i : i + data.side, j : j + data.side]
    mean = torch.tensor(data.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(data.std, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def frame_features(
    embed: Callable[[torch.Tensor], torch.Tensor], chunks: Iterator[torch.Tensor], batch: int
) -> Iterator[torch.Tensor]:
    """Stream ``(N, 3, H, W)`` chunks in, yield ``(N, D)`` features out, one row per frame."""
    for chunk in chunks:
        for i in range(0, chunk.shape[0], max(1, batch)):
            yield embed(chunk[i : i + batch])


def _embedder(model: nn.Module, device: torch.device, precision: str) -> Callable[[torch.Tensor], torch.Tensor]:
    use_half = precision == "fp16" and device.type == "cuda"

    def embed(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_half):
            return model(x).float()

    return embed


def extract_timm(
    video_path: str | Path,
    cfg: TimmConfig = TimmConfig(),
    *,
    crop: CropBox | None = None,
    device: str | None = None,
    progress: Callable[[int], None] | None = None,
) -> xr.DataArray:
    """Per-frame embeddings of *video_path* → ``(time_video, timm_dims)``."""
    info = probe_video(str(video_path))
    plan = plan_frames(info.fps, cfg.analysis_fps)
    if crop is not None:
        crop.validate(info.width, info.height, Path(video_path).name)
    dev = torch.device(resolve_device(device))
    model = load_timm(cfg.model_name, dev)
    data = DataConfig.of(model)

    consumed = 0

    def chunks() -> Iterator[torch.Tensor]:
        nonlocal consumed
        for raw in iter_frame_chunks(video_path, step=plan.step, chunk=cfg.chunk):
            consumed += raw.shape[0]
            yield preprocess(raw if crop is None else crop.apply(raw), data, dev)
            if progress is not None:
                progress(consumed)

    parts = list(frame_features(_embedder(model, dev, cfg.precision), chunks(), cfg.batch))
    width = int(model.num_features)
    feats = torch.cat(parts).cpu().numpy() if parts else np.zeros((0, width), dtype=np.float32)
    if feats.shape[0] != consumed:
        raise RuntimeError(f"Decoded {consumed} frames but produced {feats.shape[0]} features")

    attrs: dict[str, object] = {
        "video_path": info.path,
        "model_name": cfg.model_name,
        "input_side": data.side,
        "precision": cfg.precision,
    }
    if crop is not None:
        attrs["crop"] = [crop.x0, crop.y0, crop.x1, crop.y1]
    return to_dataarray(feats, name=NAME, video_fps=plan.video_fps, step=plan.step, attrs=attrs)


@dataclass(frozen=True)
class TimmExtractor:
    """The ``timm`` entry of the registry: :class:`TimmConfig` plus an optional crop."""

    config: TimmConfig = TimmConfig()
    crop: CropBox | None = None

    @property
    def name(self) -> str:
        return NAME

    def plan(self, video_fps: float) -> FramePlan:
        return plan_frames(video_fps, self.config.analysis_fps)

    def extract(
        self,
        video: str | Path,
        *,
        device: str | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> xr.DataArray:
        return extract_timm(video, self.config, crop=self.crop, device=device, progress=progress)
