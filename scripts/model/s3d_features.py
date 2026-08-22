"""Extract S3D video features: one ``{stem}_s3d.nc`` per video.

    python scripts/model/s3d_features.py VIDEO [VIDEO ...] --out DIR
        [--analysis-fps 25] [--stack-s 0.1] [--mode windows|dense]
        [--truncate-at Mixed_3c] [--device cuda] [--overwrite] [--legacy-npy]

Settings are in seconds; frames are derived from each video's own rate. The
output DataArray carries ``time_s3d`` at the effective rate — interpolate it
onto a trial's time axis when building the dataset. ``--legacy-npy`` also
writes the bare ``(T, 1024)`` array the older notebooks load.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ethograph.video_features import S3D_STAGES, S3DConfig, extract_s3d, plan_s3d
from ethograph.video_features.frames import probe_video


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--analysis-fps", type=float, default=None, help="rate S3D sees (default: every frame)")
    parser.add_argument("--stack-s", type=float, default=0.1, help="window length in seconds")
    parser.add_argument("--mode", choices=("windows", "dense"), default="windows")
    parser.add_argument("--truncate-at", choices=sorted(S3D_STAGES), default=None, help="dense mode only")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--device", default=None, help="torch device; default picks CUDA → MPS → CPU")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--legacy-npy", action="store_true", help="also write {stem}_s3d.npy")
    args = parser.parse_args(argv)

    cfg = S3DConfig(
        analysis_fps=args.analysis_fps,
        stack_s=args.stack_s,
        mode=args.mode,
        truncate_at=args.truncate_at,
        batch=args.batch,
        chunk=args.chunk,
        precision=args.precision,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for video in args.videos:
        target = args.out / f"{video.stem}_s3d.nc"
        if target.exists() and not args.overwrite:
            print(f"{target} exists — skipping (use --overwrite)")
            continue
        info = probe_video(str(video))
        print(f"{video.name}: {plan_s3d(info.fps, cfg).describe()}, mode={cfg.mode}")
        with tqdm(total=info.nframes or None, unit="frames", desc=video.stem) as bar:
            da = extract_s3d(video, cfg, device=args.device, progress=lambda n: bar.update(n - bar.n))
        da.to_netcdf(target)
        if args.legacy_npy:
            np.save(args.out / f"{video.stem}_s3d.npy", da.values)
        print(f"  → {target}  {tuple(da.shape)}")


if __name__ == "__main__":
    main()
