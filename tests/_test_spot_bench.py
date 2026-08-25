"""Where does an E2E-Spot training step spend its time? Run with cwd=spot*/."""
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

BATCH, STEPS = 2, 5


def bench_model(clip):
    from train_e2e import E2EModel

    model = E2EModel(3, "rny008_gsm", "gru", clip_len=clip, modality="rgb")
    opt, scaler = model.get_optimizer({"lr": 1e-3})
    net = model._model
    net.train()
    x = torch.rand(BATCH, clip, 3, 224, 224, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(STEPS + 1):
        if i == 1:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.amp.autocast("cuda"):
            out = net(x)
            loss = out.float().mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / STEPS
    print(f"MODEL clip={clip}: {dt:.2f} s/step  {BATCH * clip / dt:.0f} frames/s  "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
    del model, net, opt, x, out, loss
    torch.cuda.empty_cache()


def bench_loader(workers, clip):
    from dataset.frame import ActionSpotDataset
    from util.dataset import load_classes

    classes = load_classes("data/crow_pellet/class.txt")
    ds = ActionSpotDataset(classes, "data/crow_pellet/train.json", sys.argv[1], "rgb", clip, 40,
                           is_eval=False, crop_dim=224, dilate_len=0, mixup=True)
    loader = DataLoader(ds, batch_size=BATCH, num_workers=workers, pin_memory=True, prefetch_factor=1)
    t0 = time.perf_counter()
    it = iter(loader)
    batch = next(it)
    print(f"LOADER {workers}w clip={clip}: first batch {time.perf_counter() - t0:.1f} s (incl. spawn)", flush=True)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        batch = next(it)
    dt = (time.perf_counter() - t0) / STEPS
    t0 = time.perf_counter()
    frame = ds.load_frame_gpu(batch, "cuda")
    torch.cuda.synchronize()
    gpu = time.perf_counter() - t0
    print(f"LOADER {workers}w clip={clip}: {dt:.2f} s/batch  {BATCH * clip / dt:.0f} frames/s; "
          f"gpu transform {gpu:.2f} s; frame {tuple(frame.shape)} {frame.dtype}", flush=True)


if __name__ == "__main__":
    bench_model(200)
    bench_model(100)
    bench_loader(8, 200)
    bench_loader(0, 200)
