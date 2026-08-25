# Pixel-based point-event spotting — state of the work

Companion to `scripts/spot_point_events.py`. Written as a handoff: what was
decided, what is verified, what is assumed, and what to do next.

## The question this answers

Can a model trained on **raw video** predict the stick/pellet point events
(labels 31 and 32) on a session it has never seen, to within a few frames?

The answer decides whether any of this reaches the GUI. If a held-out session
cannot be predicted to a useful tolerance, there is nothing to integrate and
the work stops here.

## Why not the tools already in the repo

**`labels/onset_model.py`** already solves this task — one anchor per class
per trial, confidence read off a curve — but over *kinematic time series*.
The open question is whether the pixels carry the same information.

**`video_features/` (S3D) was considered and rejected.** Not for lack of
temporal context (its receptive field is ~99 frames) but for lack of
*resolution*: the trunk pools time 8x and the window head averages position
pairs, and `plan.py:MIN_STACK` forces >= 13 frames per window. Its features
cannot express which of two adjacent frames is the event.

**The literature is "Precise Event Spotting" (PES)** — single-timestamp
localisation scored at frame tolerance, as distinct from action segmentation
(spans) and action spotting (second-scale tolerance).

| Model | Note |
|---|---|
| **E2E-Spot** (Hong et al., ECCV 2022, BSD-3) | **Chosen.** Simplest, built for small datasets and one GPU, and the baseline the others are measured against. Cloned into `spot/`. |
| T-DEED (CVPRW 2024, GPL-3) | Upgrade path if E2E-Spot lands close but not close enough. Fork of E2E-Spot's codebase, same data format. |
| MFS (2025) | A GSM replacement. Only meaningful once a backbone works. |
| UMEG-Net (2025) | Few-shot, keypoint-distillation. Steal the idea (CoTracker3 already in-repo), not the model. |

E2E-Spot takes context at two levels: Gate Shift Module inside the 2D
backbone (channels shifted across adjacent frames at every depth), then a
**bidirectional** temporal head over the whole clip. The bidirectionality is
not optional here — "the last frame of contact" is defined by what happens
*after* it, so the judgement is inherently non-causal.

## Data, as verified (not assumed)

`C:\Users\aksel\Documents\AI_data\derivatives\sub-02_id-Poppy`

| Session | Trials with 31/32 |
|---|---|
| 20260304_01 | 189 |
| 20260305_02 | 69 |
| 20260306_01 | 180 |
| 20260307_01 | 151 |
| 20260308_01 | 182 |
| 20260309_01 | 145 |

**6 of 13 session folders qualify** — five have a labels TSV with no 31/32
events, two have no TSV. The script filters on this itself
(`discover_sessions`), so no session list is hard-coded anywhere.

916 labelled trials, 1832 point events. Every trial carries exactly one 31
and one 32; 31 always precedes 32.

Per-trial videos: **200 fps**, 4-9 s (900-1700 frames), 512x562,
`stream_offset_for_trial` = 0.0, both events well inside the video.

Also available if more data is wanted: `AK_data` Freddy `20250526_01`
(150 trials) and Ivy `20250306_01` (70 trials).

## The precision reality

At 200 fps one frame is **5 ms**, so a 3-4 frame target is **15-20 ms**. The
PES benchmarks report delta = 1-2 frames at 25 fps, i.e. 40-80 ms. The target
here is therefore 3-5x tighter *in physical time* than published results.

This may still be achievable — a contact break is a sharper visual event than
a sports action — but it is why `score` reports a **sweep** (<=1, 2, 3, 4, 6,
10, 20, 40 frames) rather than one number. "80% within 10 frames, 30% within
3" is both a useful model and a clear no to the 3-frame target, and both
halves matter.

## What is built and verified

`scripts/spot_point_events.py`, five subcommands: `plan`, `export`, `index`,
`train`, `score`.

**`index` exists because `export` writes the split JSONs last.** An
interrupted export leaves frames the JSONs know nothing about (or, worse,
JSONs from an earlier smaller run); `index` rewrites them from the folders
on disk without decoding anything, skipping any folder a killed worker left
short (it says which, with both counts). Current state: the 2026-08-24
export was `--max-trials 60` and stopped in the fourth session — **216
complete trials** over `20260304_01`, `20260305_02`, `20260306_01` (60 each)
and `20260307_01` (36; trial 39 half-written), so the held-out session is
`20260307_01`, not `20260309_01`: 153 train / 27 val / 36 test.

Verified end to end on real files: `plan` and `export` run, the emitted JSON
matches E2E-Spot's schema exactly (checked against `spot/data/fs_comp/`),
921 frames written for one trial at 204x224, ~7.3 KB per JPEG.

**The one invariant worth protecting.** `VideoSync`'s convention is
`trial = video + offset`, so `video_t = onset_s - offset`. It is written once,
in `event_frames()`. Reversed, every label shifts by the offset and nothing
in the result reveals it.

**Score the `.recall.json.gz`, not the `.json`.** E2E-Spot's plain
prediction file is the per-frame argmax — a class appears only where its
score beats background — so an under-trained class reads as "never
predicted" although its curve has a clear peak. The recall file carries
every frame above a low threshold, and `best_per_class` on it is the md's
own `tallest_peak` rule. `score` reads either.

**First result (2026-08-25, one epoch, `spot/runs/overnight/pred-test.0.recall.json.gz`):**
on the 36 held-out trials of `20260307_01`, by tallest peak —

| class | n | miss | median err | ≤3 | ≤10 | ≤20 | ≤40 |
|---|---|---|---|---|---|---|---|
| label_31 | 36 | 0 | 8 frames (40 ms) | 31 % | 56 % | 78 % | 83 % |
| label_32 | 34 | 2 | 7 frames (35 ms) | 29 % | 62 % | 76 % | 94 % |

After **one** epoch, both events land within 20 frames (100 ms) in three
quarters of unseen trials and within 3 frames in ~30 %. Read by argmax the
same file said 32 was never predicted. The mean (39 / 16 frames) is far
above the median: a few peaks sit at frame 0, the clip's padding edge —
worth a look before trusting the tail. A 15-epoch run (`runs/overnight2`,
val mAP from epoch 3) is what answers whether the ≤3 column climbs.

`test_e2e.py <run_dir> <frame_dir> -s test --save` scores any checkpoint
without waiting for the run to end (it writes `pred-test.{epoch}.json`
beside the checkpoint); an unpatched upstream crashes there on `np.int`
(`dataset/frame.py:get_labels`, numpy ≥ 1.24) — part of the patch now.

## Environment

- env `ethograph`; `conda run` is broken on this machine — call
  `~/anaconda3/envs/ethograph/python.exe` directly, with `PYTHONUTF8=1`
  and `PYTHONIOENCODING=utf-8`.
- torch 2.9.0+cu129, CUDA available (RTX 3080), av 18.0.0, PIL 12.3.0.
- The env carried a **genuine 2018 torchvision 0.2.0** (no `ops/`, no `io/`),
  which `timm` cannot import. The PyTorch cu129 index has no Windows wheel
  for torch 2.9's pair, so the fix was the PyPI wheel, torch left untouched:
  `pip install --no-deps torchvision==0.24.0` (reports `0.24.0+cpu`; only its
  Python-level `resnet` blocks and `FrozenBatchNorm2d` are used, never a GPU
  custom op). Then `pip install --no-deps timm huggingface_hub` — the second
  is where timm fetches the pretrained RegNet weights from. Every install is
  `--no-deps`: a plain install would resolve torch and replace the CUDA build
  with a CPU wheel, the failure `pyproject.toml` already warns about for the
  `model` extra.

### Patches to the `spot/` clone (all needed on this machine)

| file | change | why |
|---|---|---|
| `model/shift.py` | `timm.models.layers.conv_bn_act.ConvBnAct` → `timm.layers.ConvNormAct` | the old path is gone in timm 1.0; a GSM backbone raised `AttributeError` |
| `util/dataset.py` | `'crow_pellet'` added to `DATASETS` | it is only an argparse `choices` list |
| `train_e2e.py` | `store_config('/dev/stdout', …)` → `store_config(None, …)`, which prints | no `/dev/stdout` on Windows |
| `dataset/frame.py` | the three `torch.jit.script(nn.Sequential(…))` → plain `nn.Sequential` | Windows DataLoader workers are spawned and pickle the dataset; a scripted module cannot be pickled. Scripting was only a speed-up |
| `train_e2e.py` | `worker_init_fn` closure → module-level `_WorkerSeed` callable, `.epoch` set per epoch | same spawn pickling: a closure over `main()`'s locals cannot be sent to a worker |
| `dataset/frame.py` | `np.int` → `int` in `get_labels` | removed in numpy 1.24; the val-mAP pass crashed on it |
| `train_e2e.py`, `test_e2e.py` | `--stride` exposed, passed to every dataset, recorded in `config.json` | upstream's loaders had it, its CLI did not |

The whole patch is `scripts/spot_windows_compat.patch` (`git -C spot diff`
regenerates it); `spot/` and `spot_mod/` are gitignored, so the clone is
never committed here — vendoring is ADR 0001's decision, made later.

**The wrapping check is done**: with the `shift.py` patch, `rny008_gsm`
wraps **14 of 14** residual blocks (`GatedShift` count over `m.modules()`),
so a poor result cannot be blamed on a silently 2D backbone.

## Costs

Export is parallel across trials (`--workers`, default cores-2 capped at 12),
measured at **~1.7 s per trial** on 24 cores. Profiled per trial, warm:

| stage | cost | note |
|---|---|---|
| H.264 decode | 0.11 s | 9656 fps — free |
| YUV to RGB (`to_ndarray`) | 2.6 s | **the bottleneck** |
| PIL resize | 1.6 s | |
| JPEG encode + write | 0.6 s | disk is not the problem |

- `--max-trials 60` over 6 sessions (360 trials): **~10 min**, ~2.4 GB
- everything (916 trials): **~26 min**, ~9 GB

One pool serves every session. Creating it per session made a short export
*slower* than serial, because each worker re-imports ethograph (~3.4 s) on
spawn.

Training cost is independent of all this: `EPOCH_NUM_FRAMES = 500000` in
`spot/train_e2e.py` fixes an epoch at 500k frames however much was exported.
Measured (see *The one rule* under *Next steps*): ~780 frames/s once the
loader batch fits the card, i.e. **~11 min of compute per epoch** plus the
val passes; 15 epochs is an evening, not a week. The first run, at 800
frames per step, was 4-5 h per epoch — the same model, paging.

**The frame dump is E2E-Spot's assumption, not the task's.** It exists because
their dataloader samples random clips from long untrimmed sports video, where
seeking into H.264 means decoding from the previous keyframe; JPEGs give O(1)
random access. Note where the export's cost actually is: the H.264 decode is
0.11 s per trial; the other ~4 s is full-resolution colour conversion, PIL
resize and JPEG writing — work training never needs. Converting only the
frames a clip keeps (`frame.reformat(w, h, "rgb24")`, one swscale pass) is a
fraction of that. How this should look in the GUI — decided in discussion,
not built:

| | short per-trial videos (this data, 4-9 s) | one long session video (`ExternalFileIndex`) |
|---|---|---|
| **inference** | decode the trial whole, slice — no dump | seek to keyframe, decode forward; cheap at this resolution (a 250-frame GOP is ~13 ms), costly for 1080p+ long-GOP files — there `io/video_proxy.py`'s short-GOP proxy is the random-access cache, keyed by `media_cache_key` like the audio WAV |
| **training** | decode on the fly is viable (8 workers outrun the GPU), but a frame dump is the simpler, safer default | dump — thousands of random clips against keyframe seeks is exactly E2E-Spot's case |

So: **inference never dumps frames**; a dump is a training-time cache, and
for the GUI it would be clips around each event, not whole trials (see the
open decision below). The invariant either way is frame *identity*: whatever
serves frames returns source frame *i* for index *i*, and a proxy must be
CFR with the same frame count — the test to write is proxy frame i ==
source frame i on a real file.

## Next steps

```bash
# the environment and spot/ patches above are in place; `export` resumes
python scripts/spot_point_events.py export --max-trials 60      # finish 20260307_01 + the last two sessions
python scripts/spot_point_events.py index --test-session 20260309_01
python scripts/spot_point_events.py train --epochs 20
python scripts/spot_point_events.py score spot/runs/crow_pellet/pred-test.recall.json.gz
```

`train` is a foreground subprocess; run long jobs from a terminal, not from
a tool with a wall-clock cap.

### The one rule that decides the speed: ≤ 200 frames per loader batch

The first overnight run (2026-08-24, `spot_mod/runs/overnight`) took **~4-5
h per epoch** at 20-40 frames/s — 30-50x slower than the model should be.
Isolating the pieces (`tests/_test_spot_bench.py`, model alone on random
tensors vs. the loader alone) found it, and it is not E2E-Spot's cost:

| frames per loader batch | peak VRAM | frames/s (fwd+bwd, AMP) |
|---|---|---|
| 200 (clip 200 x batch 1, or clip 100 x 2) | 5.8 GB | **~780** |
| 300 (clip 150 x 2) | 8.7 GB | 409 |
| 400 (clip 200 x 2, or clip 100 x 4) | **11.4 GB** | 11-28 |

The RTX 3080 has 10 GB. Above it, the Windows driver pages VRAM into
system RAM: the GPU reads 100 % busy and runs 50x slow, and `nvidia-smi`
shows ~9.9 GB whatever the batch, which is why halving the batch looked
like it changed nothing. Peak memory scales with *frames per loader batch*
(= `batch_size / acc_grad * clip_len`), not with either alone. Upstream's
default is clip 100 x batch 8 on 24-80 GB cards; we doubled the clip for
200 fps and kept batch 4, i.e. 800 frames per step. The loader is not a
factor (8 workers: ~9,700 frames/s), nor is the deferred GPU transform
(0.05 s per 400 frames; re-scripting it is not worth a patch).

So `train` defaults to `--acc-grad 4`: batch 4 / 4 x clip 200 = 200 frames,
the full second of context, ~780 frames/s → **~11 min of compute per
500k-frame epoch**, plus the val passes. Our whole diff against upstream is
`spot_mod/windows_compat.patch` (26 lines, all Windows compatibility) —
`git -C spot diff` shows the same; `spot_mod/` is a snapshot from before
the fix and can be deleted once its `runs/overnight` (one epoch: train loss
0.055, val loss 0.039) is no longer wanted.

### Running overnight

Every failure so far was setup (missing package, Windows spawn pickling,
`/dev/stdout`), not instability. The wrapper covers what is left:
`train_e2e.py` checkpoints every epoch and `--resume` continues from the
newest, so `--retries N` restarts a crashed run up to N times from where it
stopped (`RETRY_PAUSE_S` between); `_run_logged` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` against fragmentation;
every attempt's output is appended to `spot/{save_dir}/train.log`,
unbuffered, so the tqdm bar is readable there.

```powershell
python scripts/spot_point_events.py train --epochs 15 --retries 5 --save-dir runs/overnight2 -- --start_val_epoch 3
```

**`--stride k` is the speed knob for high frame rates.** It reads every
k-th frame, so a clip covers k times the time for the same compute (or the
same time for 1/k of it), at the price of k-frame resolution: at 200 fps,
`--stride 4 --clip-len 100` is 2 s of context per clip, 20 ms resolution.
It is upstream's own loader option (`ActionSpotDataset` / `ActionSpotVideoDataset`
`stride=`), now exposed through `train_e2e.py` and recorded in the run's
`config.json` so `test_e2e.py` infers at the same stride. A strided run's
predictions are on the downsampled clock and say so through their `fps`;
`score` maps them back, so the sweep is always in full-rate frames. Give up
the 1-2 frame column knowingly: if the full-rate sweep says the hits are
mostly within 10 frames anyway, a strided run is the same answer for a
quarter of the time; if they are within 3, stride costs exactly that.

How many epochs: an epoch is a fixed 500k frames and the training set is
~170k unique frames, so one epoch is ~3 passes; upstream's 50 epochs on
datasets several times larger is ~10-20 passes. 5 is the minimum for a
real answer, 10-15 is comfortable, beyond 20 buys nothing. The cosine
schedule is tied to `--num_epochs`, so a short run is a complete schedule.
`--start_val_epoch 3` makes the val-mAP best-epoch selection (the only
guard against overfitting 153 trials) start early; upstream's default is
`num_epochs - 20`. Anything after `--` reaches `train_e2e.py` verbatim.

On the machine, not in the script: keep it from sleeping
(`powercfg /x standby-timeout-ac 0`, `powercfg /x monitor-timeout-ac 0` is
harmless) and pause Windows Update so it cannot reboot. Leave the terminal
open; the run is its child.

## If it works: the integration shape (discussed, not built)

**Vendoring, not importing.** `spot` has no `setup.py` and is not on PyPI, so
depending on it is not possible; a submodule would pin `torch==1.11.0` and
drag in `web/`, `external/` and eight eval scripts. Vendor the model files
only — `model/shift.py`, `model/impl/gsm.py`, `model/impl/tsm.py` and what
they need — following ADR 0001 exactly. Licences: BSD-3 (spot) with BSD-2
(GSM, FBK 2019) nested; `NOTICE.md` must name both. Strictly lighter than the
AGPL already carried for DLC2Action. Not vendored: `util/` (welded to their
format), `train_e2e.py` and the eval scripts (ADR 0004 — no CLI).

**Reuse `onset_model`'s workflow, because the output contract is identical.**
E2E-Spot's head emits a per-frame probability per class — the same object
`target_curves` produces. So everything downstream transfers untouched: the
`~/.ethograph/models/{name}` store, the frozen config, `targets` as
`{label_id: name}`, `tallest_peak` (confidence *is* peak height),
`OnsetPrediction`/`TrialPrediction`, `onset_curves.npz` and the review
overlay, `automated` -> curation -> `curated`, the trials-table filter as the
one scope, and `predict_onsets` as a workflow `STEP_KIND`.

Does not transfer: the `features` config (becomes camera/crop/clip length),
`train_data/*.npz` caching, `lag_offsets`/`build_windows`, the joblib bundle,
CPU-synchronous training, and `_CV_FOLDS` cross-fitting (three deep-net fits;
becomes a single held-out split, keeping the `hit_rate` semantics).

**Open decision:** a spot model's training store is either cached crops
(~1 GB for the events at +/-2 s, keeps the model self-contained exactly as
`onset_model` is) or an index of video paths (cheap, but breaks when videos
move — a regression against today's behaviour). The table under *Costs*
leans to the first: a training dump is fine, an inference dump is not, and a
proxy is a cache, never a store. Decide before writing the store.

**Sequencing caution.** Do *not* extract a shared spine out of
`onset_model.py` first. The parts most likely to not fit — the clip store and
the GPU worker — are the parts not yet built, so the seam is not yet known.
Build against the existing shapes, see what rubs, then factor.

**pyproject:** `ethograph[spot]` adds `timm` and essentially nothing else;
torch stays out-of-band, following the policy already documented for the
`model` extra and CoTracker.
