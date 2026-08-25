# S3D video features

A pretrained video network (S3D, Kinetics-400) turns each frame's
surroundings into a 1024-d vector: what the animal *looks like it is doing*,
which pose keypoints alone do not capture. Unlike a per-frame image encoder,
S3D reads a short **stack** of frames, so its features carry motion.

They are expensive — a forward pass per frame — so they are computed once
into a sidecar file per video and merged into your sessions afterwards.

```{important}
The window is set in **seconds** (`stack_s`) and resolved against each
video's own rate. S3D needs at least **13 frames** per window, so the 0.5 s
default works down to 26 fps but *fails* below it. The error names the
shortest window that works — at 20 fps that is 0.65 s.
```

## Starting from a folder of videos

No config, no session, no alignment — just videos in, sidecars out:

```python
import ethograph as eto

eto.segment.extract_videos(["/data/videos"], "/data/s3d", stack_s=0.5)
```

The first argument takes any mix of files, folders (searched recursively) and
globs. Each video becomes `/data/s3d/{video stem}_s3d.nc`, holding a
`(time_s3d, s3d_dims)` array **on the video's own clock** (frame 0 at t=0),
with the plan recorded in its attrs. The written paths are returned. Videos
that already have a sidecar are skipped unless you pass `overwrite=True`, so
re-running after adding footage is cheap.

| Parameter | Meaning |
|---|---|
| `stack_s` | Window length in seconds. Must be ≥ 13 frames at the effective rate; pass it explicitly. |
| `analysis_fps` | Rate S3D sees; `None` = every frame. Frames are *skipped* to reach it, never interpolated up, so halving it roughly halves the cost. |
| `include` | Regular expressions; keep only videos whose path matches one. |
| `overwrite` | Re-extract videos that already have a sidecar (default `False`). |

Those are the only settings that change the features. Batch size, decode
chunk, `fp16` and the device may be changed in {class}`~ethograph.video_features.S3DConfig`.

### Taking one camera out of a folder

Two cameras pointed at the same arena give nearly identical S3D features, so
extracting both is an hour of GPU time for nothing. `include` narrows what is
found:

```python
eto.segment.extract_videos(["/data/videos"], "/data/s3d", include=["cam-1"])
```

Each pattern is a regular expression matched against the **whole path** with
`re.search`, so a plain substring works, and it finds the camera wherever it
sits in the layout — `cam-1/trial003.mp4` and `trial003_cam-1.mp4` alike.
Several patterns are a union (`["cam-1", "cam-3"]`).

```{note}
A filter that matches nothing raises rather than extracting zero videos —
silently doing nothing looks too much like success. The message says how many
videos were found and shows a few, so you can see what to match against.
```

Working from a config instead? Use `video_features.camera` — the alignment
already knows which stream is which, so there is nothing to pattern-match.

## Starting from a config

If your sessions already have an alignment naming each trial's video, name
each session's `video_dir` and let the alignment resolve the file within it:

```yaml
sessions:
  - source: ../sub-01/ses-01/behav/Trial_data.nc
    labels_path: ../sub-01/ses-01/behav/Trial_data_labels.tsv
    video_dir: /data/videos

video_features:
  stack_s: 0.5
  analysis_fps: 50
  camera: cam-1                   # regex pattern so video features only for camera 1
```

```python
project = eto.segment.Project("project.yaml")

project.video_features()              # extract only
project.video_features(merge=True)    # extract, then merge into the sessions
```

Sidecars go to `{root}/video_features/`; the paths written are returned, and
`overwrite=True` re-extracts. Any setting can also be overridden without
editing the file:
`eto.segment.Project("project.yaml", "video_features.stack_s=0.3")`.

## Merging into a session

Extraction alone does not make S3D a *feature* — the sidecar is on the
video's clock, and the pipeline reads features from the session. Merging
samples the sidecar onto each trial's own time axis (nearest neighbour,
applying that trial's video offset) and writes a session copy carrying
`s3d (time, s3d_dims)`:

```python
project.video_features(merge=True)
# → /data/sub-01/ses-01/behav/Trial_data_s3d.nc
```

The merge **never overwrites your session file**: it writes a sibling
`{stem}_s3d.nc` and logs the path — point your config's `sessions:` at it (or
call {func}`~ethograph.segment.video_features.merge_video_features` yourself
with `in_place=True` if you would rather overwrite). Merging is xarray-only;
for pynapple and NWB sessions the sidecar exists but you carry it in yourself.

Then name it like any other feature:

```yaml
features:
  columns:
    s3d: {s3d_dims: [0, 1, 2, 3]}      # or a shortlist you selected
    speed: {keypoint: [beakTip]}
```

## Choosing which S3D features to keep

1024 columns is a lot next to a handful of kinematic ones, and they dominate
the input — in practice a small, well-chosen subset does *better* than all of
them. Two tools, both leaning on `kind="video_feature"` (see
{doc}`../variable_schema`), which the extractor stamps for you.

**Is the whole group pulling its weight?** That is a question about the
trained model, so it takes two runs — the same materialised dataset, one
extra fit:

```python
eto.segment.Project("project.yaml", "train.run_name=full").train()
eto.segment.Project(
    "project.yaml",
    "train.run_name=no_video",
    "train.drop_kinds=[video_feature]",
).train()

print(eto.segment.Project("project.yaml").compare())
```

```{important}
`compare()` reads each run's `test_metrics.yaml`, which is written only when
the split leaves something in `test`. With `train.split.test_fraction: 0` the
table comes back empty — and an ablation judged on a random trial split is
flattered anyway, so the honest version of this benchmark is
`cross_validate()` with and without the video columns.
```

**Which individual features are stereotypic?** Rank them by Cohen's d: for
each feature and each behaviour class, how far apart the feature's
distribution is *during* that behaviour versus outside it, in pooled standard
deviations. A feature with a large d for some class is one the model can act
on; a feature with a small d everywhere is noise the model has to learn to
ignore.

```python
ranking, names = project.rank_video_features()
# the 20 most discriminating dims
print([names[i] for i in ranking.top(20)] )
```

This reads the **materialised dataset**, so it ranks exactly the columns a
model would see and costs no re-extraction. It picks them out by
`kind="video_feature"`, and raises if no column declares it — describe your
features when you build the session ({func}`~ethograph.io.schema.describe`),
then materialise again. `min_frames` drops a class that barely occurs in a
trial.

The library call underneath takes the trials directly, for a bank that is not
in a project yet:

```python
from ethograph.video_features import rank_features

ranking = rank_features(trials)         # [(values (T, F), labels (T,)), ...]
```

`labels` are dense per-frame class ids with 0 = background; background is
excluded from the comparison by default (`background=0`). Scores are averaged
over trials.

In the GUI the same ranking is a heatmap — **Model ▸ Video features: rank by
Cohen's d…** — of classes against the top-k features, so you can see which
behaviours each feature actually separates, and copy the top-k straight into
your config:

```yaml
features:
  columns:
    s3d: {s3d_dims: [492, 734, 671, 640, 585, ...]}
```

```{important}
The ranking is supervised: it reads your curated labels, so it is only as
good as the trials you have labelled, and it uses `manual`/`curated` rows
only — never another model's `automated` output. Rank on your training
sessions, not on the sessions you intend to report on, or the selection
leaks the test set.
```

## One video, no ethograph at all

```python
from ethograph.video_features import S3DConfig, extract_s3d

da = extract_s3d("clip.mp4", S3DConfig(stack_s=0.5))   # (time_s3d, s3d_dims)
da.to_netcdf("clip_s3d.nc")
```
