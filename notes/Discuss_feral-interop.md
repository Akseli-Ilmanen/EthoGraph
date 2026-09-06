# Discussion with Peter and Jacopo — FERAL × EthoGraph interop

*Prepared 2026-09-06. Nothing FERAL-specific is in the EthoGraph repo yet; that is deliberate. The
outcome of this conversation decides the shape it takes. Background: `Lit_feralbeast.md` (what
FERAL is), `Lit_videofeat.md` (how video features are organised on our side),
`Lit_feral-multi-animal.md` (the longer-range ideas).*

---

## 1. Where EthoGraph is, in one paragraph

EthoGraph is a labelling + curation GUI over synchronised video / audio / ephys / pose time series,
with two scripted model pipelines beside it: `segment` (state events; DLC2Action's architectures on
pose-derived features) and `spot` (point events; E2E-Spot on pixels). Both pipelines read and write
the GUI's own label TSV, so a model's predictions and a human's labels are the same object and the
curation machinery (per-label confidence, `labeling_method`, review queues, grids, frame-by-frame
review) runs over either. Video features are a name-keyed **extractor** slot (`s3d`, `timm`, …)
writing one sidecar per video on the video's own clock, merged onto the trial clock as an ordinary
plottable feature. What we have that FERAL does not: labels, curation, pose, multimodal time
series, a GUI. What FERAL has that we do not: the best supervised video-to-ethogram model.

## 2. The main point: interop and environments

FERAL's `pyproject.toml` pins exact versions (`transformers==5.5.3`, `timm==1.0.26`,
`pandas==2.3.3`, `scikit-learn==1.7.2`, `wandb`, `opencv-python`, `matplotlib`, `hf_transfer`).
EthoGraph is a GUI environment with its own pins (pyqtgraph, xarray, pynapple, movement, timm
1.0.28 today). Installing `feral` into it would downgrade timm and pandas, and drags in wandb and
opencv for a user who only wants to import predictions.

**What we would do today**: run FERAL in its own environment by subprocess, exchanging files — the
same pattern EthoGraph already uses for the vendored E2E-Spot clone. It works, but it is a second
install for the user and a second thing to break.

**What would make an `ethograph[feral]` extra possible** (the ask, in priority order):

1. Lower-bound ranges instead of `==` pins for the libraries that are also GUI dependencies
   (`pandas`, `scikit-learn`, `timm`, `transformers`, `einops`). Keep `==` only where a
   specific version is actually load-bearing.
2. Move `wandb`, `matplotlib`, `hf_transfer` (and possibly `opencv-python`) into extras
   (`feral[train]`, `feral[viz]`). Inference and embedding extraction need none of them.
3. Decode through PyAV rather than decord (Linux/mac only) + cv2 (Windows fallback), so the same
   frame reaches the model on every OS — and so our decode and theirs agree frame-for-frame,
   which matters when we align a per-frame prediction back to a trial clock. Happy to send a PR.

**Rented GPUs are fine, but the round trip should be trivial.** The realistic workflow is: label
and curate in EthoGraph on a laptop, fine-tune FERAL on a rented A100, bring predictions back
into EthoGraph for review. So the things that need to be stable and documented are the two file
formats at the seam:

- **In**: `labels.json` (+ the video folder). We would write an exporter from our label TSV. What
  are the schema guarantees? Frame indices or seconds? Per-video fps? States only, or is there a
  point-event spelling? Is `validate_labels_json` the contract?
- **Out**: the inference JSON. Per-frame probabilities per class per video — is the schema stable
  across versions? Is the checkpoint self-describing (`class_names`, `is_multilabel`, training
  cfg — it looks like it is since v0.2.1)?

If those two are stable, an EthoGraph user never needs FERAL installed locally at all: export,
train elsewhere, import. The extra is then a convenience, not a requirement.

## 3. Things we would like FERAL to write out (small changes, large value on our side)

1. **Per-frame embeddings.** The `clip_projector` output — 64 query tokens = 64 frames, hidden dim
   — is the per-frame time series. `run_inference_folder(..., save_embeddings=True)` writing one
   `(T, D)` array per video (averaged over the overlapping chunks like the probabilities are)
   would let us treat a frozen or fine-tuned FERAL as one more extractor, plotted in the GUI as a
   heatmap next to the pose and fed to our long-range temporal models. Today we would have to
   monkeypatch the module.
2. **The unaveraged per-chunk predictions**, or at least their spread. With 50–80 % overlap every
   frame is predicted 2–5× from different windows; `save_inference_results` averages them and
   discards the disagreement. That spread is an uncertainty signal independent of the softmax,
   and our review queues are sorted by exactly such a number. Cheap for you; probably unlooked-at.
3. **`eval_smoothing_window` off by default for point-like classes**, or a per-class switch. The
   9-frame moving average in `max` shifts onset estimates; for states it is harmless.
4. **`need_weights=True` in `clip_projector`** (later, lower priority). The `(64, 8192)` attention
   map reshaped to `(64 frames, 32 tubelets, 16, 16)` is a spatial "where the model looked"
   overlay per frame — the most convincing diagnostic a GUI can show, and one line to expose.
5. **A per-video crop box in `labels.json`** (later). E2E-Spot's results and ours say resolution
   on the animal matters; a static crop is the cheap version of that, and the single-animal
   precursor of focal encoding (§5).

## 4. Questions about the modelling side

- **Frozen FERAL as a feature extractor — is there any point?** Frozen V-JEPA2 dropped CalMS21
  from 94.5 → 88.4 in your ablation, so the fine-tuning is the value. Our question is different:
  a user *with pose* already has kinematics; is a frozen V-JEPA2 embedding still a useful extra
  input to a long-range temporal model (ASFormer / MS-TCN-style, several seconds of context),
  or does the fine-tuned attention-pooled representation only make sense with FERAL's own head?
- **Long-range context.** FERAL's context is 64 frames (~2 s at 30 fps). Behaviours we care
  about are defined over longer spans (a foraging bout, a sleep state). Have you tried FERAL
  embeddings → a chunk-level temporal model (ASFormer, MS-TCN++)? Is there a reason not to?
  We can run that comparison on our data if the embeddings come out (§3.1).
- **High frame rates.** Our recordings are 200 fps. `chunk_length: 64` is then 0.32 s; the
  V-JEPA2 2-frame tubelet is 10 ms. Should we subsample to ~25–50 fps before FERAL (our video
  feature configs are in seconds and resolve against the rate for this reason), or does the model
  benefit from the dense frames?
- **Point events.** Everything in FERAL is state-oriented (per-frame CE, smoothing, mAP). Our
  point-event work needs ±1–2 frame precision at 200 fps. Is there any appetite for a
  displacement/offset head, or is that simply not FERAL's job (our answer today: it is E2E-Spot's)?
   - see https://github.com/Skovorp/feral/issues/12

- **Multi-animal.** The head is `Linear(d, num_classes)` with no individual axis, so a social
  dataset becomes pair-classes ("A grooms B", "B grooms A"). CalMS21 side-steps this because the
  mice are black and white. Would a focal encoding — highlight one individual, neutralise the
  others, run once per individual — be something you would evaluate? It is one model, N input
  encodings, one test split (details in `Lit_feral-multi-animal.md` §6). We are blocked on the
  tracking side (OCTRON headless core, issue #87) and on the rendering choice, which is empirical.
- **`lite` numbers.** Is there an accuracy figure for the ViT-B preset? Our users' cards are
  8–12 GB; that is what they would run.

## 5. What we would build once this is settled

In order, each independently useful:

1. `labels.json` exporter from the label TSV + video list (mirrors what `spot` does for
   E2E-Spot). Lets anyone fine-tune FERAL on a rented box today.
2. Inference-JSON importer → a prediction set in the GUI (`labeling_method=automated`,
   `prediction_source=feral`), reviewed with the existing curation tools.
3. Late fusion: FERAL's per-frame probabilities averaged with the pose model's — the cheapest
   test of whether video adds anything on a given dataset, no retraining.
4. FERAL as an extractor (§3.1) behind the same registry entry as `s3d` and `timm`, in-process
   if the pins allow, by subprocess otherwise.
5. FERAL as a segmenter driven from a project config (`run_training` in its env), once 1–2 have
   been used for real.

## 6. Not on the table

- Vendoring FERAL. It is MIT and pip-installable; vendoring would only re-create the pin problem
  in our tree.
- Owning any pretraining (BEAST-style or otherwise). We use frozen backbones and supervised
  fine-tuning; SSL objectives never run on user data in EthoGraph.
- A tracker of our own. Identity comes from pose, AprilTags, or an external tool.
