# EthoGraph — video-model review and decisions

*Date: 2026-08-25. Covers E2E-Spot and successors, FERAL, BEAST, and how they fit the EthoGraph library design.*

---

## 1. Decisions

1. **FERAL: promising, wait.** Best-in-class supervised video-to-ethogram tool, but default config needs ~24 GB VRAM (V-JEPA2 ViT-L) and its `lite` preset (ViT-B) has no published accuracy. Ship a **JSON importer** for its predictions now; revisit as a backend when `lite` numbers exist or the VRAM floor drops.
2. **One default per slot.** Every slot (extractor, head, codec) has exactly one default; alternatives are opt-in by name. Defaults:
   - State events: `kinematics → ASFormer → StateCodec`
   - Point events: `regnety-200mf-gsm → GRU (+ displacement head) → PointCodec (Soft-NMS)`
3. **GUI as a feature-engineering tool.** Video-model features are first-class `(time, feature)` streams in the TrialTree, next to kinematics. Cohen's-d subset selection, before/after-fine-tuning comparison and label-coloured embedding views all operate on them. First frozen extractor: ImageNet ViT-MAE CLS (768-d, CPU-tolerant).
4. **No SSL pretraining, no semi-supervised heads.** Frozen SSL backbones and supervised fine-tuning are in; running MAE/contrastive objectives on user data is out. `FeatureExtractor.fit` exists in the protocol but every shipped extractor implements it as a no-op.
5. **Vendor nothing new.** E2E-Spot stays vendored; everything else is a pip extra plus an adapter, or an importer.

---

## 2. Architecture: three protocols, one seam

The TAS literature's shared contract — per-frame features `(T, D)` plus per-frame labels `(T,)` — is the seam. Everything plugs into one of three protocols.

```python
class FeatureExtractor(Protocol):
    def fit(self, videos: Sequence[Path]) -> Self: ...        # no-op for all shipped extractors
    def transform(self, video: Path) -> xr.DataArray: ...     # dims (time, feature)

class TemporalHead(Protocol):
    def fit(self, features: xr.DataArray, targets: xr.DataArray) -> Self: ...
    def predict_proba(self, features: xr.DataArray) -> xr.DataArray: ...   # (time, class)

class TargetCodec(Protocol):
    def encode(self, events: EventTable, n_frames: int) -> xr.DataArray: ...
    def decode(self, proba: xr.DataArray) -> EventTable: ...
```

- **State vs point events live only in `TargetCodec`.** `StateCodec`: argmax + min-duration smoothing. `PointCodec`: background class + displacement target, Soft-NMS on decode. Heads are agnostic.
- **End-to-end models** (E2E-Spot, fine-tuned FERAL) are one object implementing both `FeatureExtractor` and `TemporalHead`; `transform` returns their penultimate features.
- **Config surface:** three names plus a checkpoint path. Model-native config passes through as an opaque `extra: dict`, never mirrored.

```python
@dataclass
class ExtractorConfig:
    name: str                    # "kinematics" | "s3d-kinetics" | "vit-mae-b" | "vjepa2-vitb" | "regnety-200mf-gsm"
    trainable_layers: int = 0    # 0 = frozen + cached features; >0 = fine-tune last N (fused training, [gpu])
```

`trainable_layers == 0` is today's workflow unchanged (extract once, cache, train head in seconds). `> 0` switches to the fused loop the vendored E2E-Spot already implements. Everything downstream — codec, evaluation, GUI review — is identical in both modes.

---

## 3. Inclusion criteria (apply from the paper/repo, no benchmarking)

| # | Rule | Removes |
|---|------|---------|
| 1 | Fits extractor / head / codec without a fourth concept | BEAST pretraining, semi-supervised heads |
| 2 | pip-installable with a Python entry point | anything needing vendoring |
| 3 | Runs on CPU (`base`) or an 8 GB consumer GPU (`[gpu]`) | 24 GB / Ampere-only backends |
| 4 | Serves a user need you can name today | "SOTA by 2 mAP" |
| 5 | Gain over current default exceeds its noise, per the paper's own ablations | T-DEED, AdaSpot, F3ED |
| 6 | One default per slot; alternatives selectable by name only | duplicate backbone names |

### Verdicts

| Option | Verdict | Rule |
|---|---|---|
| E2E-Spot 200MF + displacement head + Soft-NMS | **in**, default point-event backend | 4, 5 |
| MSAGSM as GSM replacement | in **as a flag on the same extractor** (`shift_module: gsm\|msagsm`), not a new name | 5, 6 |
| T-DEED, AdaSpot, F3ED | out; borrow eval code (tolerance-F1, edit score) | 2, 5 |
| Frozen ViT-MAE / V-JEPA2-B / DINO as `FeatureExtractor` | **in**; ViT-MAE CLS first, V-JEPA2 as `[gpu]` alternative | 1, 3 |
| `trainable_layers` fine-tuning knob | in, opt-in, `[gpu]` | 1, 3 |
| FERAL | importer only for now | 3, 4 |
| BEAST pretraining | out; accept a user-supplied checkpoint path | 1, 3 |
| lightning-action TCN head | optional second head, not default | 6 |
| Semi-supervised anything | out | 1 |

---

## 4. Model notes (what each is and is not)

### E2E-Spot (Hong et al., ECCV 2022)
- RegNet-Y 200MF/800MF with GSM (channel shift between neighbouring frames) → 1-layer bi-GRU → per-frame K+1 softmax. 2.8 M + 1.7 M params, 0.3 ms/frame on A5000, ~23 GFLOPs per 100-frame clip at 224².
- Built for ±1-frame precision. Long-range context via GRU over 100–500-frame clips; FS/FineGym lose 7–15 mAP going from 100 to 8 frames — clip length must cover event dependencies.
- Weaknesses: data-hungry (Tennis 33k events; FineDiving 12k events → 68 mAP), ImageNet init only, resolution matters (112 px: −7.6 mAP) → **crop around the animal**, one event per frame, states are second-class (onset/offset as two classes, end frames noisier).
- Low-data behaviour: most robust of the RGB models in UMEG-Net's 100-clip benchmark; T-DEED collapses there.
- Cheap upgrades from successors: **displacement head** (regress offset to nearest event within r_E, MSE; replaces label dilation, +3–4 mAP), **Soft-NMS** 3-frame window (+3–4 mAP), **GSF in second half of backbone only**, **MSAGSM** (14/15 wins, minimal overhead). Diagnostic: cosine similarity of adjacent-frame tokens to sequence mean (T-DEED Fig. 5) tells you whether the head smears boundaries.
- Successors and speed: T-DEED 200MF ≈ same FLOPs but 3.6× params; AdaSpot +30%; 800MF variants ≈ 4×. Real cost for users is video decoding, not the model — decode straight to tensors, no `spot_frames/` on disk; don't drop below 224 px; single pass at inference.

### FERAL (Skovorodnikov & Razzauti et al., bioRxiv 2025/2026, `pip install feral`, MIT)
- **A segmenter, not a feature extractor**: V-JEPA2 (ViT-L, Diving48 ckpt, last 12/24 layers fine-tuned) → 64 learned query tokens cross-attend over patch tokens → one embedding per frame → BN → linear → per-frame CE with label smoothing, √inv-freq class weights, mixup. 64-frame chunks at 256², 50% overlap, averaged.
- Frozen backbone drops 94.5 → 88.4 mAP on CalMS21: the fine-tuning *is* the value. Features (`FeralModel.clip_projector` output, `(64, 768)` per chunk) are meaningful only after fine-tuning.
- No long-range temporal model (context = 64 frames ≈ 2 s). No segment-level metrics reported (no edit score, no F1@k). V-JEPA tubelets are 2 frames → systematic ±1-frame ambiguity for point events (mitigations: odd `chunk_shift`, frame doubling).
- Public API: `feral.run_training(cfg)`, `feral.run_inference_folder(ckpt, folder)`, `feral.apply_mode(cfg, "lite"|"max"|"rare")`, `FeralModel`, `BACKBONES`. `rare` preset = no mixup, no label smoothing, grad-clip, class-weight cap — matches sparse-event data.
- Hardware: Ampere+ (bf16, flash-attn), 24 GB recommended; `lite` (ViT-B) + `gradient_checkpointing` + `train_bs 2` should fit a 10 GB RTX 3080. CUDA only. Windows via `triton-windows`; prefer WSL; set `training.compile: false` on first run.
- Potential two-stage use: fine-tune FERAL → dump per-frame embeddings → ASFormer for long-range coherence. Unpublished comparison; directly tests whether your head adds anything over a chunk-level transformer.

### BEAST (Wang, Yu et al., ICLR 2026, `paninski-lab/beast`, MIT)
- **A backbone-pretraining recipe**, not a segmenter. ViT-B/16 on single frames, ImageNet-MAE init, then MAE (75% mask) + temporal contrastive loss (positive = t±1, InfoNCE on CLS through a BN projector). Frame selection: motion-energy filter + k-means, ~600 anchor triplets/video, ~100k frames total. 800 epochs, 8 A40s, 12–25 h.
- Per-frame features at full frame rate (no tubelets); CLS = 768-d storable; patch tokens = 196×768, streamed. Inference 17.6 GFLOPs / 5.4 ms per frame on A100 (~75× E2E-Spot 200MF per frame).
- Downstream lives elsewhere: `lightning-pose` (pose), `lightning-action` (segmentation: linear or 2-block dilated TCN, features + Δ-features, sliding window), IBL RRR / Facemap TCN (neural encoding).
- Key findings: **frozen ImageNet ViT-MAE already beats keypoints, PCA, CEBRA, DINOv2, CLIP** for neural encoding with no pretraining. Domain pretraining adds a moderate increment (IBL seg 0.84→0.87, CalMS21 0.74→0.81); contrastive-only hurts. CLS wins for neural encoding, attention-pooled patches win for segmentation (0.81 vs 0.63 on CalMS21). Δ-features help across every feature type. Pose: ViT-MAE backbone beats ResNet-50/DLC/Lightning-Pose with 100 labels; domain pretraining helps further.
- Not a foundation model — checkpoints are dataset-specific and won't transfer to a crow.

### Ethology context
- No ethology/neuroscience paper uses E2E-Spot or successors; that lineage is entirely sports (SoccerNet). The animal side jumped straight to video foundation models (FERAL, TRACE, BEAST, Autobehaver, PlayClass), all state-oriented, none evaluated at frame tolerance. The frame-precise point-event gap for animals is open.
- Transferable non-sport insights from the E2E-Spot citation graph: few-shot (UMEG-Net), class-imbalance losses (Santra et al. SoftIC), label dilation vs displacement (T-DEED), dense events under handheld/occluded cameras (TTA dataset), sequence metrics (F3Set edit score), single-frame-label ambiguity and loss/metric mismatch (BME runner-up report).
- SSL in practice: (1) frozen SSL backbone — indistinguishable from S3D-Kinetics from the user's side, no SSL code runs; (2) fine-tune with your labels — still supervised, only the init differs; (3) run the SSL objective on your unlabeled data — the only level where contrastive/masked losses enter your code. EthoGraph does 1 and 2, not 3.

---

## 5. Next steps

### Week 1 — E2E-Spot upgrades (point events)
- [ ] Add displacement head (linear → `(T, 1)` offset, MSE within r_E = 1–2) alongside the K+1 classifier; drop label dilation.
- [ ] Replace hard NMS with Soft-NMS, 3-frame window.
- [ ] Add `shift_module: gsm | msagsm` flag on the `regnety-200mf-gsm` extractor; only keep MSAGSM if it is a drop-in `nn.Module`.
- [ ] Replace `spot_frames/` extraction with on-the-fly decoding (PyAV/decord/torchcodec) from the ffmpeg proxy; crop to a user-drawn box.
- [ ] Add the adjacent-token cosine-similarity diagnostic to the training log.

### Week 2 — protocols and codecs
- [ ] Introduce `FeatureExtractor`, `TemporalHead`, `TargetCodec` protocols; move point/state logic into `PointCodec` / `StateCodec`.
- [ ] Wrap current kinematics pipeline and ASFormer as the state default; wrap vendored E2E-Spot as a fused extractor+head for the point default.
- [ ] `ExtractorConfig(name, trainable_layers=0)`; registry keyed by name; `check_requirements()` per component so the GUI greys out unavailable backends.
- [ ] Export to standard TAS layout (`features/*.npy`, `groundTruth/*.txt`, `mapping.txt`) so any TAS repo is a thin adapter.

### Week 3 — frozen video features + feature engineering GUI
- [ ] `vit-mae-b` extractor: HuggingFace `facebook/vit-mae-base`, CLS per frame, weights cached in `~/.ethograph/models/`, CPU path.
- [ ] Add Δ-features (frame-to-frame difference) as an option on any feature stream.
- [ ] Run Cohen's-d selection on ViT dims per event class; surface the effect-size histogram in the GUI.
- [ ] Label-coloured PCA/UMAP scatter linked to the video frame.
- [ ] **The one benchmark worth running:** kinematics vs frozen ViT-MAE CLS(+Δ), both → ASFormer, on the crow data, edit score + F1@k.

### Later / conditional
- [ ] FERAL JSON importer (per-frame probabilities → state events). Revisit as a backend when `lite` accuracy is published.
- [ ] `trainable_layers > 0` fused training for ViT extractors (`[gpu]`), reusing the E2E-Spot loop.
- [ ] `vjepa2-vitb` as `[gpu]` extractor; try odd `chunk_shift` if used for point events.
- [ ] Optional `lightning-action` TCN head for comparison against ASFormer.
- [ ] Accept a user-supplied BEAST checkpoint as an extractor; never own pretraining.
- [ ] README: 2-minute demo video, Colab, label validator (the FERAL packaging lessons).
- [ ] **Temporal ROI for long recordings — as a state event, never a fourth stage.** Point-event
  training on an unsegmented session spends almost all of it on frames where nothing is
  plausible. The composition that needs no new concept: the state default (`kinematics →
  ASFormer → StateCodec`) predicts the task bout, and the point model trains and predicts only
  inside it — stage 1 is a `TemporalHead` we already ship, on features that are already there.
  Not yet needed for pre-cut trials: on the crow data (216 trials, ~6 s each, events at a
  stereotyped 0.37/0.56 of the trial) the trial *is* the ROI, and narrowing further buys ~3x on
  class imbalance where `--stride 4` buys 4x plus 4x the temporal context for a config flag
  (`scripts/spot_point_events.md`, *200 fps is not 25 fps*). Revisit when a user brings a
  continuous session, and price it against stage-1 misses, which stage 2 cannot recover.

---

## 6. References

- Hong et al. 2022, *Spotting Temporally Precise, Fine-Grained Events in Video*, ECCV. https://arxiv.org/abs/2207.10213
- Xarles et al. 2024, *T-DEED*, CVPRW. https://github.com/arturxe2/T-DEED
- Xu et al. 2025, *MSAGSM / Multi-Focus Temporal Shifting*. https://arxiv.org/abs/2507.07381
- Liu et al. 2025, *F3Set / F3ED*, ICLR. https://github.com/F3Set/F3Set
- Xarles et al. 2026, *AdaSpot*. https://arxiv.org/abs/2602.22073
- *Few-Shot PES via Unified Multi-Entity Graph (UMEG-Net)*, AAAI 2026. https://arxiv.org/abs/2511.14186
- Wang, Guo, Liu 2023, *BME runner-up*, SoccerNet BAS. https://arxiv.org/abs/2306.05772
- Skovorodnikov, Razzauti et al. 2025/26, *FERAL*. https://github.com/Skovorp/feral · https://www.biorxiv.org/content/10.1101/2025.11.16.688666
- Wang, Yu et al. 2026, *BEAST*, ICLR. https://github.com/paninski-lab/beast · https://arxiv.org/abs/2507.09513
- Blau et al. 2024, *A study of animal action segmentation algorithms across supervised, unsupervised, and semi-supervised paradigms*. https://github.com/paninski-lab/lightning-action
- RegNet: https://arxiv.org/abs/2003.13678 · GSM: https://github.com/swathikirans/GSM · GSF: https://github.com/swathikirans/GSF
