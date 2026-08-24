# 3. The variable schema is advisory, and `kind` never selects behaviour

Date: 2026-08-23

## Status

Accepted, with one clause amended: the legacy `type="changepoints"` spelling
is no longer read (see "Both spellings are read" below). A raw mask is marked
by `changepoint_mask`, and an old file is converted once with
`ethograph.io.schema.migrate_legacy_attrs`.

## Context

movement [issue #978](https://github.com/neuroinformatics-unit/movement/issues/978)
proposes describing derived features by their `attrs` — `kind`, `source`,
`units`, `is_egocentric` — so that trackers and behaviour classifiers can
exchange engineered features, selected with `ds.filter_by_attrs(...)`. The
issue is explicitly a sketch and asks segmentation-tool authors what they
need. EthoGraph is such a tool, and already had two ad-hoc conventions doing
part of the job: `attrs["type"] = "changepoints"` (read in a dozen places)
and `attrs["normalise"]` (invented for the segmentation pipeline's
normalisation).

Adopting a shared vocabulary is only worth it if it removes mechanisms
rather than adding a layer over them. Two risks had to be settled:

1. A dataset that predates the convention must not degrade. Most existing
   files carry no `kind` at all.
2. A category is a tempting place to hang behaviour ("kinematic features get
   z-scored"), and that is wrong: `speed` and `heading` are both kinematic,
   but z-scoring a unit vector destroys it.

## Decision

Adopt the schema in `ethograph/io/schema.py` with three rules.

**Advisory.** Nothing validates `kind` and no code path requires it. Feature
detection stays "any variable with a time dim"; plot-type gating stays
shape-based. `kind` only refines: it groups a feature list, and it names a
group to drop in an ablation (`train.drop_kinds`).

**A label, never a switch.** No arithmetic is chosen by `kind`. Behaviour
that depends on a variable's nature reads a separate *behavioural* attr —
today only `normalise`, read where normalisation happens.

**Both spellings are read; both are written for changepoints.** `kind_of()`
maps the legacy `type="changepoints"` onto `changepoint_feature`, and
changepoint producers write both, so no file needs migrating.

*Amended:* keeping two spellings alive turned out to be the layer this ADR
set out to avoid, and it left `kind` doing a mask's job. There is now one
spelling — `kind="changepoint_feature"` labels the family, `changepoint_mask`
marks the mask — and `type="changepoints"` is neither written nor read. A
file from before is converted once with `migrate_legacy_attrs`; until then
its changepoints read as ordinary variables.

`units` is deliberately **not** adopted: we wrote it in three places and read
it nowhere, and the one plausible consumer (`plots_radial.angular_unit`)
deliberately probes the data instead of trusting metadata.

## Consequences

* Ablation by category costs one run instead of a re-materialisation, and
  replaces the old pipeline's hardcoded `no_changepoint | no_kinematic |
  no_s3d` string literals.
* A mislabelled `kind` groups a feature oddly; it cannot corrupt training.
  That is the property that makes the attr cheap to adopt incrementally.
* Two attrs mean two things to set. `describe()` is the single place that
  stamps them, so the pairing stays consistent.
* Flags are stored as `0`/`1`: NetCDF has no boolean attribute type and
  refuses to save one.
* We owe #978 feedback: that `kind` must stay advisory for consumers, that a
  category cannot carry normalisation semantics, and that what a consumer
  actually needs in order to build a model input is *what pins a column*
  (dims plus coord values), which the sketch does not cover.
