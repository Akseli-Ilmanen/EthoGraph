# 5. Search on a ratio split, then cross-validate by session

Date: 2026-08-23

## Status

Accepted. Replaces the per-session `role` key introduced with the pipeline.

## Context

The pipeline divided trials by giving each **session** a `role` in the config
— `train`, `test` or `infer` — with validation carved out of the training
sessions by `train.split.val_fraction`. Two problems showed up as soon as the
pipeline was used for more than one run.

**Roles conflated two unrelated questions.** "Which sessions do I have?" is a
property of the dataset and changes rarely. "What is this particular run
holding out?" is a property of the experiment and changes constantly. Writing
both in the same list meant editing the dataset every time you wanted to try a
different holdout, and a config no longer described a project — it described
one run of one project.

**Validation was doing a job nobody asked it to do.** It selected the
checkpoint, which is worth having, but the actual decisions — learning rate,
loss weights, feature-map counts — were still made by hand, by reading
`compare()` and editing the file. That is a hyperparameter search run by a
human at one configuration per coffee break.

Meanwhile the thing we most wanted to see — *where* the model is still wrong —
was the one thing the split could not give. A random trial split's test trials
share a recording day, lighting and animal with the trials the model trained
on, and they are scattered across sessions rather than making up one you can
open in the GUI beside the labels you drew.

## Decision

Split the workflow in two, and let each stage divide the trials the way its
own question needs.

**Stage 1 — search.** `train.split` becomes three ratios that must sum to 1
(`train_fraction` / `val_fraction` / `test_fraction`, 60/20/20 by default),
drawn by whole trial across every session. `project.search()` runs an Optuna
study over `search.params`, every trial a full training run scored by
`train.select_on` **on the validation trials**. Validation now has exactly one
job and it is the one it is good at. The winner is written to
`searches/{name}/best.yaml` — a config that inherits the one searched and pins
the parameters that won.

**Stage 2 — cross-validate.** `project.cross_validate()` holds out one whole
session per fold, trains on the rest, and runs inference over the held-out
session. Roles survive here, but as something the *fold* computes rather than
something the config declares: `train.split.holdout_sessions`, written per
fold. Folds are independent, so `folds=["ses-01", "ses-02"]` runs two of six.

`role` is deleted from `SessionSpec`; a config still carrying one is an error
naming the replacement. `infer()` defaults to every session, since there is no
longer an `infer` role to select by, with `sessions=` to narrow it.

A search space is keyed by the same dotted path an override uses
(`train.learning_rate`), so a setting keeps exactly one name across the file,
an override and a search — the same reasoning ADR 0004 applied to the CLI.

## Consequences

* A config describes a **project**, not a run. Adding a session is editing the
  dataset; choosing a holdout is calling a method.
* Every fold's predictions are a labels TSV beside its own session, written by
  a model that never saw it — so "60% F1" becomes *which* class, *which*
  trials and *how far off* the boundaries are, in the GUI, against the curated
  labels. This is the payoff the whole change is for.
* Optuna is a new dependency of the `model` extra. The study is a SQLite file
  under `searches/{name}/`, so a search resumes rather than restarts, and a
  trial whose validation curve is already behind the others is abandoned.
* The three fractions must sum to 1, which is a constraint the old two
  independent fractions did not have. It is worth it: 60/20/20 is one number a
  reader can check, where `val_fraction: 0.2, test_fraction: 0` left the third
  share implicit.
* `train.split` no longer reads DLC2Action's `val_frac`/`test_frac`. Those
  defaults belong to upstream's own training loop and its notion of a sample
  (a 128-frame window); a sample here is a whole trial, and 60/20/20 is the
  ratio the two-stage workflow needs. Documented at the field, as ADR 0001
  requires of anything we default ourselves.
* A study's trials and a cross-validation's folds are ordinary runs nested one
  level deeper (`runs/{name}/…`), so `compare()` — which reads only the top
  level — keeps showing the runs trained by hand.
* Cross-validation costs N training runs. That is the price of an honest
  answer about a session the model has never seen, and `folds=` is there for
  when you only want a sample of it.
