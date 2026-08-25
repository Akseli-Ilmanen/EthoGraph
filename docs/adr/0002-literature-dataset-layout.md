# 2. The materialised dataset uses the action-segmentation literature's layout

Date: 2026-08-22

## Status

Accepted

## Context

Feature engineering has to write something Train reads. Two candidates: a
torch `Dataset` reading `.nc` + labels TSV directly (no intermediate files),
or the on-disk layout used by MS-TCN, MS-TCN++, ASFormer, DiffAct, FACT,
LTContext and ASRF — `features/*.npy (F, T)`, `groundTruth/*.txt` (a class
name per frame), `mapping.txt`, `splits/*.bundle`. New models from that
literature ship loaders for the latter.

## Decision

Feature engineering writes the literature layout under
`{root}/data/{features.name}/`, plus two files the literature lacks and this
project needs to round-trip: `index.tsv` (which sample came from which
session, trial and individual, at what rate) and `columns.yaml` (the input
layout with normalise flags and vector groups), and `classes.yaml` (class
index ↔ label id). Sample keys are `{session_id}_trial{trial}_{individual}`.
Roles and normalisation statistics are *not* stored in the dataset; a run
records them (`splits/`, `stats.npz`).

## Consequences

* Any model that reads the literature layout can be pointed at the dataset
  with no adapter.
* "What did the model see?" has exactly one answer on disk.
* The dataset is role-agnostic, so one materialisation serves every split
  and every run of a benchmark.
* Text ground truth is larger and slower than integer arrays; at the trial
  lengths involved this is irrelevant.
