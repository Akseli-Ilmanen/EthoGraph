# 6. Post-processing takes the GUI's settings by default

Date: 2026-08-24

## Status

Accepted.

## Context

Label post-processing — purge short labels, stitch small gaps, snap edges
onto changepoints — exists twice: the GUI's *CP Correction* section, whose
values live in `~/.ethograph/gui_settings.yaml`, and `infer.postprocess` in
a project config, which cleans a model's predictions with the same
functions (`features/changepoints.correct_changepoints`). The two sets had
different names for the same numbers and nothing kept them equal, so the
labels a person curated by hand and the predictions they compared them to
could be cleaned with different tolerances without anyone noticing.

The GUI is where these numbers get tuned: you see a boundary move and set
the tolerance by eye. The pipeline is where they get reproduced. Most users
of the GUI never write a project config at all, and the GUI must work for
them exactly as it does now.

## Decision

`gui_settings.yaml` stays the one store, global, edited through the GUI.
A project config can *take* those numbers instead of spelling them:

```yaml
infer:
  postprocess:
    gui_settings: true      # or a path to a gui_settings.yaml
```

`config.GUI_POSTPROCESS_KEYS` is the translation from `PostprocessConfig`
fields to the GUI's keys and the one place it is written; a test holds it to
`AppStateSpec.VARS`. The values are read every time the config is loaded,
so the pipeline follows the GUI. Anything spelled explicitly beside
`gui_settings` — in the file, a `base:` chain, or a dotlist override — wins,
so a project with its own needs spells them, and a sweep can still override
one number over the GUI's base.

A saved run config carries the resolved values explicitly and records the
path in place of `true`: a finished run does not change when the GUI does,
and it says where its numbers came from.

The GUI's step checkboxes have no pipeline counterpart — the pipeline derives
its steps from whether each value is meaningful — so an unticked step reads
as its value zeroed.

## Consequences

- One place to tune, and the same numbers on both sides by default.
- The GUI keeps its own store and needs no project config; nothing in
  `ethograph/gui` imports from `ethograph/segment`.
- The boundary-head settings and the `changepoints` selection stay the
  config's: the GUI holds neither.
- The reverse direction — the GUI reading a project config — is not offered.
  It would make the GUI depend on a file most of its users do not have, and
  the GUI's settings are already the source.
