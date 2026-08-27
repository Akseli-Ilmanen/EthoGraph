# 4. The segmentation pipeline is scripted, with no command line

Date: 2026-08-23

## Status

Accepted. Supersedes the CLI described in ADR 0002's consequences.

## Context

The pipeline first shipped with both a Python API and a `ethograph segment
features | train | infer | compare | video-features | architectures`
command line. The CLI grew quickly: `video-features` alone reached twelve
flags, six of which (`--stack-s`, `--analysis-fps`, `--mode`,
`--truncate-at`, `--precision`, `--device`) duplicated keys that already
existed in the config, and two mutually exclusive invocation modes had to be
validated against each other by hand.

That is the cost of two interfaces over one set of functions: every setting
needs a name in both, every combination needs a rule, and a reader of
someone's analysis cannot tell from the config alone what was actually run.

## Decision

Delete the command line. One YAML config becomes a
:class:`~ethograph.segment.project.Project`, and each stage is a method on
it. Overrides keep the same dotted `key=value` syntax, passed to the
constructor or to `update()`, so a benchmark is an ordinary Python loop:

```python
import ethograph as eto

for architecture in ("asformer", "mstcn", "mlp"):
    eto.segment.Project(
        "project.yaml",
        f"model.architecture={architecture}",
        f"train.run_name={architecture}",
    ).train()

print(eto.segment.Project("project.yaml").compare())
```

The same reasoning trimmed `VideoFeaturesConfig` to the settings that change
the *features* — `stack_s`, `analysis_fps`, `camera`. Batch size, decode
chunk, precision, device and the `dense` ablation mode each have one
sensible answer and stay inside `S3DConfig` for the rare caller who needs
them.

## Consequences

* A pipeline run is a file: diffable, re-runnable, and reviewable next to
  the results it produced.
* There is one place a setting can be named, so a config is a complete
  description of a run.
* No shell-quoting or flag-combination rules to test; the argument parser
  and its two-mode validation are gone.
* Interactive use is slightly longer to type — three lines of Python instead
  of one shell command. This is the trade accepted for the above.
* `ethograph launch` and `ethograph shortcut` are unaffected; they are GUI
  entry points, not analysis.
