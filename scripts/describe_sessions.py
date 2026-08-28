"""Declare what each session variable *is*, so an ablation has something to drop.

``train.drop_kinds`` names a **kind** (``ethograph/io/schema.py``), and a kind
is a property of the variable, not of a run: it has to be declared once, beside
the session. Variables built by ethograph's own code carry theirs already
(``features/geometry.py`` stamps ``kinematic_feature``, the S3D extractor
``video_feature``, the changepoint expansion ``changepoint_feature``), but a
session assembled elsewhere — a MATLAB export, a merge written before the
convention — says nothing, and an ablation naming a kind nothing declares
silently drops no column at all.

This writes the **schema sidecar** each session's loader already reads:
``{session}/.ethograph/schema.yaml``, beside its ``alignment.nwb``. The
``.nc`` is never touched, a variable's own attrs still win over the sidecar,
and only :data:`~ethograph.io.schema.KIND` is written — never ``normalise``,
which changes arithmetic — so a dataset materialised before and after is
numerically identical.

Every feature the config selects is a kinematic feature unless it is named in
:data:`VIDEO_VARS`; the columns ``features.changepoint_features`` generates are
left out, because they are built at session-open time and stamp themselves.

    python scripts/describe_sessions.py                     # data/project.yaml
    python scripts/describe_sessions.py --dry-run           # say what it would write
    python scripts/describe_sessions.py data/crow1.yaml     # another config
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

import ethograph as eto
from ethograph.io import schema

# NOTE: this whole script is a stopgap. A variable's ``kind`` belongs to
# whatever builds it — ``features/geometry.py`` and the S3D extractor already
# stamp theirs — so the sessions here should carry their kinds from the code
# that writes the ``.nc``, and this sidecar pass should then be deleted rather
# than kept in step with the table below.

#: Variable → the kind it declares, spelled out rather than inferred: a rule
#: like "everything that is not S3D is kinematic" would quietly label the next
#: neural or audio feature kinematic, and an ablation would then drop it
#: without saying so. A feature in ``features.columns`` and missing here is an
#: error (:func:`kinds_for`), so adding one to the config forces a decision.
KINDS: dict[str, str] = {
    # kinematic features
    "position": schema.KINEMATIC_FEATURE,
    "velocity": schema.KINEMATIC_FEATURE,
    "speed": schema.KINEMATIC_FEATURE,
    "acceleration": schema.KINEMATIC_FEATURE,
    "angles": schema.KINEMATIC_FEATURE,
    # distances between keypoints
    "pellet_beakTip_dist": schema.KINEMATIC_FEATURE,
    "pellet_stickTip_dist": schema.KINEMATIC_FEATURE,
    "pellet_stickClosest_dist": schema.KINEMATIC_FEATURE,
    "disp_beakTip_dist": schema.KINEMATIC_FEATURE,
    "disp_stickTip_dist": schema.KINEMATIC_FEATURE,
    "sticktip_cornerLFront_dist": schema.KINEMATIC_FEATURE,
    "sticktip_cornerLBack_dist": schema.KINEMATIC_FEATURE,
    "sticktip_cornerRBack_dist": schema.KINEMATIC_FEATURE,
    "sticktip_cornerRFront_dist": schema.KINEMATIC_FEATURE,
    # kinematic features (accelerometer)
    "aux_acceleration": schema.KINEMATIC_FEATURE,
    "aux_velocity": schema.KINEMATIC_FEATURE,
    "aux_speed": schema.KINEMATIC_FEATURE,
    # video features
    "s3d": schema.VIDEO_FEATURE,
}

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "data" / "project.yaml"

logger = logging.getLogger("describe")


def kinds_for(config: eto.segment.SegmentConfig) -> dict[str, str]:
    """Every variable the config selects → the kind :data:`KINDS` gives it.

    The columns ``features.changepoint_features`` generates are left out: they
    are built at session-open time and stamp themselves ``changepoint_feature``.
    """
    generated = set()
    if config.features.changepoint_features is not None:
        generated = set(config.features.changepoint_features.expanded_columns())
    selected = [name for name in config.features.columns if name not in generated]
    undeclared = [name for name in selected if name not in KINDS]
    if undeclared:
        raise SystemExit(
            f"{sorted(undeclared)} are selected by features.columns but named in no group in "
            f"{__file__}. Add each to KINDS (see ethograph.io.schema.KNOWN_KINDS) — an ablation "
            f"drops a kind, so a variable with none is silently always kept."
        )
    return {name: KINDS[name] for name in selected}


def describe_session(source: Path, kinds: dict[str, str], dry_run: bool) -> dict[str, str]:
    """Merge *kinds* into *source*'s sidecar; return the entries it changes."""
    existing = schema.read_sidecar(source)
    changes = {name: kind for name, kind in kinds.items() if existing.get(name, {}).get(schema.KIND) != kind}
    if not changes:
        return {}
    merged = {name: {**existing.get(name, {}), schema.KIND: kind} for name, kind in kinds.items()}
    for name, attrs in existing.items():
        merged.setdefault(name, attrs)
    if not dry_run:
        schema.write_sidecar(source, merged)
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG, help="segmentation config to read")
    parser.add_argument("--dry-run", action="store_true", help="print what would be written; write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = eto.segment.Project(args.config).config
    kinds = kinds_for(config)
    logger.info("%s selects %d variables:\n%s", args.config, len(kinds), yaml.safe_dump(kinds, sort_keys=False))
    for spec in config.sessions:
        changed = describe_session(spec.source, kinds, args.dry_run)
        path = schema.sidecar_path(spec.source)
        if changed:
            verb = "would write" if args.dry_run else "wrote"
            logger.info("%s %s (%d variable(s) newly declared)", verb, path, len(changed))
        else:
            logger.info("%s already declares every kind", path)


if __name__ == "__main__":
    main()
