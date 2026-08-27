"""The closed loop, end to end: labelled clips in, video-only predictions out.

    python scripts/spot_closed_loop.py data/spot/project.yaml --backend spline

Every step is a method on ``eto.spot.Project``; this script only strings them
together in the order they depend on each other, and stops where the pipeline
does. What runs today:

    fill_poses      sidecars the labelling dialog left -> <video>.keypoints.nc
    merge_poses     each clip's keypoints onto its trial's clock, into the session
    materialise     frames/ + dataset/ + keypoints/ (the entity graph per trial)
    train_teacher   the graph model on the keypoints; writes its embeddings
    train           the pixel model on labels only — the baseline the student must beat
    distil          the student: the baseline taught the teacher's embeddings, then its head
    inference       the student's predictions into each session's labels/

``merge_poses`` writes a sibling ``{stem}_pose2d.nc`` per session; the config
then has to point at those files for the later stages. ``--in-place`` writes
the keypoints into the session files themselves instead, so the config stays
as it is — the one place this script changes a file you did not create.
"""

from __future__ import annotations

import argparse
import logging

import ethograph as eto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="spot project YAML")
    parser.add_argument("--backend", default="spline", choices=["spline", "flow", "posepal"], help="fill backend")
    parser.add_argument("--in-place", action="store_true", help="merge keypoints into the session files themselves")
    parser.add_argument("--pose-var", default="position", help="variable name the merged keypoints land in")
    parser.add_argument("--skip-pixels", action="store_true", help="stop after the teacher (no pixel training)")
    parser.add_argument("override", nargs="*", help="dotted key=value overrides, e.g. trials.limit=20")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    project = eto.spot.Project(args.config, *args.override)
    exported = project.fill_poses(backend=args.backend)
    logging.info("fill_poses: %d clips exported", len(exported))
    merged = project.merge_poses(var=args.pose_var, in_place=args.in_place)
    logging.info("merge_poses: %s", ", ".join(str(p) for p in merged))
    if not args.in_place:
        logging.warning(
            "The merged sessions are siblings of the originals; point the config's sessions at them "
            "(or re-run with --in-place) before materialise() reads %r.",
            args.pose_var,
        )
    project.materialise()
    teacher_dir = project.train_teacher()
    logging.info("teacher: %s", teacher_dir)
    if args.skip_pixels:
        return
    baseline = project.train()
    logging.info("pixel baseline: %s", baseline.run_dir)
    student = project.distil()
    logging.info("student: %s", student.run_dir)
    for tsv in project.inference(run=student.run_dir):
        logging.info("predictions: %s", tsv)


if __name__ == "__main__":
    main()
