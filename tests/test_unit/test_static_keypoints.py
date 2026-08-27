"""Static keypoints: labelled once, present everywhere.

What earns a test is the invariant nothing else enforces — one position per
static keypoint, on every frame, through every reader (canvas accessors, the
fill's input and output, the export, the sidecar round trip, the seed into the
next video).
"""

from __future__ import annotations

import numpy as np

from ethograph.gui.pose_annotate import KeypointStore, store_to_movement_ds


def _store(n_frames=10) -> KeypointStore:
    s = KeypointStore(keypoint_names=["beak", "cornerL", "cornerR"], n_frames=n_frames)
    s.set_static("cornerL", True)
    s.set_static("cornerR", True)
    return s


class TestOnePosition:
    def test_placed_once_read_everywhere(self):
        s = _store()
        s.set_point(3, "cornerL", (10.0, 20.0))
        for frame in (0, 3, 9):
            assert s.anchor_positions(frame)[0, 1].tolist() == [10.0, 20.0]
            assert s.positions(frame)[0, 1].tolist() == [10.0, 20.0]
        assert len(s.anchors) == 1  # the sidecar stays one frame

    def test_placing_again_moves_it_not_duplicates_it(self):
        s = _store()
        s.set_point(3, "cornerL", (10.0, 20.0))
        s.set_point(7, "cornerL", (11.0, 21.0))
        assert s.anchor_positions(0)[0, 1].tolist() == [11.0, 21.0]
        assert 3 not in s.anchors  # the old frame was emptied and dropped

    def test_clearing_clears_everywhere(self):
        s = _store()
        s.set_point(3, "cornerL", (10.0, 20.0))
        s.clear_point(8, "cornerL")  # any frame will do
        assert np.isnan(s.anchor_positions(3)[0, 1, 0])
        assert not s.anchors

    def test_a_moving_keypoint_is_untouched(self):
        s = _store()
        s.set_point(2, "beak", (1.0, 1.0))
        s.set_point(5, "beak", (2.0, 2.0))
        assert np.isnan(s.anchor_positions(3)[0, 0, 0])
        assert len(s.anchors) == 2

    def test_making_static_keeps_the_first_label_only(self):
        s = KeypointStore(keypoint_names=["c"], n_frames=10)
        s.set_point(4, "c", (1.0, 1.0))
        s.set_point(6, "c", (2.0, 2.0))
        s.set_static("c", True)
        assert s.static_anchor("c").tolist() == [1.0, 1.0]
        assert 6 not in s.anchors


class TestFillAndExport:
    def test_observations_carry_the_static_points_on_every_labelled_frame(self):
        s = _store()
        s.set_point(0, "cornerL", (10.0, 20.0))
        s.set_point(2, "beak", (1.0, 1.0))
        s.set_point(8, "beak", (3.0, 3.0))
        obs = s.observations()
        # frame 0 holds only the corner: not an observation of anything that
        # moves, so it must not widen the span the fill covers
        assert sorted(obs) == [2, 8]
        assert obs[8][0, 1].tolist() == [10.0, 20.0]
        assert obs[2][0, 1].tolist() == [10.0, 20.0]

    def test_pin_static_overrides_a_drifting_fill(self):
        s = _store()
        s.set_point(0, "cornerL", (10.0, 20.0))
        filled = np.random.default_rng(0).normal(size=(10, 1, 3, 2))
        conf = np.full((10, 1, 3), 0.5)
        pinned, pconf = s.pin_static(filled, conf)
        assert np.all(pinned[:, 0, 1] == [10.0, 20.0])
        assert np.all(pconf[:, 0, 1] == 1.0)
        assert np.array_equal(pinned[:, 0, 0], filled[:, 0, 0])  # the beak column is the backend's
        assert np.array_equal(
            pinned[:, 0, 2], filled[:, 0, 2]
        )  # an unplaced static keypoint is left as the backend had it

    def test_export_writes_the_static_point_on_every_frame(self):
        s = _store(n_frames=5)
        s.set_point(2, "cornerL", (10.0, 20.0))
        ds = store_to_movement_ds(s, fps=25.0)
        corner = ds.position.sel(keypoint="cornerL").isel(individual=0).transpose("time", "space").values
        assert np.all(corner == [10.0, 20.0])
        assert np.all(ds.confidence.sel(keypoint="cornerL").values == 1.0)
        assert np.isnan(ds.position.sel(keypoint="beak").values).all()


class TestSidecarAndSeed:
    def test_round_trips_through_the_sidecar(self):
        s = _store()
        s.set_point(3, "cornerL", (10.0, 20.0))
        again = KeypointStore.from_dict(s.to_dict())
        assert again.static_keypoints == ["cornerL", "cornerR"]
        assert again.anchor_positions(9)[0, 1].tolist() == [10.0, 20.0]

    def test_seed_copies_static_points_into_a_new_video(self):
        old = _store()
        old.set_point(3, "cornerL", (10.0, 20.0))
        old.set_point(5, "cornerR", (90.0, 20.0))
        new = KeypointStore(keypoint_names=["beak", "cornerL", "cornerR"], n_frames=40)
        assert new.seed_static_from(old) == 2
        assert new.static_keypoints == ["cornerL", "cornerR"]
        assert new.anchor_positions(39)[0, 2].tolist() == [90.0, 20.0]

    def test_seed_never_overwrites_and_skips_unknown_names(self):
        old = _store()
        old.set_point(3, "cornerL", (10.0, 20.0))
        new = KeypointStore(keypoint_names=["beak", "cornerL"], n_frames=40)
        new.set_static("cornerL", True)
        new.set_point(1, "cornerL", (99.0, 99.0))
        assert new.seed_static_from(old) == 0
        assert new.static_anchor("cornerL").tolist() == [99.0, 99.0]
