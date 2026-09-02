"""Tests for ethograph.features.geometry and add_changepoint_features."""

import numpy as np
import pytest
import xarray as xr

from ethograph.features.changepoints import add_changepoint_features
from ethograph.features.geometry import (
    egocentric_position,
    heading,
    heading_angle,
    inter_distances,
    intra_distances,
    joint_angles,
    polygon_area,
    speed_direction,
)
from ethograph.io import schema

KEYPOINTS = ["tail", "body", "head"]
INDIVIDUALS = ["a", "b"]
T = 5


def _position(space: list[str]) -> xr.DataArray:
    """Two individuals, three keypoints; the second individual is the first shifted by (+3, +4[, +0])."""
    n_space = len(space)
    base = np.zeros((T, n_space, len(KEYPOINTS)))
    # tail at (0, 0), body at (1, 0), head at (1, 1): a right angle at body
    base[:, 0, 1] = 1.0
    base[:, 0, 2] = 1.0
    base[:, 1, 2] = 1.0
    if n_space == 3:
        base[:, 2, :] = 2.0
    shift = np.zeros(n_space)
    shift[:2] = [3.0, 4.0]
    arr = np.stack([base, base + shift[None, :, None]], axis=-1)
    return xr.DataArray(
        arr,
        dims=("time", "space", "keypoint", "individual"),
        coords={"time": np.arange(T) / 10.0, "space": space, "keypoint": KEYPOINTS, "individual": INDIVIDUALS},
        attrs={"units": "px"},
    )


@pytest.fixture
def position() -> xr.DataArray:
    return _position(["x", "y"])


@pytest.fixture
def position3d() -> xr.DataArray:
    return _position(["x", "y", "z"])


class TestEgocentric:
    def test_centre_at_origin_heading_on_x(self, position):
        ego = egocentric_position(position, "tail", heading_keypoint="head")
        assert ego.dims == ("time", "space", "keypoint", "individual")
        np.testing.assert_allclose(ego.sel(keypoint="tail"), 0.0, atol=1e-12)
        head = ego.sel(keypoint="head")
        np.testing.assert_allclose(head.sel(space="y"), 0.0, atol=1e-12)
        np.testing.assert_allclose(head.sel(space="x"), np.sqrt(2.0))
        # the individual shift is removed by centring
        np.testing.assert_allclose(ego.sel(individual="a").values, ego.sel(individual="b").values, atol=1e-12)

    def test_3d_keeps_z_translated_only(self, position3d):
        ego = egocentric_position(position3d, "tail", heading_keypoint="head")
        np.testing.assert_allclose(ego.sel(space="z"), 0.0, atol=1e-12)
        np.testing.assert_allclose(ego.sel(keypoint="head", space="y"), 0.0, atol=1e-12)

    def test_nan_propagates(self, position):
        position = position.copy()
        position[0, :, 2, 0] = np.nan
        ego = egocentric_position(position, "tail", heading_keypoint="head")
        assert np.isnan(ego.isel(time=0, individual=0)).all()
        assert not np.isnan(ego.isel(time=1)).any()

    def test_single_individual_keeps_dim(self, position):
        ego = egocentric_position(position.isel(individual=[0]), "tail", heading_keypoint="head")
        assert ego.sizes["individual"] == 1

    def test_left_right_pair_heading(self, position):
        # centre="body" (1,0); left="tail" (0,0), right="head" (1,1) -> forward perpendicular to tail-head
        ego = egocentric_position(position, "body", left_keypoint="tail", right_keypoint="head")
        np.testing.assert_allclose(ego.sel(keypoint="body"), 0.0, atol=1e-12)
        half_sqrt2 = np.sqrt(2.0) / 2.0
        np.testing.assert_allclose(ego.sel(keypoint="tail", space="x"), -half_sqrt2)
        np.testing.assert_allclose(ego.sel(keypoint="tail", space="y"), -half_sqrt2)
        np.testing.assert_allclose(ego.sel(keypoint="head", space="x"), -half_sqrt2)
        np.testing.assert_allclose(ego.sel(keypoint="head", space="y"), half_sqrt2)

    def test_mutually_exclusive_heading_args_required(self, position):
        with pytest.raises(ValueError, match="heading_keypoint"):
            egocentric_position(position, "tail")

    def test_centroid_centre(self, position):
        ego = egocentric_position(position, centre_on_centroid=True, heading_keypoint="head")
        np.testing.assert_allclose(ego.mean(dim="keypoint"), 0.0, atol=1e-12)
        # the individual shift is removed by centring, same as the named-keypoint case
        np.testing.assert_allclose(ego.sel(individual="a").values, ego.sel(individual="b").values, atol=1e-12)

    def test_mutually_exclusive_centre_args_required(self, position):
        with pytest.raises(ValueError, match="centre_on_centroid"):
            egocentric_position(position, heading_keypoint="head")
        with pytest.raises(ValueError, match="centre_on_centroid"):
            egocentric_position(position, "tail", centre_on_centroid=True, heading_keypoint="head")


class TestIntraDistances:
    def test_pairs_and_values(self, position):
        d = intra_distances(position)
        assert d.dims == ("time", "pair", "individual")
        assert list(d.pair.values) == ["tail-body", "tail-head", "body-head"]
        np.testing.assert_allclose(d.sel(pair="tail-body"), 1.0)
        np.testing.assert_allclose(d.sel(pair="tail-head"), np.sqrt(2.0))
        np.testing.assert_allclose(d.sel(pair="body-head"), 1.0)

    def test_symmetric_under_keypoint_order(self, position):
        forward = intra_distances(position, ["tail", "head"])
        backward = intra_distances(position, ["head", "tail"])
        np.testing.assert_allclose(forward.values, backward.values)

    def test_3d(self, position3d):
        d = intra_distances(position3d)
        np.testing.assert_allclose(d.sel(pair="tail-head"), np.sqrt(2.0))


class TestInterDistances:
    def test_diagonal_nan_offdiagonal_hand_computed(self, position):
        d = inter_distances(position, "head")
        assert d.dims == ("time", "individual", "other")
        assert list(d.other.values) == INDIVIDUALS
        assert np.isnan(d.sel(individual="a", other="a")).all()
        assert np.isnan(d.sel(individual="b", other="b")).all()
        np.testing.assert_allclose(d.sel(individual="a", other="b"), 5.0)
        np.testing.assert_allclose(d.sel(individual="b", other="a"), 5.0)

    def test_single_individual_raises(self, position):
        with pytest.raises(ValueError):
            inter_distances(position.isel(individual=[0]), "head")


class TestHeading:
    def test_unit_norm_and_attrs(self, position):
        h = heading(position, "tail", "head")
        assert h.dims == ("time", "space", "individual")
        np.testing.assert_allclose(np.linalg.norm(h.values, axis=1), 1.0)
        np.testing.assert_allclose(h.sel(space="x"), np.sqrt(0.5))
        assert h.attrs["normalise"] == 0
        assert "units" not in h.attrs

    def test_coincident_points_nan(self, position):
        h = heading(position, "tail", "tail")
        assert np.isnan(h.values).all()

    def test_3d_unit_norm(self, position3d):
        h = heading(position3d, "tail", "head")
        np.testing.assert_allclose(np.linalg.norm(h.values, axis=1), 1.0)

    def test_angle(self, position):
        a = heading_angle(position, "tail", "head")
        assert a.dims == ("time", "individual")
        np.testing.assert_allclose(a, np.pi / 4)
        assert a.attrs["normalise"] == 0
        assert a.attrs["units"] == "rad"

    def test_angle_range_excludes_minus_pi(self, position):
        a = heading_angle(position, "body", "tail")
        np.testing.assert_allclose(a, np.pi)


class TestJointAngles:
    def test_right_angle(self, position):
        ja = joint_angles(position, [("tail", "body", "head")])
        assert ja.dims == ("time", "angle", "individual")
        assert list(ja.angle.values) == ["tail-body-head"]
        np.testing.assert_allclose(ja, np.pi / 2)
        # an angle in radians is not normalised, exactly like heading_angle
        assert ja.attrs["normalise"] == 0

    def test_acute_and_degenerate(self, position):
        ja = joint_angles(position, [("tail", "head", "body"), ("tail", "tail", "head")])
        np.testing.assert_allclose(ja.sel(angle="tail-head-body"), np.pi / 4)
        assert np.isnan(ja.sel(angle="tail-tail-head")).all()


class TestPolygonArea:
    def test_unit_square(self, position):
        square = np.zeros((T, 2, 4, 2))
        square[:, 0, 1] = 1.0
        square[:, :, 2] = 1.0
        square[:, 1, 3] = 1.0
        pos = xr.DataArray(
            square,
            dims=("time", "space", "keypoint", "individual"),
            coords={
                "time": position.time,
                "space": ["x", "y"],
                "keypoint": ["p0", "p1", "p2", "p3"],
                "individual": INDIVIDUALS,
            },
        )
        area = polygon_area(pos, ["p0", "p1", "p2", "p3"])
        assert area.dims == ("time", "individual")
        np.testing.assert_allclose(area, 1.0)

    def test_triangle(self, position):
        np.testing.assert_allclose(polygon_area(position, KEYPOINTS), 0.5)


class TestSpeedDirection:
    def test_unit_norm(self, position):
        rng = np.random.default_rng(0)
        velocity = position.copy(data=rng.normal(size=position.shape))
        velocity[0, :, 0, 0] = 0.0
        sd = speed_direction(velocity)
        assert sd.dims == position.dims
        norms = np.linalg.norm(sd.values, axis=1)
        assert np.isnan(norms[0, 0, 0])
        np.testing.assert_allclose(norms[1:], 1.0)
        assert sd.attrs["normalise"] == 0


class TestAddChangepointFeatures:
    @pytest.fixture
    def ds(self) -> xr.Dataset:
        rng = np.random.default_rng(1)
        n = 40
        time = np.arange(n) / 10.0
        speed = xr.DataArray(
            rng.uniform(0, 5, size=(n, 3, 2)),
            dims=("time", "keypoint", "individual"),
            coords={"time": time, "keypoint": KEYPOINTS, "individual": INDIVIDUALS},
        )
        cp = xr.zeros_like(speed, dtype=np.int8)
        cp[[5, 20, 33], :, :] = 1
        cp.attrs = schema.changepoint_attrs(target_feature="speed")
        return xr.Dataset({"speed": speed, "speed_troughs": cp})

    def test_names_dims_attrs(self, ds):
        out = add_changepoint_features(ds, sigmas=[0.5, 3])
        expected = [
            "speed_troughs_cp_binary",
            "speed_troughs_cp_prox0",
            "speed_troughs_cp_prox1",
            "speed_troughs_cp_since",
            "speed_troughs_cp_until",
            "speed_troughs_cp_length",
        ]
        for name in expected:
            assert name in out.data_vars, name
            assert out[name].dims == ds["speed_troughs"].dims
            assert out[name].attrs["normalise"] == 0
            # the family's label, but not the mask marker
            assert out[name].attrs[schema.KIND] == schema.CHANGEPOINT_FEATURE
            assert schema.CHANGEPOINT_MASK not in out[name].attrs
        # the raw input stays the only changepoint mask
        assert schema.changepoint_vars(out) == ["speed_troughs"]

    def test_values_match_per_column(self, ds):
        out = add_changepoint_features(ds, sigmas=[2])
        binary = out["speed_troughs_cp_binary"].sel(keypoint="head", individual="b")
        np.testing.assert_array_equal(binary, ds["speed_troughs"].sel(keypoint="head", individual="b"))
        smooth = out["speed_troughs_cp_prox0"].sel(keypoint="head", individual="b").values
        # an isolated changepoint reads 1 at its frame — kernels add, nothing is normalised per trial
        assert smooth[5] == pytest.approx(1.0, abs=1e-2)
        assert smooth[5] > smooth[7] > smooth[9] > 0
        # offsets: default horizon 4 * max(sigmas) = 8 samples, clipped and scaled to [0, 1]
        since = out["speed_troughs_cp_since"].sel(keypoint="head", individual="b").values
        until = out["speed_troughs_cp_until"].sel(keypoint="head", individual="b").values
        assert since[5] == 0 and until[5] == 0
        assert since[6] == pytest.approx(1 / 8) and until[4] == pytest.approx(1 / 8)
        assert since[13] == 1.0 and since[0] == 1.0  # saturated: horizon reached / no candidate yet
        assert until[-1] == 1.0  # nothing ahead of the last changepoint
        # length: log scale, saturating at 16 horizons = 128 samples; segments here are 5, 15, 13, 7
        length = out["speed_troughs_cp_length"].sel(keypoint="head", individual="b").values
        assert length[2] == pytest.approx(np.log1p(5) / np.log1p(128))
        assert length[10] == pytest.approx(np.log1p(15) / np.log1p(128))
        assert length[10] > length[2] and length.max() < 1.0

    def test_max_length_is_where_the_length_saturates(self, ds):
        out = add_changepoint_features(ds, sigmas=[2], max_length=10)
        length = out["speed_troughs_cp_length"].sel(keypoint="head", individual="b").values
        assert length[2] == pytest.approx(np.log1p(5) / np.log1p(10))
        assert length[10] == 1.0

    def test_horizon_is_where_the_offsets_saturate(self, ds):
        out = add_changepoint_features(ds, sigmas=[2], horizon=3)
        since = out["speed_troughs_cp_since"].sel(keypoint="head", individual="b").values
        assert since[6] == pytest.approx(1 / 3) and since[8] == 1.0
        with pytest.raises(ValueError, match="positive"):
            add_changepoint_features(ds, sigmas=[2], horizon=0)

    def test_scale_by_with_fewer_dims_raises(self, ds):
        ds["global_cp"] = ds["speed_troughs"].isel(keypoint=0, individual=0, drop=True)
        ds["global_cp"].attrs = schema.changepoint_attrs(target_feature="speed")
        with pytest.raises(ValueError, match="pin them first"):
            add_changepoint_features(ds, sigmas=[1], scale_by="speed")

    def test_scale_by_emphasises_changepoints_where_the_signal_is_low(self, ds):
        other = xr.zeros_like(ds["speed"])
        other[5, :, :] = 100.0  # the first changepoint sits on a high value, the other two on zero
        ds["other"] = other
        plain = add_changepoint_features(ds, sigmas=[1])["speed_troughs_cp_prox0"]
        scaled = add_changepoint_features(ds, sigmas=[1], scale_by="other")["speed_troughs_cp_prox0"]
        plain = plain.sel(keypoint="head", individual="b").values
        scaled = scaled.sel(keypoint="head", individual="b").values
        assert plain[5] == pytest.approx(plain[20], abs=1e-3)
        assert scaled[5] < 1e-6 < 0.99 < scaled[20]
        out = add_changepoint_features(ds, sigmas=[1], scale_by="other")
        assert "sigma 1 samples, scaled by other" in out["speed_troughs_cp_prox0"].attrs["description"]

    def test_scales_from_durations(self):
        from ethograph.features.changepoints import scales_from_durations

        rng = np.random.default_rng(0)
        durations = rng.uniform(0.2, 2.0, size=200)  # seconds
        scales = scales_from_durations(durations, fs=100.0)
        p5, p95 = np.percentile(durations, [5, 95])
        assert scales.horizon == pytest.approx(0.5 * p5 * 100.0, abs=1e-4)
        assert scales.max_length == pytest.approx(p95 * 100.0, abs=1e-4)
        assert scales.sigmas == pytest.approx((scales.horizon / 16, scales.horizon / 8, scales.horizon / 4), abs=1e-4)
        # a horizon never drops below one sample, and max_length never below two horizons
        tiny = scales_from_durations([0.001, 0.002], fs=10.0)
        assert tiny.horizon == 1.0 and tiny.max_length == 2.0
        with pytest.raises(ValueError, match="at least two"):
            scales_from_durations([0.5], fs=100.0)
