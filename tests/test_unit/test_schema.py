"""Tests for the variable schema (``ethograph.io.schema``) and the producers that stamp it."""

import numpy as np
import pytest
import xarray as xr

import ethograph as eto
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
from ethograph.io.catalog import catalog_from_xarray

KEYPOINTS = ["tail", "body", "head"]
INDIVIDUALS = ["a", "b"]
T = 5


@pytest.fixture
def position() -> xr.DataArray:
    """Two individuals, three keypoints; the second is the first shifted by (+3, +4)."""
    base = np.zeros((T, 2, len(KEYPOINTS)))
    base[:, 0, 1] = 1.0
    base[:, 0, 2] = 1.0
    base[:, 1, 2] = 1.0
    arr = np.stack([base, base + np.array([3.0, 4.0])[None, :, None]], axis=-1)
    return xr.DataArray(
        arr,
        dims=("time", "space", "keypoint", "individual"),
        coords={
            "time": np.arange(T) / 10.0,
            "space": ["x", "y"],
            "keypoint": KEYPOINTS,
            "individual": INDIVIDUALS,
        },
        attrs={"units": "px"},
    )


def _plain(name: str = "speed") -> xr.DataArray:
    return xr.DataArray(np.arange(T, dtype=float), dims=("time",), coords={"time": np.arange(T) / 10.0}, name=name)


class TestDescribe:
    def test_writes_what_is_given(self):
        da = schema.describe(_plain(), schema.KINEMATIC_FEATURE, is_egocentric=True, normalise=False)
        assert da.attrs[schema.KIND] == schema.KINEMATIC_FEATURE
        assert da.attrs[schema.IS_EGOCENTRIC] == 1
        assert da.attrs[schema.NORMALISE] == 0

    def test_omits_what_is_not_given(self):
        da = schema.describe(_plain(), schema.VIDEO_FEATURE)
        assert da.attrs == {schema.KIND: schema.VIDEO_FEATURE}

    def test_false_flags(self):
        """``is_egocentric=False`` is a statement and is written; ``normalise=True`` is not."""
        da = schema.describe(_plain(), schema.KINEMATIC_FEATURE, is_egocentric=False, normalise=True)
        assert da.attrs[schema.IS_EGOCENTRIC] == 0
        assert schema.NORMALISE not in da.attrs

    def test_extra_attrs_pass_through(self):
        da = schema.describe(_plain(), schema.CHANGEPOINT_FEATURE, target_feature="speed")
        assert da.attrs["target_feature"] == "speed"

    def test_netcdf_roundtrip(self, tmp_path):
        """Flags are ints, so a described dataset saves (NetCDF has no boolean attr)."""
        ds = xr.Dataset({"speed": schema.describe(_plain(), schema.KINEMATIC_FEATURE, is_egocentric=False)})
        path = tmp_path / "described.nc"
        ds.to_netcdf(path)
        with xr.open_dataset(path) as loaded:
            assert schema.kind_of(loaded["speed"]) == schema.KINEMATIC_FEATURE


class TestKindOf:
    def test_reads_the_new_attr(self):
        assert schema.kind_of(schema.describe(_plain(), schema.NEURAL_FEATURE)) == schema.NEURAL_FEATURE

    def test_no_legacy_type_is_a_kind(self):
        """`type` in any spelling is dead metadata until migrated."""
        for stale in ("changepoints", "audio_changepoints", "pca", "features"):
            da = _plain()
            da.attrs["type"] = stale
            assert schema.kind_of(da) is None

    def test_undeclared_variable(self):
        da = _plain()
        assert schema.kind_of(da) is None
        assert schema.is_changepoint(da) is False
        assert schema.is_normalise(da) is True
        assert schema.is_egocentric(da) is None


class TestIsChangepoint:
    """A *mask* carries its own marker; the label alone is a category."""

    def test_changepoint_attrs_mark_a_mask(self):
        cp = _plain("cp")
        cp.attrs.update(schema.changepoint_attrs(target_feature="speed"))
        assert schema.is_changepoint(cp)
        assert schema.kind_of(cp) == schema.CHANGEPOINT_FEATURE
        assert cp.attrs["target_feature"] == "speed"
        assert "type" not in cp.attrs

    def test_the_label_alone_is_not_a_mask(self):
        labelled = schema.describe(_plain("cp_sigma3"), schema.CHANGEPOINT_FEATURE)
        assert schema.kind_of(labelled) == schema.CHANGEPOINT_FEATURE
        assert not schema.is_changepoint(labelled)

    def test_the_legacy_attr_is_not_read(self):
        """`type="changepoints"` means nothing now — migrate_legacy_attrs converts it."""
        old = _plain("cp_old")
        old.attrs["type"] = "changepoints"
        assert schema.kind_of(old) is None
        assert not schema.is_changepoint(old)

    def test_dataset_helpers_find_masks_only(self):
        cp = _plain("cp")
        cp.attrs.update(schema.changepoint_attrs())
        ds = xr.Dataset(
            {
                "speed": schema.describe(_plain(), schema.KINEMATIC_FEATURE),
                "cp": cp,
                "cp_sigma3": schema.describe(_plain("cp_sigma3"), schema.CHANGEPOINT_FEATURE),
            }
        )
        assert schema.changepoint_vars(ds) == ["cp"]
        assert list(schema.filter_changepoints(ds).data_vars) == ["cp"]
        # The smooth expansion groups with the family but is not a mask.
        assert schema.kinds_in(ds)[schema.CHANGEPOINT_FEATURE] == ["cp", "cp_sigma3"]


class TestMigrateLegacyAttrs:
    def test_converts_masks_and_drops_stale_types(self):
        cp = _plain("cp")
        cp.attrs["type"] = "changepoints"
        cp.attrs["target_feature"] = "speed"
        pca = _plain("pca")
        pca.attrs["type"] = "pca"
        plain = _plain("speed")
        ds = xr.Dataset({"cp": cp, "pca": pca, "speed": plain})

        out = schema.migrate_legacy_attrs(ds)

        assert schema.is_changepoint(out["cp"])
        assert schema.kind_of(out["cp"]) == schema.CHANGEPOINT_FEATURE
        assert out["cp"].attrs["target_feature"] == "speed"
        assert "type" not in out["cp"].attrs
        # A stale type nothing ever read is dropped, and invents no kind.
        assert "type" not in out["pca"].attrs
        assert schema.kind_of(out["pca"]) is None
        # An undescribed variable is left exactly as it was.
        assert schema.kind_of(out["speed"]) is None
        assert out["speed"].attrs == {}

    def test_is_idempotent(self):
        cp = _plain("cp")
        cp.attrs["type"] = "changepoints"
        ds = xr.Dataset({"cp": cp})
        once = schema.migrate_legacy_attrs(ds)
        twice = schema.migrate_legacy_attrs(once)
        assert twice["cp"].attrs == once["cp"].attrs
        assert schema.is_changepoint(twice["cp"])


class TestKindsIn:
    def test_groups_and_omits_undeclared(self):
        ds = xr.Dataset(
            {
                "speed": schema.describe(_plain(), schema.KINEMATIC_FEATURE),
                "heading": schema.describe(_plain("heading"), schema.KINEMATIC_FEATURE, normalise=False),
                "motion": schema.describe(_plain("motion"), schema.VIDEO_FEATURE),
                "raw": _plain("raw"),
            }
        )
        assert schema.kinds_in(ds) == {
            schema.KINEMATIC_FEATURE: ["speed", "heading"],
            schema.VIDEO_FEATURE: ["motion"],
        }


class TestSelectAndDrop:
    @pytest.fixture
    def ds(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "speed": schema.describe(_plain(), schema.KINEMATIC_FEATURE),
                "motion": schema.describe(_plain("motion"), schema.VIDEO_FEATURE),
                "raw": _plain("raw"),
            }
        )

    def test_select_kinds(self, ds):
        assert schema.select_kinds(ds, [schema.KINEMATIC_FEATURE]) == ["speed"]
        assert schema.select_kinds(ds, [schema.KINEMATIC_FEATURE, schema.VIDEO_FEATURE]) == ["speed", "motion"]
        assert schema.select_kinds(ds, []) == []

    def test_drop_kinds(self, ds):
        assert schema.drop_kinds(["speed", "motion", "raw"], ds, [schema.VIDEO_FEATURE]) == ["speed", "raw"]

    def test_drop_keeps_unknown_name_and_undeclared_var(self, ds):
        names = ["speed", "motion", "raw", "not_in_dataset"]
        kept = schema.drop_kinds(names, ds, [schema.VIDEO_FEATURE, schema.KINEMATIC_FEATURE])
        assert kept == ["raw", "not_in_dataset"]


class TestGeometryStamps:
    """The matrix of kind / is_egocentric / normalise every geometry output carries."""

    def test_all_are_kinematic(self, position):
        outputs = [
            egocentric_position(position, "tail", heading_keypoint="head"),
            intra_distances(position),
            inter_distances(position, "head"),
            heading(position, "tail", "head"),
            heading_angle(position, "tail", "head"),
            joint_angles(position, [("tail", "body", "head")]),
            polygon_area(position, KEYPOINTS),
            speed_direction(position),
        ]
        for da in outputs:
            assert schema.kind_of(da) == schema.KINEMATIC_FEATURE, da.name

    def test_only_egocentric_position_is_egocentric(self, position):
        assert schema.is_egocentric(egocentric_position(position, "tail", heading_keypoint="head")) is True
        for da in (intra_distances(position), heading(position, "tail", "head"), polygon_area(position, KEYPOINTS)):
            assert schema.is_egocentric(da) is False, da.name

    def test_angles_and_unit_vectors_are_not_normalised(self, position):
        not_normalised = [
            heading(position, "tail", "head"),
            heading_angle(position, "tail", "head"),
            joint_angles(position, [("tail", "body", "head")]),
            speed_direction(position),
        ]
        for da in not_normalised:
            assert not schema.is_normalise(da), da.name

    def test_distances_areas_and_coordinates_stay_scalable(self, position):
        scalable = [
            egocentric_position(position, "tail", heading_keypoint="head"),
            intra_distances(position),
            inter_distances(position, "head"),
            polygon_area(position, KEYPOINTS),
        ]
        for da in scalable:
            assert schema.is_normalise(da), da.name

    def test_source_attrs_do_not_leak_into_the_stamp(self, position):
        """A described input cannot make its output egocentric or exempt from normalisation."""
        schema.describe(position, schema.KINEMATIC_FEATURE, is_egocentric=True, normalise=False)
        out = intra_distances(position)
        assert schema.is_egocentric(out) is False
        assert schema.is_normalise(out)
        assert out.attrs["units"] == "px"


class TestChangepointProducers:
    @pytest.fixture
    def ds(self) -> xr.Dataset:
        n = 40
        rng = np.random.default_rng(0)
        speed = xr.DataArray(
            rng.uniform(0, 5, size=(n, 2)),
            dims=("time", "individual"),
            coords={"time": np.arange(n) / 10.0, "individual": INDIVIDUALS},
        )
        cp = xr.zeros_like(speed, dtype=np.int8)
        cp[[5, 20, 33], :] = 1
        cp.attrs = schema.changepoint_attrs(target_feature="speed")
        return xr.Dataset({"speed": speed, "speed_troughs": cp})

    def test_changepoint_attrs_are_the_label_and_the_marker(self, ds):
        attrs = ds["speed_troughs"].attrs
        assert attrs[schema.KIND] == schema.CHANGEPOINT_FEATURE
        assert attrs[schema.CHANGEPOINT_MASK] == 1
        assert "type" not in attrs
        assert schema.is_changepoint(ds["speed_troughs"])
        assert schema.changepoint_vars(ds) == ["speed_troughs"]

    def test_derived_features_are_not_changepoint_masks(self, ds):
        """Only the raw binary mask is a changepoint — the smooth features are inputs, not masks."""
        out = add_changepoint_features(ds, sigmas=[2])
        assert schema.changepoint_vars(out) == ["speed_troughs"]
        assert out["speed_troughs_cp_sigma2"].attrs["normalise"] == 0


class TestAdvisory:
    """``kind`` is advisory: a dataset that declares none still behaves as before."""

    def _ds(self, described: bool, legacy_cp: bool = False) -> xr.Dataset:
        speed = xr.DataArray(
            np.arange(2 * T, dtype=float).reshape(T, 2),
            dims=("time", "individual"),
            coords={"time": np.arange(T) / 10.0, "individual": INDIVIDUALS},
        )
        heading_angle_da = speed.copy()
        cp = xr.zeros_like(speed, dtype=np.int8)
        cp[2, :] = 1
        cp.attrs = {"type": "changepoints", "target_feature": "speed"} if legacy_cp else {"target_feature": "speed"}
        if described:
            schema.describe(speed, schema.KINEMATIC_FEATURE, is_egocentric=False)
            schema.describe(heading_angle_da, schema.KINEMATIC_FEATURE, is_egocentric=False, normalise=False)
            cp.attrs = schema.changepoint_attrs(target_feature="speed")
        data = {"speed": speed, "heading_angle": heading_angle_da, "speed_troughs": cp}
        return xr.Dataset(data, attrs={"trial": 1})

    def _catalog(self, ds: xr.Dataset):
        return catalog_from_xarray(ds, eto.from_datasets([ds]))

    def test_undeclared_features_are_still_features(self):
        """No `kind` anywhere: the feature list is unchanged by the convention."""
        bare = self._catalog(self._ds(described=False))
        stamped = self._catalog(self._ds(described=True))
        assert bare.feature_choices() == ["speed", "heading_angle", "speed_troughs"]
        assert [f for f in stamped.feature_choices()] == ["speed", "heading_angle"]
        # The only difference is that a described mask is known to be a mask.
        assert stamped.changepoints == ["speed_troughs"]

    def test_an_unmigrated_legacy_mask_is_an_ordinary_feature(self):
        """Dropping legacy support has one visible cost, and migration pays it."""
        legacy = self._ds(described=False, legacy_cp=True)
        assert self._catalog(legacy).changepoints == []

        schema.migrate_legacy_attrs(legacy)
        catalog = self._catalog(legacy)
        assert catalog.changepoints == ["speed_troughs"]
        assert catalog.feature_choices() == ["speed", "heading_angle"]


class TestChangepointLabelVsPredicate:
    """`kind` labels the whole changepoint family; only a raw mask is a mask."""

    def _ds(self) -> xr.Dataset:
        time = np.arange(T) / 10.0
        speed = xr.DataArray(np.arange(T, dtype=float), dims="time", coords={"time": time})
        cp = xr.zeros_like(speed, dtype=np.int8)
        cp[2] = 1
        cp.attrs = schema.changepoint_attrs(target_feature="speed")
        return xr.Dataset({"speed": speed, "speed_troughs": cp})

    def test_derived_features_share_the_label_but_are_not_masks(self):
        from ethograph.features.changepoints import add_changepoint_features

        out = add_changepoint_features(self._ds(), sigmas=[2])
        derived = out["speed_troughs_cp_sigma2"]
        # Same category — so one ablation drops the whole family...
        assert schema.kind_of(derived) == schema.CHANGEPOINT_FEATURE
        assert schema.kind_of(out["speed_troughs"]) == schema.CHANGEPOINT_FEATURE
        # ...but only the raw binary mask is a mask.
        assert schema.is_changepoint(out["speed_troughs"])
        assert not schema.is_changepoint(derived)
        assert schema.changepoint_vars(out) == ["speed_troughs"]

    def test_derived_features_stay_visible_as_features(self):
        """The advisory rule: labelling them must not hide them from the GUI."""
        from ethograph.features.changepoints import add_changepoint_features
        from ethograph.io.catalog import _feature_vars

        out = add_changepoint_features(self._ds(), sigmas=[2])
        features = _feature_vars(out)
        assert "speed_troughs_cp_sigma2" in features
        assert "speed_troughs" not in features

    def test_a_migrated_legacy_mask_reads_as_both(self):
        """An old file becomes a proper mask once converted."""
        cp = xr.DataArray(np.zeros(T, dtype=np.int8), dims="time", attrs={"type": "changepoints"})
        out = schema.migrate_legacy_attrs(xr.Dataset({"cp": cp}))
        assert schema.kind_of(out["cp"]) == schema.CHANGEPOINT_FEATURE
        assert schema.is_changepoint(out["cp"])


class TestPynappleBackend:
    """A TsGroup says the same thing as a DataArray, in columns instead of attrs."""

    def _group(self, n_units: int = 2, **extra):
        nap = pytest.importorskip("pynapple")
        units = {i: nap.Ts(t=np.array([1.0, 3.5])) for i in range(n_units)}
        group = nap.TsGroup(units)
        group.set_info(**schema.changepoint_metadata(n_units, **extra))
        return group

    def test_metadata_mirrors_the_attrs(self):
        n = 3
        columns = schema.changepoint_metadata(n, target_feature="speed")
        assert set(columns) == set(schema.changepoint_attrs(target_feature="speed"))
        assert columns[schema.KIND] == [schema.CHANGEPOINT_FEATURE] * n
        assert columns[schema.CHANGEPOINT_MASK] == [1] * n
        assert columns["target_feature"] == ["speed"] * n

    def test_changepoint_units_finds_the_marked_units(self):
        group = self._group(2, target_feature="speed")
        assert schema.changepoint_units(group.metadata) == [0, 1]

    def test_an_unmarked_group_has_no_changepoint_units(self):
        nap = pytest.importorskip("pynapple")
        group = nap.TsGroup({0: nap.Ts(t=np.array([1.0, 3.5]))})
        group.set_info(source_label=["nose"])
        assert schema.changepoint_units(group.metadata) == []
        assert schema.changepoint_units(None) == []

    def test_the_legacy_metadata_column_is_not_read(self):
        nap = pytest.importorskip("pynapple")
        group = nap.TsGroup({0: nap.Ts(t=np.array([1.0, 3.5]))})
        group.set_info(type=["changepoints"])
        assert schema.changepoint_units(group.metadata) == []


class TestSidecar:
    """Where a backend without attrs keeps its schema."""

    def test_path_is_beside_the_alignment(self, tmp_path):
        folder = tmp_path / "sess"
        folder.mkdir()
        assert schema.sidecar_path(folder) == folder / ".ethograph" / "schema.yaml"
        # For a file source it sits beside the file, like the alignment does.
        npz = folder / "speed.npz"
        npz.write_bytes(b"")
        assert schema.sidecar_path(npz) == folder / ".ethograph" / "schema.yaml"

    def test_absent_sidecar_is_empty_not_an_error(self, tmp_path):
        assert schema.read_sidecar(tmp_path) == {}

    def test_roundtrip_normalises_booleans(self, tmp_path):
        """Written as 0/1 so the sidecar says the same thing NetCDF attrs do."""
        schema.write_sidecar(
            tmp_path,
            {
                "speed": {schema.KIND: schema.KINEMATIC_FEATURE},
                "heading": {schema.KIND: schema.KINEMATIC_FEATURE, schema.NORMALISE: False},
            },
        )
        out = schema.read_sidecar(tmp_path)
        assert out["speed"] == {schema.KIND: schema.KINEMATIC_FEATURE}
        assert out["heading"][schema.NORMALISE] == 0
        assert not schema.is_normalise(out["heading"])
        assert schema.kind_of(out["speed"]) == schema.KINEMATIC_FEATURE

    def test_a_malformed_sidecar_says_so(self, tmp_path):
        path = schema.sidecar_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("- not\n- a mapping\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected a mapping"):
            schema.read_sidecar(tmp_path)
        path.write_text("speed: kinematic_feature\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must map to a mapping"):
            schema.read_sidecar(tmp_path)


class TestAttrsOf:
    """Readers take a DataArray or a plain mapping, so both backends work."""

    def test_a_mapping_reads_like_a_dataarray(self):
        as_attrs = schema.describe(_plain(), schema.VIDEO_FEATURE, normalise=False)
        as_dict = {schema.KIND: schema.VIDEO_FEATURE, schema.NORMALISE: 0}
        for var in (as_attrs, as_dict):
            assert schema.kind_of(var) == schema.VIDEO_FEATURE
            assert not schema.is_normalise(var)

    def test_empty_and_none(self):
        assert schema.kind_of({}) is None
        assert schema.kind_of(None) is None
        assert schema.is_normalise({}) is True
