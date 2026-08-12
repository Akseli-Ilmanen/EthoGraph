"""Derived features built from what a panel plots (``io/derived.py``)."""

import numpy as np
import pytest
import xarray as xr

import ethograph as eto
from ethograph.io.catalog import XarrayLoader, catalog_from_xarray
from ethograph.io.derived import (
    DERIVED_COLUMN_DIM,
    DerivedLoader,
    Root,
    TracedArray,
    make_derived,
    stack,
)


@pytest.fixture
def loader():
    time = np.linspace(0.0, 10.0, 101)
    ds = xr.Dataset(
        {
            "angle": (("time", "individuals"), np.linspace(0.0, 360.0, 101)[:, None]),
            "position": (
                ("time", "keypoint", "individuals"),
                np.random.default_rng(0).normal(size=(101, 3, 1)),
            ),
        },
        coords={"time": time, "keypoint": ["head", "body", "tail"], "individuals": ["a"]},
        attrs={"trial": 1},
    )
    dt = eto.from_datasets([ds])
    return DerivedLoader(XarrayLoader(ds, catalog_from_xarray(ds, dt)))


def _bind(loader, feature, selections=None, t0=None, t1=None):
    """What the console does on a panel click."""
    selections = selections or {}
    plot_data = loader.select(feature, selections, t0=t0, t1=t1)
    root = Root(feature=feature, pinned=tuple(sorted((k, v) for k, v in selections.items())))
    return TracedArray(plot_data.data, time=plot_data.time, node=root, name=feature)


def test_binding_is_a_real_array(loader):
    angle = _bind(loader, "angle")
    assert isinstance(angle, np.ndarray)
    assert angle.shape == (101,)
    assert np.isclose(angle[-1], 360.0)


def test_ufunc_chain_is_traced(loader):
    theta = np.deg2rad(_bind(loader, "angle"))
    result = np.cos(theta)
    assert isinstance(result, TracedArray)
    assert result.eto_node is not None
    assert "cos" in result.eto_node.describe()


def test_recipe_re_evaluates_on_a_window_it_was_never_built_on(loader):
    """The whole point: a derived feature pans and zooms like a real one."""
    theta = np.deg2rad(_bind(loader, "angle", t0=0.0, t1=2.0))
    derived = make_derived("cos_theta", np.cos(theta))
    assert not derived.is_snapshot
    loader.register(derived)

    plot_data = loader.select("cos_theta", {}, t0=8.0, t1=10.0)
    expected = np.cos(np.deg2rad(np.linspace(0.0, 360.0, 101)))[80:]
    assert np.allclose(plot_data.data, expected)
    assert plot_data.time[0] == pytest.approx(8.0)


def test_recipe_pins_the_selections_the_panel_showed(loader):
    """A variable made from the 'head' keypoint stays the head keypoint."""
    head = _bind(loader, "position", {"keypoint": "head"})
    loader.register(make_derived("head_sq", head * head))

    plot_data = loader.select("head_sq", {}, t0=0.0, t1=10.0)
    raw = loader.base.select("position", {"keypoint": "head"}, t0=0.0, t1=10.0)
    assert np.allclose(plot_data.data, raw.data**2)


def test_multivariate_input_is_transformed_column_by_column(loader):
    """(T, D) in, (T, D) out — elementwise, no per-column bookkeeping."""
    all_keypoints = _bind(loader, "position")
    assert all_keypoints.ndim == 2
    derived = make_derived("pos_abs", np.abs(all_keypoints))
    loader.register(derived)

    plot_data = loader.select("pos_abs", {}, t0=0.0, t1=10.0)
    assert plot_data.data.shape == (101, 3)
    assert plot_data.dim_labels is not None


def test_untraceable_result_becomes_a_snapshot(loader):
    """np.diff leaves the ufunc world, so only the values survive."""
    angle = _bind(loader, "angle")
    derived = make_derived("d_angle", np.diff(angle, prepend=angle[0]), fallback_times=[np.asarray(angle.eto_time)])
    assert derived is not None
    assert derived.is_snapshot
    loader.register(derived)

    plot_data = loader.select("d_angle", {}, t0=0.0, t1=1.0)
    assert plot_data.data.shape[0] == 11


def test_a_derived_feature_can_be_built_on_another(loader):
    """The intended workflow: one line at a time, each building on the last."""
    loader.register(make_derived("theta", np.deg2rad(_bind(loader, "angle"))))
    theta = _bind(loader, "theta")
    loader.register(make_derived("wave", np.cos(theta)))

    plot_data = loader.select("wave", {}, t0=0.0, t1=10.0)
    assert np.allclose(plot_data.data, np.cos(np.deg2rad(np.linspace(0.0, 360.0, 101))))


def test_stack_makes_one_feature_with_named_columns(loader):
    """Two curves in ONE panel with a legend, not two separate features."""
    rad = np.deg2rad(_bind(loader, "angle"))
    rings = stack(sin=np.sin(rad), cos=np.cos(rad))
    assert rings.shape == (101, 2)
    assert rings.eto_labels == ["sin", "cos"]

    derived = make_derived("rings", rings)
    assert not derived.is_snapshot, "a stack of recipes must stay a recipe"
    assert derived.n_columns == 2
    loader.register(derived)

    plot_data = loader.select("rings", {}, t0=0.0, t1=10.0)
    assert plot_data.dim_labels == ["sin", "cos"]
    degrees = np.linspace(0.0, 360.0, 101)
    assert np.allclose(plot_data.data[:, 0], np.sin(np.deg2rad(degrees)))
    assert np.allclose(plot_data.data[:, 1], np.cos(np.deg2rad(degrees)))


def test_stack_labels_positional_columns_from_their_variable_names(loader):
    rad = np.deg2rad(_bind(loader, "angle"))
    sin, cos = np.sin(rad), np.cos(rad)
    sin._eto_name, cos._eto_name = "sin", "cos"  # what the console stamps

    rings = stack(sin, cos)
    assert rings.eto_labels == ["sin", "cos"]
    assert rings.shape == (101, 2)
    assert make_derived("rings", rings).is_snapshot is False


def test_stack_says_so_when_a_positional_column_has_no_name(loader):
    rad = np.deg2rad(_bind(loader, "angle"))
    with pytest.raises(ValueError, match="no name to label it with"):
        stack(np.sin(rad))
    with pytest.raises(ValueError, match="unique"):
        named = np.sin(rad)
        named._eto_name = "sin"
        stack(named, sin=np.cos(rad))


def test_stacked_feature_re_evaluates_per_window(loader):
    rad = np.deg2rad(_bind(loader, "angle", t0=0.0, t1=2.0))
    loader.register(make_derived("rings", stack(sin=np.sin(rad), cos=np.cos(rad))))

    plot_data = loader.select("rings", {}, t0=8.0, t1=10.0)
    assert plot_data.data.shape == (21, 2)
    assert plot_data.dim_labels == ["sin", "cos"]


def test_a_stacked_feature_exposes_its_columns_as_a_dim(loader):
    """A space plot picks one value per axis, so it needs a dim to pick from —
    with none, X and Y have nothing to offer and the panel comes up empty."""
    rad = np.deg2rad(_bind(loader, "angle"))
    loader.register(make_derived("rings", stack(sin=np.sin(rad), cos=np.cos(rad))))

    assert loader.feature_dims("rings") == {DERIVED_COLUMN_DIM: ["sin", "cos"]}

    degrees = np.linspace(0.0, 360.0, 101)
    for column, expected in (("sin", np.sin), ("cos", np.cos)):
        one = loader.select("rings", {DERIVED_COLUMN_DIM: column}, t0=0.0, t1=10.0)
        assert one.data.ndim == 1
        assert np.allclose(one.data, expected(np.deg2rad(degrees)))
    assert loader.select("rings", {DERIVED_COLUMN_DIM: "nope"}, t0=0.0, t1=10.0) is None


def test_a_single_column_derived_feature_has_no_dim(loader):
    loader.register(make_derived("theta", np.deg2rad(_bind(loader, "angle"))))
    assert loader.feature_dims("theta") == {}


def test_elementwise_maths_keeps_the_column_names(loader):
    rad = np.deg2rad(_bind(loader, "angle"))
    doubled = stack(sin=np.sin(rad), cos=np.cos(rad)) * 2
    assert doubled.eto_labels == ["sin", "cos"]


def test_stack_rejects_input_it_cannot_label(loader):
    rad = np.deg2rad(_bind(loader, "angle"))
    with pytest.raises(ValueError, match="1-D columns"):
        stack(pos=_bind(loader, "position"))
    with pytest.raises(ValueError, match="same length"):
        stack(a=np.sin(rad), b=np.sin(rad)[:10])
    with pytest.raises(ValueError, match="at least one"):
        stack()


def test_registering_adds_the_feature_to_the_catalog(loader):
    loader.register(make_derived("theta", np.deg2rad(_bind(loader, "angle"))))
    assert "theta" in loader.catalog.feature_choices()
    assert loader.feature_dims("theta") == {}

    loader.unregister("theta")
    assert "theta" not in loader.catalog.feature_choices()


def test_wrapper_forwards_everything_else(loader):
    assert loader.backend == loader.base.backend
    assert loader.feature_dims("angle") == loader.base.feature_dims("angle")
    assert loader.select("angle", {}, t0=0.0, t1=1.0) is not None
    assert hasattr(loader, "update_ds")


def test_non_plottable_assignment_is_rejected(loader):
    assert make_derived("scalar", 3.0) is None
    assert make_derived("text", "hello") is None
    assert make_derived("cube", np.zeros((5, 5, 5))) is None
