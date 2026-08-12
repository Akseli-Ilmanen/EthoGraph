"""Derived features — user-defined arrays built from what a panel plots.

The console panel (``gui/plots_console.py``) binds the *rendered* contents of
the clicked panel to a Python name, so the user works with exactly what they
see: a ``(T,)``/``(T, D)`` numpy array for the panel's current feature,
selections and visible window — never the underlying multi-dimensional
DataArray.

Anything the user assigns in the console becomes a new feature offered by the
add-panel popup.  Two flavours, decided by whether the expression could be
traced:

* **Recipe** — the whole expression was numpy ufuncs (``np.deg2rad``,
  ``np.cos``, ``+``, ``/``, …), so :class:`TracedArray` recorded the graph.
  The derived feature re-evaluates that graph per window and per trial, with
  the source feature's *selections pinned* to what the panel showed when the
  variable was made.  It pans, zooms and follows trials like any other feature.
* **Snapshot** — the expression left the ufunc world (``np.diff``,
  ``savgol_filter``, a slice), so only the values survive.  The feature is
  frozen to the time vector it was made on and shows nothing outside it.

Recipes are elementwise by construction, so a ``(T, D)`` input is transformed
column by column for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ethograph.io.catalog import ComboSpec, PlotData

# ---------------------------------------------------------------------------
# Expression graph
# ---------------------------------------------------------------------------


class Node:
    """One step of a recorded expression."""

    def roots(self) -> list[Root]:
        raise NotImplementedError

    def evaluate(self, values: dict[Root, np.ndarray]) -> Any:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Root(Node):
    """A panel's plotted contents: one feature under pinned selections."""

    feature: str
    pinned: tuple[tuple[str, str], ...]

    @property
    def selections(self) -> dict[str, str]:
        return dict(self.pinned)

    def roots(self) -> list[Root]:
        return [self]

    def evaluate(self, values: dict[Root, np.ndarray]) -> Any:
        return values[self]

    def describe(self) -> str:
        return self.feature


class Const(Node):
    """A scalar operand (``theta * 2``)."""

    def __init__(self, value: Any):
        self.value = value

    def roots(self) -> list[Root]:
        return []

    def evaluate(self, values: dict[Root, np.ndarray]) -> Any:
        return self.value

    def describe(self) -> str:
        return repr(self.value)


class Ufunc(Node):
    """A numpy ufunc applied to other nodes."""

    def __init__(self, ufunc, inputs: tuple[Node, ...], kwargs: dict | None = None):
        self.ufunc = ufunc
        self.inputs = inputs
        self.kwargs = dict(kwargs or {})

    def roots(self) -> list[Root]:
        out: list[Root] = []
        for node in self.inputs:
            for root in node.roots():
                if root not in out:
                    out.append(root)
        return out

    def evaluate(self, values: dict[Root, np.ndarray]) -> Any:
        args = [node.evaluate(values) for node in self.inputs]
        return self.ufunc(*args, **self.kwargs)

    def describe(self) -> str:
        return f"np.{self.ufunc.__name__}({', '.join(n.describe() for n in self.inputs)})"


class Stack(Node):
    """Several 1-D expressions side by side as one ``(T, D)`` feature.

    This is how a user gets two curves in *one* panel with a legend rather than
    two panels: the column names become ``dim_labels``, which is what
    ``plot_multidim`` colours and labels by.
    """

    def __init__(self, inputs: tuple[Node, ...], labels: list[str]):
        self.inputs = inputs
        self.labels = list(labels)

    def roots(self) -> list[Root]:
        out: list[Root] = []
        for node in self.inputs:
            for root in node.roots():
                if root not in out:
                    out.append(root)
        return out

    def evaluate(self, values: dict[Root, np.ndarray]) -> Any:
        return np.column_stack([np.asarray(node.evaluate(values)) for node in self.inputs])

    def describe(self) -> str:
        parts = [f"{label}={node.describe()}" for label, node in zip(self.labels, self.inputs)]
        return f"stack({', '.join(parts)})"


# ---------------------------------------------------------------------------
# TracedArray
# ---------------------------------------------------------------------------


class TracedArray(np.ndarray):
    """A real ndarray that also remembers the expression that produced it.

    It *is* an ndarray, so ``np.deg2rad(speed)``, ``speed.mean()``,
    ``speed[:10]`` and ``speed.shape`` all behave exactly as the user expects.
    ``__array_ufunc__`` additionally records the operation, which is what lets
    a derived feature re-evaluate itself on a window it was never built on.

    Anything that is not a ufunc call (a slice, ``np.diff``, a scipy filter)
    produces an array with no recipe — still usable, but only as a snapshot.
    """

    def __new__(
        cls,
        values,
        time=None,
        node: Node | None = None,
        name: str | None = None,
        labels: list[str] | None = None,
    ):
        obj = np.asarray(values).view(cls)
        obj._eto_time = None if time is None else np.asarray(time)
        obj._eto_node = node
        obj._eto_name = name
        obj._eto_labels = None if labels is None else list(labels)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        time = getattr(obj, "_eto_time", None)
        # A view or slice is no longer described by the parent's recipe, and its
        # time axis only still applies while the length lines up.
        if time is not None and (self.ndim == 0 or self.shape[0] != len(time)):
            time = None
        self._eto_time = time
        self._eto_node = None
        self._eto_name = None
        self._eto_labels = None

    @property
    def eto_time(self) -> np.ndarray | None:
        return self._eto_time

    @property
    def eto_node(self) -> Node | None:
        return self._eto_node

    @property
    def eto_labels(self) -> list[str] | None:
        return self._eto_labels

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raw = tuple(np.asarray(x) if isinstance(x, np.ndarray) else x for x in inputs)
        if "out" in kwargs:
            kwargs = dict(kwargs)
            kwargs["out"] = tuple(np.asarray(o) for o in kwargs["out"])
        result = getattr(ufunc, method)(*raw, **kwargs)
        if method != "__call__" or not isinstance(result, np.ndarray):
            return result

        time = next(
            (x._eto_time for x in inputs if isinstance(x, TracedArray) and x._eto_time is not None),
            None,
        )
        # Column names survive elementwise maths: np.abs(stacked) is still the
        # same columns, so the legend must not be lost on the way through.
        labels = next(
            (x._eto_labels for x in inputs if isinstance(x, TracedArray) and x._eto_labels is not None),
            None,
        )
        if labels is not None and (result.ndim != 2 or result.shape[1] != len(labels)):
            labels = None
        nodes: list[Node] = []
        traceable = True
        for x in inputs:
            if isinstance(x, TracedArray) and x._eto_node is not None:
                nodes.append(x._eto_node)
            elif isinstance(x, np.ndarray):
                traceable = False
                break
            else:
                nodes.append(Const(x))
        node = Ufunc(ufunc, tuple(nodes), kwargs) if traceable else None
        return TracedArray(result, time=time, node=node, labels=labels)


# ---------------------------------------------------------------------------
# Derived feature
# ---------------------------------------------------------------------------


class DerivedFeature:
    """A user-defined feature: either a re-evaluable recipe or a snapshot."""

    def __init__(
        self,
        name: str,
        *,
        node: Node | None = None,
        time: np.ndarray | None = None,
        values: np.ndarray | None = None,
        dim_labels: list[str] | None = None,
        n_columns: int = 1,
    ):
        if node is None and (time is None or values is None):
            raise ValueError("A derived feature needs either a recipe node or time + values")
        self.name = name
        self.node = node
        self.time = None if time is None else np.asarray(time)
        self.values = None if values is None else np.asarray(values)
        self.dim_labels = dim_labels
        #: Column count recorded at creation (recipes have no data of their own);
        #: the add-panel popup uses it to decide whether Heatmap / Space fit.
        self._n_columns = n_columns

    @property
    def is_snapshot(self) -> bool:
        return self.node is None

    @property
    def n_columns(self) -> int:
        if self.values is not None:
            return 1 if self.values.ndim == 1 else int(self.values.shape[1])
        return self._n_columns

    def describe(self) -> str:
        if self.node is None:
            return f"snapshot, {len(self.time)} samples"
        return self.node.describe()

    def evaluate(self, loader, t0: float | None, t1: float | None) -> PlotData | None:
        if self.node is None:
            return self._evaluate_snapshot(t0, t1)

        values: dict[Root, np.ndarray] = {}
        time = None
        dim_labels = None
        for root in self.node.roots():
            plot_data = loader.select(root.feature, root.selections, t0=t0, t1=t1)
            if plot_data is None:
                return None
            values[root] = np.asarray(plot_data.data)
            if time is None:
                time, dim_labels = plot_data.time, plot_data.dim_labels
        if time is None or len(time) == 0:
            return None

        data = np.asarray(self.node.evaluate(values))
        if data.ndim == 0 or data.shape[0] != len(time):
            return None
        # Explicit column names (from ``stack``) win over the root feature's:
        # the columns are the user's, not the source's.
        if self.dim_labels is not None:
            dim_labels = self.dim_labels
        if dim_labels is not None and (data.ndim != 2 or data.shape[1] != len(dim_labels)):
            dim_labels = None
        return PlotData(
            time=np.asarray(time),
            data=data,
            dim_labels=dim_labels,
            title=self.name,
            ylabel=self.name,
        )

    def _evaluate_snapshot(self, t0: float | None, t1: float | None) -> PlotData | None:
        time, values = self.time, self.values
        if t0 is not None and t1 is not None:
            mask = (time >= t0) & (time <= t1)
            time, values = time[mask], values[mask]
        if len(time) == 0:
            return None
        return PlotData(
            time=time,
            data=values,
            dim_labels=self.dim_labels,
            title=self.name,
            ylabel=self.name,
        )


def stack(*columns, **named) -> TracedArray:
    """Combine 1-D expressions into one ``(T, D)`` feature, one column each.

    ``stack(sin, cos)`` and ``stack(sin=np.sin(rad), cos=np.cos(rad))`` both
    work: a positional column takes its label from the variable's own name
    (stamped on it when it became a feature), a keyword column from the
    keyword. Either way the result is a single panel with one coloured,
    legended curve per column, instead of the separate one-column features
    plain assignment would make — and it stays a recipe as long as every
    column is one.
    """
    labelled: list[tuple[str, Any]] = []
    for index, value in enumerate(columns):
        label = getattr(value, "_eto_name", None)
        if not label:
            raise ValueError(
                f"stack() argument {index + 1} has no name to label it with — pass it as a keyword, e.g. stack(sin=...)"
            )
        labelled.append((label, value))
    labelled += list(named.items())

    if not labelled:
        raise ValueError("stack() needs at least one column, e.g. stack(sin, cos)")
    labels = [label for label, _ in labelled]
    if len(set(labels)) != len(labels):
        raise ValueError(f"stack() column names must be unique, got {labels}")

    arrays = [np.asarray(value) for _, value in labelled]
    bad = [label for label, a in zip(labels, arrays) if a.ndim != 1]
    if bad:
        raise ValueError(f"stack() takes 1-D columns; {', '.join(bad)} is not")
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"stack() columns must be the same length, got {sorted(lengths)}")

    values = [value for _, value in labelled]
    traced = [v for v in values if isinstance(v, TracedArray)]
    time = next((v._eto_time for v in traced if v._eto_time is not None), None)
    nodes = [v._eto_node for v in traced]
    complete = len(traced) == len(arrays) and all(node is not None for node in nodes)
    node = Stack(tuple(nodes), labels) if complete else None
    return TracedArray(np.column_stack(arrays), time=time, node=node, labels=labels)


def make_derived(name: str, value, fallback_times: list[np.ndarray] | None = None) -> DerivedFeature | None:
    """Build a :class:`DerivedFeature` from a value the user assigned.

    Returns ``None`` when the value cannot be plotted against time — a scalar,
    a 3-D array, or an array whose length matches no known time axis.
    """
    array = np.asarray(value)
    if array.ndim not in (1, 2) or array.shape[0] < 2 or not np.issubdtype(array.dtype, np.number):
        return None

    n_columns = 1 if array.ndim == 1 else int(array.shape[1])
    labels = value._eto_labels if isinstance(value, TracedArray) else None
    node = value._eto_node if isinstance(value, TracedArray) else None
    if node is not None:
        return DerivedFeature(name, node=node, dim_labels=labels, n_columns=n_columns)

    time = value._eto_time if isinstance(value, TracedArray) else None
    if time is None or len(time) != array.shape[0]:
        time = next((t for t in fallback_times or [] if len(t) == array.shape[0]), None)
    if time is None:
        return None
    return DerivedFeature(name, time=np.asarray(time), values=np.asarray(array), dim_labels=labels)


# ---------------------------------------------------------------------------
# DerivedLoader
# ---------------------------------------------------------------------------

#: The name of the ``D`` axis of a stacked ``(T, D)`` feature — deliberately
#: generic, since its columns are whatever the user stacked. Space plots (and
#: anything else pinning one value per axis) address a column with
#: ``{DERIVED_COLUMN_DIM: "sin"}``.
DERIVED_COLUMN_DIM = "Dimension"


def _select_column(plot_data: PlotData, column: str) -> PlotData | None:
    """One named column of a stacked feature, as a 1-D ``PlotData``."""
    labels = [str(label) for label in plot_data.dim_labels or []]
    if column not in labels or plot_data.data.ndim != 2:
        return None
    values = plot_data.data[:, labels.index(column)]
    return PlotData(
        time=plot_data.time,
        data=values,
        dim_labels=None,
        title=f"{plot_data.title}.{column}" if plot_data.title else column,
        ylabel=plot_data.ylabel,
    )


class DerivedLoader:
    """A :class:`~ethograph.io.catalog.DataLoader` plus user-derived features.

    Every attribute it does not define itself is forwarded to the wrapped
    loader, so ``update_ds``, ``backend``, ``dims`` and friends keep working
    and callers never need to know they hold a wrapper.  Registering a feature
    also appends it to the wrapped catalog, which is what makes it appear in
    the features combo and the add-panel popup without touching either.
    """

    def __init__(self, base):
        self._base = base
        self._derived: dict[str, DerivedFeature] = {}
        # The catalog's feature list before any derived feature was added.
        # Captured once so `forget()` can restore it exactly — filtering the
        # live list by what is still derived would strand the name that was
        # just removed.
        self._base_features: list[str] | None = None
        self._base_choices: tuple[str, ...] | None = None

    def __getattr__(self, item):
        # Only reached for attributes this class does not define. Private names
        # are never forwarded, so a missing ``_base`` raises instead of looping.
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._base, item)

    @property
    def base(self):
        return self._base

    @property
    def derived(self) -> dict[str, DerivedFeature]:
        return dict(self._derived)

    def is_derived(self, feature: str) -> bool:
        return feature in self._derived

    def select(
        self,
        feature: str,
        selections: dict[str, str],
        t0: float | None = None,
        t1: float | None = None,
        color_variable: str | None = None,
    ) -> PlotData | None:
        derived = self._derived.get(feature)
        if derived is None:
            return self._base.select(feature, selections, t0=t0, t1=t1, color_variable=color_variable)
        # Resolved against *self*, not the base loader, so a variable built
        # from an earlier derived one ("wave = np.cos(theta)") resolves through
        # the chain. A recipe can only name features that already existed, so
        # this cannot cycle.
        plot_data = derived.evaluate(self, t0, t1)
        column = (selections or {}).get(DERIVED_COLUMN_DIM)
        if plot_data is None or column is None:
            return plot_data
        return _select_column(plot_data, str(column))

    def feature_dims(self, feature: str) -> dict[str, list[str]]:
        derived = self._derived.get(feature)
        if derived is None:
            return self._base.feature_dims(feature)
        # A derived feature has no xarray dims, but a ``stack``'s columns ARE a
        # selectable axis, and callers that pick one value per axis need one:
        # a space plot with no dim to offer has no X or Y to choose, so the
        # panel comes up empty.
        labels = derived.dim_labels
        if labels and derived.n_columns > 1:
            return {DERIVED_COLUMN_DIM: [str(label) for label in labels]}
        return {}

    # ------------------------------------------------------------------

    def register(self, feature: DerivedFeature) -> None:
        self._derived[feature.name] = feature
        self._sync_catalog()

    def unregister(self, name: str) -> None:
        if self._derived.pop(name, None) is not None:
            self._sync_catalog()

    def _sync_catalog(self) -> None:
        catalog = getattr(self._base, "catalog", None)
        if catalog is None:
            return
        if self._base_features is None:
            self._base_features = list(catalog.features)
            spec = catalog.combos.get("features")
            self._base_choices = tuple(spec.values) if spec is not None else None
        catalog.features = list(self._base_features) + list(self._derived)
        if self._base_choices is not None:
            catalog.combos["features"] = ComboSpec("features", self._base_choices + tuple(self._derived))


def derived_loader_for(app_state) -> DerivedLoader | None:
    """The session's :class:`DerivedLoader`, or ``None`` when data is unloaded."""
    loader = getattr(app_state, "data_loader", None)
    return loader if isinstance(loader, DerivedLoader) else None
