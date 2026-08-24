"""Compute firing rates, PCA, and PSTH from spike times using pynapple.

Assumes spike_times are in seconds and sorted ascending (standard for Kilosort/Phy).
"""

import numpy as np
import pynapple as nap
import xarray as xr

from ethograph.io import schema


def build_tsgroup(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    cluster_ids: np.ndarray | None = None,
    time_support: nap.IntervalSet | None = None,
) -> nap.TsGroup:
    """Build a pynapple TsGroup from flat spike arrays.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times in seconds, shape ``(N,)``.
    spike_clusters : numpy.ndarray
        Cluster ID for each spike, shape ``(N,)``.
    cluster_ids : numpy.ndarray, optional
        Subset of clusters to include. Defaults to all unique.
    time_support : pynapple.IntervalSet, optional
        Epoch boundaries. Defaults to full data range.

    Returns
    -------
    pynapple.TsGroup
        One :class:`pynapple.Ts` per cluster.
    """
    spike_times = spike_times.ravel()
    spike_clusters = spike_clusters.ravel()

    if cluster_ids is None:
        cluster_ids = np.unique(spike_clusters)

    if time_support is None:
        time_support = nap.IntervalSet(spike_times[0], spike_times[-1])

    units = {}
    for cid in cluster_ids:
        mask = spike_clusters == cid
        units[int(cid)] = nap.Ts(t=spike_times[mask], time_support=time_support)

    return nap.TsGroup(units, time_support=time_support)


def firing_rate_by_cluster(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    bin_size: float,
    t_start: float | None = None,
    t_stop: float | None = None,
    cluster_ids: np.ndarray | None = None,
    _tsgroup: nap.TsGroup | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin spike times per cluster into firing rate curves via pynapple.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times in seconds, sorted ascending, shape ``(N,)``.
    spike_clusters : numpy.ndarray
        Cluster ID for each spike, shape ``(N,)``.
    bin_size : float
        Bin width in seconds.
    t_start : float, optional
        Start of time range. Defaults to first spike time.
    t_stop : float, optional
        End of time range. Defaults to last spike time.
    cluster_ids : numpy.ndarray, optional
        Which clusters to include. Defaults to all unique.
    _tsgroup : pynapple.TsGroup, optional
        Pre-built TsGroup; pass to skip reconstruction.

    Returns
    -------
    rates : numpy.ndarray
        Firing rate in Hz, shape ``(n_clusters, n_bins)``.
    bin_centers : numpy.ndarray
        Time of each bin center, shape ``(n_bins,)``.
    cluster_ids : numpy.ndarray
        Cluster ID for each row, shape ``(n_clusters,)``.
    """
    if _tsgroup is not None:
        if cluster_ids is None:
            cluster_ids = np.array(list(_tsgroup.keys()))
        if t_start is None:
            t_start = float(_tsgroup.time_support.start[0])
        if t_stop is None:
            t_stop = float(_tsgroup.time_support.end[-1])
        time_support = nap.IntervalSet(t_start, t_stop)
        _tsgroup = _tsgroup.restrict(time_support)
    else:
        if cluster_ids is None:
            cluster_ids = np.unique(spike_clusters.ravel())
        if t_start is None:
            t_start = float(spike_times.ravel()[0])
        if t_stop is None:
            t_stop = float(spike_times.ravel()[-1])
        time_support = nap.IntervalSet(t_start, t_stop)
        _tsgroup = build_tsgroup(
            spike_times,
            spike_clusters,
            cluster_ids,
            time_support,
        )

    counts = _tsgroup.count(bin_size=bin_size)
    rates = (counts.values / bin_size).T
    bin_centers = counts.times()

    return rates, bin_centers, cluster_ids


def compute_pca(
    firing_rate: xr.DataArray,
    n_components: int = 3,
    zscore: bool = True,
) -> xr.DataArray:
    """Project population firing rates into PCA space via SVD.

    Parameters
    ----------
    firing_rate : xarray.DataArray
        Firing rates with dims ``("cluster_id", "time_fr")``.
    n_components : int
        Number of principal components to keep.
    zscore : bool
        Z-score each cluster's firing rate before PCA.

    Returns
    -------
    xarray.DataArray
        Scores with dims ``("time_fr", "pc")``, coords
        ``pc=["PC1", "PC2", ...]``, and ``attrs["explained_variance"]``.
    """
    X = firing_rate.values.T  # (time, clusters)

    if zscore:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

    U, S, _ = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]

    total_var = (S**2).sum()
    explained = (S[:n_components] ** 2) / total_var

    pc_labels = [f"PC{i + 1}" for i in range(n_components)]

    return schema.describe(
        xr.DataArray(
            data=scores,
            dims=("time_fr", "pc"),
            coords={
                "time_fr": firing_rate.coords["time_fr"].values,
                "pc": pc_labels,
            },
            attrs={
                "explained_variance": explained.tolist(),
                "zscore": zscore,
                "n_clusters": firing_rate.sizes["cluster_id"],
            },
        ),
        schema.NEURAL_FEATURE,
    )


def firing_rate_to_xarray(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    bin_size: float,
    t_start: float | None = None,
    t_stop: float | None = None,
    cluster_ids: np.ndarray | None = None,
    _tsgroup: nap.TsGroup | None = None,
):
    """Compute firing rates and return as an xarray DataArray.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Spike times in seconds, shape ``(N,)``.
    spike_clusters : numpy.ndarray
        Cluster ID for each spike, shape ``(N,)``.
    bin_size : float
        Bin width in seconds.
    t_start : float, optional
        Start of time range.
    t_stop : float, optional
        End of time range.
    cluster_ids : numpy.ndarray, optional
        Which clusters to include.
    _tsgroup : pynapple.TsGroup, optional
        Pre-built TsGroup.

    Returns
    -------
    xarray.DataArray
        Firing rates with dims ``("cluster_id", "time_fr")`` and
        ``attrs["bin_size"]``.
    """
    rates, bin_centers, cluster_ids = firing_rate_by_cluster(
        spike_times,
        spike_clusters,
        bin_size,
        t_start,
        t_stop,
        cluster_ids,
        _tsgroup,
    )

    return xr.DataArray(
        data=rates,
        dims=("cluster_id", "time_fr"),
        coords={
            "cluster_id": cluster_ids,
            "time_fr": bin_centers,
        },
        attrs={"bin_size": bin_size, "units": "Hz"},
    )
