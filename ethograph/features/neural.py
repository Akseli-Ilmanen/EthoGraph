"""Compute firing rates, PCA, and PSTH from spike times using pynapple.

Assumes spike_times are in seconds and sorted ascending (standard for Kilosort/Phy).
"""

import numpy as np
import pynapple as nap
import xarray as xr


def build_tsgroup(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    cluster_ids: np.ndarray | None = None,
    time_support: nap.IntervalSet | None = None,
) -> nap.TsGroup:
    """Build a pynapple TsGroup from flat spike arrays.

    Parameters
    ----------
    spike_times : (N,) spike times in seconds
    spike_clusters : (N,) cluster id for each spike
    cluster_ids : subset of clusters to include (defaults to all unique)
    time_support : epoch boundaries (defaults to full data range)
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
    spike_times : (N,) spike times in seconds, sorted ascending
    spike_clusters : (N,) cluster id for each spike
    bin_size : bin width in seconds
    t_start, t_stop : time range (defaults to data range)
    cluster_ids : which clusters to include (defaults to all unique)
    _tsgroup : pre-built TsGroup, pass to skip reconstruction

    Returns
    -------
    rates : (n_clusters, n_bins) firing rate in Hz
    bin_centers : (n_bins,) time of each bin center
    cluster_ids : (n_clusters,) cluster id for each row
    """
    if cluster_ids is None:
        cluster_ids = np.unique(spike_clusters.ravel())

    if t_start is None:
        t_start = float(spike_times.ravel()[0])
    if t_stop is None:
        t_stop = float(spike_times.ravel()[-1])

    time_support = nap.IntervalSet(t_start, t_stop)

    if _tsgroup is None:
        _tsgroup = build_tsgroup(
            spike_times, spike_clusters, cluster_ids, time_support,
        )
    else:
        _tsgroup = _tsgroup.restrict(time_support)

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
    firing_rate : DataArray with dims ("cluster_id", "time_fr")
    n_components : number of principal components to keep
    zscore : z-score each cluster's firing rate before PCA

    Returns
    -------
    xr.DataArray with dims ("time_fr", "pc"), coords pc=["PC1","PC2",...],
    and attrs including explained_variance ratios.
    """
    X = firing_rate.values.T  # (time, clusters)

    if zscore:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

    U, S, _ = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]

    total_var = (S ** 2).sum()
    explained = (S[:n_components] ** 2) / total_var

    pc_labels = [f"PC{i + 1}" for i in range(n_components)]

    return xr.DataArray(
        data=scores,
        dims=("time_fr", "pc"),
        coords={
            "time_fr": firing_rate.coords["time_fr"].values,
            "pc": pc_labels,
        },
        attrs={
            "type": "pca",
            "explained_variance": explained.tolist(),
            "zscore": zscore,
            "n_clusters": firing_rate.sizes["cluster_id"],
        },
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
    """Compute firing rates and return as xarray.DataArray.

    Returns
    -------
    xr.DataArray with dims ("cluster_id", "time_fr") and attrs["bin_size"].
    """
    rates, bin_centers, cluster_ids = firing_rate_by_cluster(
        spike_times, spike_clusters, bin_size, t_start, t_stop, cluster_ids,
        _tsgroup,
    )

    return xr.DataArray(
        data=rates,
        dims=("cluster_id", "time_fr"),
        coords={
            "cluster_id": cluster_ids,
            "time_fr": bin_centers,
        },
        attrs={"bin_size": bin_size, "units": "Hz", "type": "features"},
    )

