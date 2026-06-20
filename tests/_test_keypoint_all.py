import ethograph as eto
from ethograph.io.catalog import catalog_from_xarray, XarrayLoader

candidates = [
    "data/margot_data/test.nc",
    "data/20210119_Recording_SR1_SR2_social_vidtwo/pair24.nc",
    "data/whisker/ab.nc",
    "data/canary/hb.nc",
]

for f in candidates:
    try:
        dt = eto.open(f)
    except Exception as e:
        print(f, "OPEN ERR", e)
        continue
    ds = dt.itrial(0)
    kp = [d for d in ds.dims if "key" in d.lower()]
    if not kp:
        print(f, "no keypoint dim; dims=", dict(ds.sizes))
        continue
    print("=" * 60)
    print("FILE", f)
    print("dims", dict(ds.sizes))
    for v in ds.data_vars:
        print("  var", v, ds[v].dims)
    cat = catalog_from_xarray(ds, dt)
    print("combos:", {k: len(s.values) for k, s in cat.combos.items()})
    loader = XarrayLoader(ds, cat)
    print("loader.dims keys:", list(loader.dims.keys()))

    # pick a feature with keypoint dim
    feat = next((v for v in ds.data_vars if any("key" in str(d).lower() for d in ds[v].dims)), None)
    print("feature with keypoint:", feat)
    kpdim = [d for d in ds[feat].dims if "key" in str(d).lower()][0]
    kpvals = [str(x) for x in ds.coords[kpdim].values] if kpdim in ds.coords else None
    print("keypoint dim name:", kpdim, "vals:", kpvals)

    # Simulate "All": keypoint omitted
    sel_all = {}
    # add specific selections for other non-time dims (single)
    for d in ds[feat].dims:
        dn = str(d)
        if "time" in dn.lower() or "key" in dn.lower():
            continue
        if dn in ds.coords:
            sel_all[dn] = str(ds.coords[dn].values[0])
    print("selections (All keypoint):", sel_all)
    pdat = loader.select(feat, dict(sel_all))
    print("  -> data.shape", None if pdat is None else pdat.data.shape, "dim_labels", None if pdat is None else pdat.dim_labels)

    # Simulate single keypoint
    sel_single = dict(sel_all)
    sel_single[kpdim] = kpvals[0]
    print("selections (single keypoint):", sel_single)
    pdat2 = loader.select(feat, dict(sel_single))
    print("  -> data.shape", None if pdat2 is None else pdat2.data.shape)
    break
