# Reference geometries

Reference geometries (arenas, 3D objects, …) that can be drawn behind the
GUI's Space plot.

## How loading works

On first GUI launch, every `*.yaml` file in this folder is copied into the
user's geometry library at `~/.ethograph/geometries/`. Users add their own by
dropping more `*.yaml` files into that directory. Each file is one selectable
geometry, identified by its filename (without extension); all `references` in
the file are drawn together.

To pick which geometry is drawn, either:

- select its filename in the GUI under **Data → Space controls →
  "Library geometry:"**, or
- set it as a default in `gui_settings.yaml` (global) or a dataset's
  `local_settings.yaml` (per-project) — these are equivalent to the GUI
  selection:

  ```yaml
  space_library_geometry: moll2025_geometry
  ```

## Schema

```yaml
references:
  - name: setup            # label used in log messages only
    vertices:              # [x, y] or [x, y, z] points
      - [0.0, 0.0, 0.0]
      - [1.0, 0.0, 0.0]
    edges:                 # pairs of vertex indices
      - [0, 1]
    color: black           # any pyqtgraph color name/hex
```

## Contributing

Add a `*.yaml` file to this folder via PR to ship it as a default. Existing
user libraries are never overwritten or re-seeded, so people who delete or edit
geometries locally keep their changes.
