For me:

## Build

```bash
# Live preview (auto-rebuilds on save, serves at http://127.0.0.1:8000)
sphinx-autobuild docs/source docs/build/html
sphinx-build -b html docs/source docs/build/html # single build?

# Full rebuild (cleans first)
cd docs && make clean && make html

# Incremental rebuild (faster, only changed files)
cd docs && make html

# Windows
cd docs && make.bat html
```

Output lands in `docs/build/html/`. Open `docs/build/html/index.html` in a browser.

## Cross-references (MyST Markdown)

```markdown
{doc}`loading_script`                              # link to a sibling page
{func}`~ethograph.utils.xr_utils.sel_valid`        # link to a Python function
{class}`~pynapple.TsdFrame`                        # link to an external class (intersphinx)
{attr}`~pynwb.file.NWBFile.subject`                # link to an attribute
```

The `~` prefix hides the module path, showing only the short name.

## Tabs

```markdown
::::{tab-set}

:::{tab-item} Tab Title
Content here.
:::

::::
```

## Admonitions

````markdown
```{note}
This is a note.
```

```{warning}
This is a warning.
```
````

## Debug broken cross-references

Build with `-n` (nitpicky mode) to surface all broken refs:

```bash
sphinx-build -n -M html docs/source docs/build
```

Or check the build output for lines like `WARNING: py:func reference target not found`.

## Find the right intersphinx reference string

```bash
python -m sphinx.ext.intersphinx https://pynwb.readthedocs.io/en/stable/objects.inv | grep Subject
```

## Force full rebuild

Sphinx caches doctrees. If something looks stale:

```bash
cd docs && make clean && make html
```

## Config reference

| Setting | Location |
|---------|----------|
| Extensions, theme, intersphinx | `docs/source/conf.py` |
| Doc dependencies | `pyproject.toml` `[project.optional-dependencies]` → `docs` |
| Notebook execution | `nb_execution_mode = "off"` in conf.py |
| Custom CSS | `docs/source/_static/css/custom.css` |
