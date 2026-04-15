# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


# Used when building API docs, put the dependencies
# of any class you are documenting here
autodoc_mock_imports = [
    "audioio",
    "av",
    "crowsetta",
    "dandi",
    "h5py",
    "jinja2",
    "lindi",
    "matplotlib",
    "movement",
    "napari",
    "napari_pyav",
    "natsort",
    "neo",
    "noisereduce",
    "numpy",
    "pandas",
    "phylib",
    "pynapple",
    "pynwb",
    "pyqtgraph",
    "qtpy",
    "remfile",
    "ruptures",
    "scipy",
    "sounddevice",
    "soundfile",
    "torch",
    "torchvision",
    "tqdm",
    "vocalpy",
    "vocalseg",
    "xarray",
    "yaml",
]

# Add the module path to sys.path here.
sys.path.insert(0, os.path.abspath("../.."))

project = "ethograph"
copyright = "2022, Akseli Ilmanen"
author = "Akseli Ilmanen"
release = ""

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_sitemap",
    "myst_nb",
    "sphinx_design",
]

# Don't execute notebooks during build
nb_execution_mode = "off"

# Configure the myst parser to enable cool markdown features
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Automatically generate stub pages for API
autosummary_generate = True
autosummary_generate_overwrite = False
autodoc_default_flags = ["members", "inherited-members"]

# List of patterns to ignore when looking for source files.
exclude_patterns = [
    "**.ipynb_checkpoints",
    "**/includes/**",
    "api_generated/**",
    "user_guide/data_index.md",
    "user_guide/gui_index.md",
    "user_guide/examples.md",
    "media/changepoints.ipynb",
]

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pynapple": ("https://pynapple.org/", None),
    "movement": (
        "https://movement.neuroinformatics.dev/latest/",
        None,
    ),
    "pynwb": ("https://pynwb.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "ethograph"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_theme_options = {
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Akseli-Ilmanen/ethograph",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "logo": {
        "image_light": "media/icon.png",
        "image_dark": "media/icon.png",
        "text": project,
    },
}

# GitHub pages
github_user = "Akseli-Ilmanen"
html_baseurl = f"https://{github_user}.github.io/{project}"
sitemap_url_scheme = "{link}"


# -- Copy tutorial notebooks into examples/ at build time --------------------

import shutil
from pathlib import Path


def _remove_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _sync_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        _remove_if_exists(dst)
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def setup(app):
    src = Path(app.srcdir).resolve().parent.parent / "examples"
    dst = Path(app.srcdir) / "examples"
    dst.mkdir(exist_ok=True)

    keep_paths: set[Path] = {dst / "index.rst", dst / ".gitignore"}

    for nb in src.glob("*.ipynb"):
        # Fix double extensions like foo.ipynb.ipynb -> foo.ipynb
        name = nb.name
        while name.endswith(".ipynb.ipynb"):
            name = name[: -len(".ipynb")]
        target = dst / name
        _sync_link_or_copy(nb, target)
        keep_paths.add(target)

    # Also link/copy the assets folder if it exists.
    assets_src = src / "assets"
    assets_dst = dst / "assets"
    if assets_src.is_dir():
        _sync_link_or_copy(assets_src, assets_dst)
        keep_paths.add(assets_dst)

    # Remove stale generated files from prior runs.
    for child in dst.iterdir():
        if child in keep_paths:
            continue
        _remove_if_exists(child)
