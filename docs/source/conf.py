# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from importlib.metadata import version as get_version

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
try:
    full_version = get_version(project)
    release = full_version.split("+", 1)[0]
except LookupError:
    release = "0.0.0"

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
    "myst_parser",
    "nbsphinx",
    "sphinx_design",
]

# Don't execute notebooks during build
nbsphinx_execute = "never"

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None).replace("\\", "/") %}

.. raw:: html

    <div style="float: right; margin-bottom: 1em;">
        <a href="https://github.com/Akseli-Ilmanen/ethograph/tree/main/examples"
           style="text-decoration: none; margin-right: 0.5em;">
            View on GitHub
        </a>
        &middot;
        <a href="{{ docname.split('/')[-1] }}.ipynb" download
           style="text-decoration: none;">
            ⬇ Download notebook
        </a>
    </div>
"""

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
        "text": f"{project} v{release}",
    },
}

# GitHub pages
github_user = "Akseli-Ilmanen"
html_baseurl = f"https://{github_user}.github.io/{project}"
sitemap_url_scheme = "{link}"


# -- Copy tutorial notebooks into examples/ at build time --------------------

import shutil
from pathlib import Path


def setup(app):
    src = Path(app.srcdir).resolve().parent.parent / "examples"
    dst = Path(app.srcdir) / "examples"
    dst.mkdir(exist_ok=True)
    for nb in src.glob("*.ipynb"):
        # Fix double extensions like foo.ipynb.ipynb -> foo.ipynb
        name = nb.name
        while name.endswith(".ipynb.ipynb"):
            name = name[: -len(".ipynb")]
        target = dst / name
        if not target.exists() or nb.stat().st_mtime > target.stat().st_mtime:
            shutil.copy2(nb, target)
    # Also copy the assets folder if it exists
    assets_src = src / "assets"
    assets_dst = dst / "assets"
    if assets_src.is_dir():
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(assets_src, assets_dst)
