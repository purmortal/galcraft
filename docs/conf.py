
"""Build configuration for the GalCraft documentation website."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Project information
project = "GalCraft"
author = "Zixian Wang and contributors"

try:
    from GalCraft._version import __version__ as release
except ImportError:
    release = "development"

# Sphinx extensions
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"
exclude_patterns = ["_build"]

# API documentation
autodoc_typehints = "none"
autodoc_member_order = "bysource"

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "astropy",
    "ebf",
    "ephem",
    "spectres",
    "matplotlib",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

# MyST Markdown
myst_heading_anchors = 3

# HTML theme
html_theme = "pydata_sphinx_theme"
html_title = "GalCraft Documentation"

html_theme_options = {
    "github_url": "https://github.com/purmortal/galcraft",
    "show_toc_level": 2,

    # Collapsible left navigation
    "navigation_depth": 3,
    "show_nav_level": 2,
    "collapse_navigation": False,
}

# Show the left sidebar only in Usage
html_sidebars = {
    "**": [],
    "usage/index": ["sidebar-nav-bs.html"],
    "usage/*": ["sidebar-nav-bs.html"],
}

# Other HTML settings
html_static_path = []

html_context = {
    "default_mode": "light",
}

# Intersphinx
intersphinx_mapping = {}
