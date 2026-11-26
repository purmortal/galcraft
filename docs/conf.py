"""Build configuration for the GalCraft documentation website."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "GalCraft"
author = "Zixian Wang and contributors"
try:
    from GalCraft._version import __version__ as release
except ImportError:
    release = "development"

extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.autosummary", "sphinx.ext.intersphinx"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build"]
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["numpy", "scipy", "astropy", "ebf", "ephem", "spectres", "matplotlib"]
napoleon_numpy_docstring = True
napoleon_google_docstring = False
myst_heading_anchors = 3
html_theme = "pydata_sphinx_theme"
html_title = "GalCraft Documentation"
html_theme_options = {"github_url": "https://github.com/purmortal/galcraft", "show_toc_level": 2}
html_static_path = []
html_context = {"default_mode": "light"}
intersphinx_mapping = {}
