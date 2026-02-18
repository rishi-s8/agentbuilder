# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# -- Project information -----------------------------------------------------
project = "AgentBuilder"
copyright = "2025, Rishi Sharma"
author = "Rishi Sharma"
release = "0.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Napoleon settings (Google-style docstrings) -----------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True

# -- Autodoc settings --------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Suppress warnings from Sphinx documenting dataclass fields twice
# (as class attribute and __init__ parameter). Harmless but noisy.
suppress_warnings = ["duplicate"]

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "AgentBuilder"

# -- Copybutton settings -----------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
