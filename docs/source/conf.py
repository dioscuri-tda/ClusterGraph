import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

print(os.getcwd())


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ClusterGraph"
copyright = "2023, Mathis Hallier"
author = "Mathis Hallier"
release = "0.3.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_rtd_theme",
]

# Enable MathJax for rendering LaTeX
# mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []

autoclass_content = "both"

autodoc_mock_imports = [
    "pandas",
    "networkx",
    "numba",
    "matplotlib",
    "bokeh",
    "sklearn",
    "scipy",
    "tqdm",
    "ot",
    "ipywidgets",
    "IPython",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = 'classic'

html_static_path = ["_static"]

debug = True
