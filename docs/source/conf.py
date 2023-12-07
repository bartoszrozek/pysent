# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../pysent/overall_annotators"))
sys.path.insert(0, os.path.abspath("../../pysent/aspect_annotators/"))
sys.path.insert(0, os.path.abspath("../../pysent/aspect_annotators/extractors"))
sys.path.insert(0, os.path.abspath("../../pysent/aspect_annotators/classifiers"))
sys.path.insert(0, os.path.abspath("../../pysent/aspect_annotators/extrassifiers"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pysent"
copyright = "2023, BAMK"
author = "BAMK"
release = "0.0.0.900"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.autosummary"]
autosummary_generate = False
autoclass_content = "init"
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
