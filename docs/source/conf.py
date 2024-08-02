# -*- coding: utf-8 -*-

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import datetime

project = 'PAI-TorchAcc'
copyright = '2024, alibaba-cloud'
author = 'Alibaba Cloud'
release = '2.3.0'
copyright = f'{datetime.datetime.now().year}, {author}'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

source_suffix = ['.rst', '.md']

extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.intersphinx',
  'sphinx.ext.mathjax',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
]

add_module_names = False
autodoc_member_order = 'bysource'
napoleon_google_docstring = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# language = 'zh_CN'
latex_engine = "xelatex"
latex_elements = {
    'preamble': r'''
        \usepackage{xeCJK}
        \setCJKmainfont{Noto Sans CJK SC}
    ''',
}
