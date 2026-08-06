"""Sphinx configuration for the pyJac documentation."""

from pyjac import __version__

project = 'pyJac'
copyright = '2016-2026, Kyle E. Niemeyer and Nicholas J. Curtis'
author = 'Kyle E. Niemeyer, Nicholas J. Curtis'

version = '.'.join(__version__.split('.')[:2])
release = __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# `any` lets single-backtick references resolve against any domain, which is
# what the existing prose assumes.
default_role = 'any'

# -- autodoc ---------------------------------------------------------------

autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# -- napoleon --------------------------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- intersphinx -----------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'cantera': ('https://cantera.org/stable/', None),
}

# -- HTML output -----------------------------------------------------------

html_theme = 'furo'
html_title = f'pyJac {release}'

html_theme_options = {
    'source_repository': 'https://github.com/SLACKHA/pyJac/',
    'source_branch': 'main',
    'source_directory': 'docs/',
}

# -- LaTeX output ----------------------------------------------------------

latex_documents = [
    ('index', 'pyJac.tex', 'pyJac Documentation', author, 'manual'),
]
