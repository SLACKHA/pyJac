.. pyJac documentation master file, created by
   sphinx-quickstart on Tue Apr 12 16:33:22 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyJac
=====

**pyJac** is a Python package that generates source code used to analytically
calculate chemical kinetics Jacobian matrices, customized for a particular
model/reaction mechanism.

**pyJac** welcomes your feedback and/or contributions. It relies heavily on
the `numpy`_ libraries for core functionality, and other libraries including
the `Cython`_ language and `Cantera`_ for functional and performance testing.

.. _numpy: http://numpy.org
.. _Cython: http://cython.org
.. _Cantera: http://www.cantera.org

Documentation
-------------

.. toctree::
   :maxdepth: 1

   overview
   faqs
   examples
   installing
   src/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

Up-to-date information about citing **pyJac** can be found within the
`CITATION.md`_ file.

.. _CITATION.md: https://github.com/SLACKHA/pyJac/blob/main/CITATION.md

See also
--------

pyJac is described in full, with derivations, validation and performance
results, by Niemeyer et al. :cite:`Niemeyer2017`. The published article is paywalled; the
accepted version is openly available as
`arXiv:1605.03262 <https://arxiv.org/abs/1605.03262>`_.

Get in touch
------------

- Please report bugs, suggest feature ideas, and browse the source code `on GitHub`_.
- There, new contributors can also find `a guide to contributing`_.

.. _on GitHub: https://github.com/SLACKHA/pyJac
.. _a guide to contributing: https://github.com/SLACKHA/pyJac/blob/main/CONTRIBUTING.md


License
-------

**pyJac** is available under the open-source `MIT License`__.

__ https://raw.githubusercontent.com/kyleniemeyer/pyJac/main/LICENSE

.. bibliography::
   :filter: docname in docnames
