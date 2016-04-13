.. pyJac documentation master file, created by
   sphinx-quickstart on Tue Apr 12 16:33:22 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyJac
=====

**pyJac** is a Python package that generates source code used to analytically calculate chemical kinetics Jacobian matrices, customized for a particular model/reaction mechanism.

**pyJac** welcomes your feedback and/or contributions. It relies heavily on the `numpy`_ libraries for core functionality, and other libraries including the `Cython`_ language and `Cantera`_ for functional and performance testing.

.. _numpy: http://numpy.org
.. _Cython: http://cython.org
.. _Cantera: http://www.cantera.org

Documentation
-------------

.. toctree::
   :maxdepth: 1

   overview
   installing
   examples
   src/index


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

Up-to-date information about citing **pyJac** can be found within the `citation`_
file.

.. _citation: https://github.com/kyleniemeyer/pyJac/blob/master/CITATION.md

See also
--------

- To-be-published `pyJac v1 paper`_
- Kyle Niemeyer, Nick Curtis, and Chih-Jen Sung's `WSSCI Fall 2015 paper`_ introducing
- The associated `WSSCI Fall 2015 slides`_

.. _pyJac v1 paper: http://kyleniemeyer.github.io/pyJac-paper/
.. _WSSCI Fall 2015 paper: https://dx.doi.org/10.6084/m9.figshare.2075515.v1
.. _WSSCI Fall 2015 slides: http://www.slideshare.net/kyleniemeyer/initial-investigation-of-pyjac-an-analytical-jacobian-generator-for-chemical-kinetics

Get in touch
------------

- Please report bugs, suggest feature ideas, and browse the source code `on GitHub`_.
- There, new contributors can also find `a guide to contributing`_.
- You can also contact Kyle `on Twitter`_.

.. _on GitHub: https://github.com/kyleniemeyer/pyJac
.. _a guide to contributing: https://github.com/kyleniemeyer/pyJac/blob/master/CONTRIBUTING.md
.. _on Twitter: http://twitter.com/kyle_niemeyer


License
-------

**pyJac** is available under the open-source `MIT License`__.

__ https://raw.githubusercontent.com/kyleniemeyer/pyJac/master/LICENSE
