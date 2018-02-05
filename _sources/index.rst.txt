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

.. _CITATION.md: https://github.com/slackha/pyJac/blob/master/CITATION.md

See also
--------

- The published `pyJac v1 paper`_
- Kyle Niemeyer, Nick Curtis, and Chih-Jen Sung's `WSSCI Fall 2015 paper`_ introducing
- The associated `WSSCI Fall 2015 slides`_

.. _pyJac v1 paper: https://arxiv.org/abs/1605.03262
.. _WSSCI Fall 2015 paper: https://dx.doi.org/10.6084/m9.figshare.2075515.v1
.. _WSSCI Fall 2015 slides: http://www.slideshare.net/kyleniemeyer/initial-investigation-of-pyjac-an-analytical-jacobian-generator-for-chemical-kinetics

Get in touch
------------

- Please report bugs, suggest feature ideas, and browse the source code `on GitHub`_.
- There, new contributors can also find `a guide to contributing`_.
- Additionally, you may join our `user group`_ for further support and to be notified of new releases, features, etc.
- You can also contact Kyle `on Twitter`_.

.. _on GitHub: https://github.com/slackha/pyJac
.. _a guide to contributing: https://github.com/slackha/pyJac/blob/master/CONTRIBUTING.md
.. _user group: https://groups.io/g/slackha-users
.. _on Twitter: http://twitter.com/kyleniemeyer


License
-------

**pyJac** is available under the open-source `MIT License`__.

__ https://raw.githubusercontent.com/kyleniemeyer/pyJac/master/LICENSE
