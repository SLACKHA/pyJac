Installation
############

pyJac requires Python 3.10 or newer. Install it with ``pip``:

.. code-block:: bash

   pip install pyjac

That pulls in the required dependencies: `NumPy`_, `PyYAML`_, and `Cantera`_.
Cantera is required rather than optional, since pyJac uses it to read
mechanisms in Cantera's YAML format and to source atomic weights and physical
constants.

To install from a checkout instead:

.. code-block:: bash

   pip install .

==================
Optional features
==================

Some parts of pyJac need extra packages, installed as extras:

.. code-block:: bash

   pip install 'pyjac[pywrap]'      # build the Python wrapper around generated code
   pip install 'pyjac[cache-opt]'   # experimental cache-optimizing reordering

``pywrap`` installs `Cython`_ and setuptools, needed to compile the generated
source into a module Python can import. ``cache-opt`` installs `bitarray`_,
used only by the ``--cache-optimizer`` option.

Compiling the generated source needs a C compiler, and for the CUDA backend the
`CUDA Toolkit`_. Neither is installed by pip.

=======
Testing
=======

To run the test suite from a checkout:

.. code-block:: bash

   pip install -e '.[test,pywrap]'
   pytest

Tests that need a C compiler skip themselves when none is available. The
cache-optimizer tests are marked ``slow`` and can be excluded:

.. code-block:: bash

   pytest -m "not slow"

Please let us know if you run into trouble installing pyJac.

.. _NumPy: https://numpy.org
.. _PyYAML: https://pyyaml.org
.. _Cantera: https://cantera.org
.. _Cython: https://cython.org
.. _bitarray: https://github.com/ilanschnell/bitarray
.. _CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
