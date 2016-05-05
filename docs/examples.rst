Examples
########

Some example input files are included in the ``examples`` directory.

After installing pyJac, or in the package directory, you can see all of the
usage options with a standard ``--help`` or ``-h`` option::

    python -m pyjac --help

========================
Jacobian file generation
========================

To generate the Jacobian source files for a hydrogen-air system in C (without
any cache optimization)::

    python -m pyjac --lang c --input examples/h2o2.inp -nco

CUDA source code can be generated similarly::

    python -m pyjac --lang cuda --input examples/h2o2.inp -nco

==================
Functional testing
==================
