Examples
########

Some example input files are included in the ``data`` directory.

After installing pyJac, or in the package directory, you can see all of the
usage options with a standard ``--help`` or ``-h`` option::

    python -m pyjac --help

========================
Jacobian file generation
========================

To generate the Jacobian source files for a hydrogen-air system in C (without
any cache optimization)::

    python -m pyjac --lang c --input data/h2o2.inp

CUDA source code can be generated similarly::

    python -m pyjac --lang cuda --input data/h2o2.inp

==================
Functional testing
==================

Functional testing (i.e., testing whether pyJac gives the correct results)
requires thermochemical state data. This can be generated using the built-in
partially stirred reactor (PaSR) module::

    python -m pyjac.functional_tester.partially_stirred_reactor \
    --mech data/h2o2.cti --input data/pasr_input.yaml \
    --output h2_pasr_output.npy

Then, functional testing using this data can be performed via::

    python3 -m pyjac.functional_tester --mech data/h2o2.cti --lang c \
    --pasr_output h2_pasr_output.npy

**Alternatively**, you can perform the test using provided example data::

    python3 -m pyjac.functional_tester --mech data/h2o2.cti --lang c \
    --pasr_output data/h2_pasr_output.npy

Detailed error statistics are saved in ``error_arrays.npz``, and overall results
printed to screen.
