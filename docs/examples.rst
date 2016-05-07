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

    python -m pyjac.functional_tester --mech data/h2o2.cti --lang c \
    --pasr_output h2_pasr_output.npy

**Alternatively**, you can perform the test using provided example data::

    python -m pyjac.functional_tester --mech data/h2o2.cti --lang c \
    --pasr_output data/h2_pasr_output.npy

Detailed error statistics are saved in ``error_arrays.npz``, and overall results
printed to screen.

===================
Performance testing
===================

Performance testing (i.e. testing the speed of the various pyJac libraries against TChem)
is given (typically two arguements), a base directory and a number of OMP threads to use.
The program scans for all subdirectories in the base directory, looking for the following keys:

    * A cantera mechanism (ending with .cti)
    * A chemkin mechanism of the same name (ending with .dat)
    * An (optional) chemkin thermo file (with 'therm' in filename) if required
    * If the thermo file is not specified, the chemkin mechanism is assumed to contain the thermo data.
    * Thermochemical state data (as generated previously using the PaSR module) files ending with *.npy

Note that all *.npy files in a directory will be used for testing purposes

The performance tester can be called using:

python -m pyjac.performance_tester -w data/ -nt 12
