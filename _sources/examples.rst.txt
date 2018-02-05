Examples
########

Some example input files are included in the ``data`` directory.

After installing pyJac, or in the package directory, you can see all of the
usage options with a standard ``--help`` or ``-h`` option:

.. code-block:: bash

    python -m pyjac --help

========================
Jacobian file generation
========================

To generate the Jacobian source files for a hydrogen-air system in C (without
any cache optimization):

.. code-block:: bash

    python -m pyjac --lang c --input data/h2o2.inp

CUDA source code can be generated similarly:

.. code-block:: bash

    python -m pyjac --lang cuda --input data/h2o2.inp

==================
Functional testing
==================

Functional testing (i.e., testing whether pyJac gives the correct results)
requires thermochemical state data. This can be generated using the built-in
partially stirred reactor (PaSR) module:

.. code-block:: bash

    python -m pyjac.functional_tester.partially_stirred_reactor \
    --mech data/h2o2.cti --input data/pasr_input.yaml \
    --output h2_pasr_output.npy

Then, functional testing using this data can be performed via:

.. code-block:: bash

    python -m pyjac.functional_tester --mech data/h2o2.cti --lang c \
    --pasr_output h2_pasr_output.npy

**Alternatively**, you can perform the test using provided example data:

.. code-block:: bash

    python -m pyjac.functional_tester --mech data/h2o2.cti --lang c \
    --pasr_output data/h2_pasr_output.npy

Detailed error statistics are saved in ``error_arrays.npz``, and overall results
printed to screen.

===================
Performance testing
===================

The performance of the analytical Jacobian matrix evaluation using pyJac can be
tested against finite differencing and TChem. With the appropriate environment
established, this test can be performed by giving only two arguments: a base
directory and a number of OpenMP threads to use. The program scans for
subdirectories in the base directory, looking for the following keys:

 * A Cantera mechanism (ending with .cti)
 * A Chemkin mechanism of the same name (ending with .dat)
 * An (optional) Chemkin thermodynamic file (with "therm" in filename)
   if required. If the thermo file is not specified, the mechanism is assumed
   to contain the thermo data.
 * Thermochemical state data (as generated previously using the PaSR module)
   files ending with ``*.npy``

Note that all ``*.npy`` files in a directory will be used for testing purposes.

The performance tester can be called using:

.. code-block:: bash

    python -m pyjac.performance_tester -w data/

==================
Library Generation
==================

pyJac also has the ability to generate shared / static libraries for
linkage to external programs.  This functionality is available via the
:py:mod:`pyjac.libgen` submodule, and requires a gcc/nvcc installation available
on the path.  It can be called as:

.. code-block:: bash

    python -m pyjac.libgen --source_dir /path/to/generated/pyjac/output \
           --lang cuda --static

Note that for linkage into an external program, CUDA requires use of a
static library.

=========================
Python Wrapper Generation
=========================

In addition to the library generation described above, pyJac can directly
generate a python wrapper for chemical source term / Jacobian evaluation
(among others) directly from python.  This functionality can be called
via (e.g.,):

.. code-block:: bash

    python -m pyjac.pywrap --source_dir /path/to/generated/pyjac/output \
           --lang cuda

For details of the functions included in the python wrapper, look at the
.pyx files in :py:mod:`pyjac.pywrap`, or the calls in
:py:class:`pyjac.functional_tester.test.cpyjac_evaluator`

==========================
Using the Python Interface
==========================

Once generated, the python wrapper can be imported from a python script in
the same directory, e.g.:

.. code-block:: python

    import pyjacob

Then the :func:`dydt` or :func:`eval_jacob` functions can be called, e.g. for
the GRI-Mech 3.0 model as:

.. code-block:: python

    import pyjacob
    import cantera as ct
    import numpy as np

    #create gas from original mechanism file gri30.cti
    gas = ct.Solution('gri30.cti')
    #reorder the gas to match pyJac
    n2_ind = gas.species_index('N2')
    specs = gas.species()[:]
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
            species=specs[:n2_ind] + specs[n2_ind + 1:] + [specs[n2_ind]],
            reactions=gas.reactions())

    #set the gas state
    T = 1000
    P = ct.one_atm
    gas.TPY = T, P, "CH4:1.0, O2:2, N2:7.52"

    #setup the state vector
    y = np.zeros(gas.n_species)
    y[0] = T
    y[1:] = gas.Y[:-1]

    #create a dydt vector
    dydt = np.zeros_like(y)
    pyjacob.py_dydt(0, P, y, dydt)

    #create a jacobian vector
    jac = np.zeros(gas.n_species * gas.n_species)

    #evaluate the Jacobian
    pyjacob.py_eval_jacobian(0, P, y, jac)

The above uses the state vector discussed in (:ref:`ordering`), as well as the
reordering in (:ref:`cantera_comp`) to enable direct comparison to Cantera.
Also note that we can pass a dummy time of 0, as explained in
(:ref:`param_names`).

The CUDA interface is less modular, and currently only supports evaluating the
Jacobian directly (which in turn populates the other values).  For example,
if we have 1000 states to evaluate:

.. code-block:: python

    import cu_pyjacob
    import cantera as ct
    import numpy as np

    #create gas from original mechanism file gri30.cti
    gas = ct.Solution('gri30.cti')
    #reorder the gas to match pyJac
    n2_ind = gas.species_index('N2')
    specs = gas.species()[:]
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
            species=specs[:n2_ind] + specs[n2_ind + 1:] + [specs[n2_ind]],
            reactions=gas.reactions())

    N_state = 1000
    #setup the state vectors
    y = np.zeros((N_state, gas.n_species))
    pres = np.zeros(N_state)

    #populate with dummy data
    for i in range(N_state):
        #use cantera to normalize mass fractions
        gas.TPY = 2400 * np.random.rand(), 25 * ct.one_atm * np.random.rand(), \
            np.random.random(gas.n_species)

        #set state
        y[i, 0] = gas.T
        y[i, 1:] = gas.Y[:-1]
        pres[i] = gas.P

    #flatten
    y = y.flatten(order='f').astype(np.dtype('d'), order='c')

    #find # of reversible reactions
    num_rev = np.array([rxn.reversible
                                for rxn in gas.reactions()]
                                ).sum()
    def __is_pdep(rxn):
        return (isinstance(rxn, ct.ThreeBodyReaction) or
            isinstance(rxn, ct.FalloffReaction) or
            isinstance(rxn, ct.ChemicallyActivatedReaction)
            )

    num_pdep = np.array([__is_pdep(rxn)
                             for rxn in gas.reactions()]
                             ).sum()

    #create other arrays
    def __czeros(shape):
        #Return array of zeros in C ordering.
        arr = np.zeros(shape)
        return arr.flatten(order='c')

    concs = __czeros((N_state, gas.n_species))
    fwd_rates = __czeros((N_state, gas.n_reactions))
    rev_rates = __czeros((N_state, num_rev))
    pres_mod = __czeros((N_state, num_pdep))
    spec_rates = __czeros((N_state, gas.n_species))
    dydt = __czeros((N_state, gas.n_species))
    jac = __czeros((N_state, gas.n_species * gas.n_species))

    #intialize and get padding
    N_pad = cu_pyjacob.py_cuinit(N_state)

    #call jacobian function
    cu_pyjacob.py_cujac(N_state, N_pad, pres, y, concs,
                            fwd_rates, rev_rates, pres_mod,
                            spec_rates, dydt, jac
                            )

    #finally reshape arrays for sensible comparison
    dydt = dydt.reshape((N_state, gas.n_species), order='f').astype(
        np.dtype('d'), order='c')
    jac = jac.reshape((N_state, gas.n_species * gas.n_species),
        order='f').astype(np.dtype('d'), order='c')

Note that this uses the ordering discussed in (:ref:`data_passing`), while the
Jacobian values are explained in (:ref:`jac_vals`).