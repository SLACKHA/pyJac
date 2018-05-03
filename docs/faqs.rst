FAQs
####

====
FAQs
====

.. _ordering:

What state vector should I pass into pyJac?
===========================================

As described in :ref:`state_vec`, pyJac expectes state vector :math:`\Phi` that
consists of:

.. math::
    \Phi = \left \lbrace T, Y_1, Y_2, \dotsc,
    Y_{N_{\text{sp}} - 1} \right \rbrace^{\text{T}}

where :math:`T` is the temperature, and :math:`Y_i` is the mass fraction of
species *i*.  Because pyJac is formulated to ensure strict mass conservation,
the mass fraction of the last species in the model :math:`Y_{N_{sp}}` is
determined as:

.. math::
    Y_{N_{\text{sp}}} = 1 - \sum_{k=1}^{N_{\text{sp}} - 1} Y_k

This means that supplying the last species mass fraction is *optional* in
all the pyJac forms.

.. _cantera_comp:

What does this mean for comparison to say, Cantera?
===================================================

The main *gotcha* here, is that pyJac automatically picks a species
to place at the end of the model.  This choice defaults to the first
of N\ :sub:`2`, Ar or He found in the model.  The last species can
also be specified by the user, with the "-ls" command line option.

This species (say, N\ :sub:`2`) is taken out of its' original position in
the model and placed at the end of the model.  If you want to compare results
to Cantera, this can be confusing; say you have a Solution object "gas", and
have selected N\ :sub:`2` as the last species.  The Solution object can be
updated as:

.. code-block:: python

    import cantera as ct
    #create gas from original mechanism file `mech.cti`
    gas = ct.Solution('mech.cti')
    #reorder the gas to match pyJac
    n2_ind = gas.species_index('N2')
    specs = gas.species()[:]
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
            species=specs[:n2_ind] + specs[n2_ind + 1:] + [specs[n2_ind]],
            reactions=gas.reactions())

.. _jac_vals:

What is in the Jacobian?
========================

As described in :ref:`jacobian_formulation`, the Jacobian consists of
temperature and mass fraction derivatives:

.. math::
    \mathcal{J}_{i, j} = \frac{\partial \dot{\Phi_i}}{\partial \Phi_j}

This translates to a Jacobian matrix that looks like:

.. math::
    \left[
    \begin{array}{cccc}
        \frac{\partial \dot{T}}{\partial T} & \frac{\partial \dot{T}}{\partial Y_1} & \ldots & \frac{\partial \dot{T}}{\partial Y_{N_{\text{sp}} - 1}} \\
        \frac{\partial \dot{Y_1}}{\partial T} & \frac{\partial \dot{Y_1}}{\partial Y_1} & \ldots & \frac{\partial \dot{Y_1}}{\partial Y_{N_{\text{sp}} - 1}} \\
        \vdots & & \ddots & \vdots \\
        \frac{\partial \dot{Y}_{N_{\text{sp}} - 1}}{\partial T} & \frac{\partial \dot{Y}_{N_{\text{sp}} - 1}}{\partial Y_1} & \ldots & \frac{\partial \dot{Y}_{N_{\text{sp}} - 1}}{\partial Y_{N_{\text{sp}} - 1}}
    \end{array}
    \right]

In code, the Jacobian is flattened in column-major (Fortran) order:

.. math::
    \vec{\mathcal{J}} = \left\{ \frac{\partial \dot{T}}{\partial T}, \frac{\partial \dot{Y_1}}{\partial T}, \ldots \frac{\partial \dot{Y}_{N_{\text{sp}} - 1}}{\partial T}, \ldots, \frac{\partial \dot{T}}{\partial Y_1}, \frac{\partial \dot{Y_1}}{\partial Y_1} \ldots, \frac{\partial \dot{T}}{\partial Y_{N_{\text{sp}} - 1}}, \frac{\partial \dot{Y_1}}{\partial Y_{N_{\text{sp}} - 1}} \ldots \frac{\partial \dot{Y}_{N_{\text{sp}} - 1}}{\partial Y_{N_{\text{sp}} - 1}} \right\}

The resulting Jacobian is of length :math:`N_{\text{sp}} * N_{\text{sp}}`.
Note that the ordering issues disucssed in :ref:`ordering` apply here as well.

.. _paper: https://Niemeyer-Research-Group.github.io/pyJac-paper/

.. _units:

What units does pyJac use?
=========================

pyJac uses a default of kilogram, meters and seconds for its unit system.
This means that pressures are in pascals, temperature in Kelvin, and time in seconds.

In chemical kinetic mechanism files, users should utilize the default units for the
parser (Chemkin/Cantera) they are using.

The only known exception to this, is the *-ic* or *--initial-conditions*
command line flag for code-generation, where pressure is specified in atmospheres for
convenience.


.. _param_names:

What is the difference between "t" and "T" in the Python interface?
===================================================================

Most ODE solvers allow you to pass the current system time (or independent
variable) to the RHS / Jacobian functions. We conform to this standard, as it
allows for interfacing pyJac with other solvers, e.g. CVODEs.  Hence the
:func:`dydt` and :func:`eval_jacob` functions have a "t" parameter for the
current system time, although it is not used.

However, several other functions (e.g., the reaction / species rates, etc.)
require the temperature.  Internally, we separate the temperature out from the
state vector for easier comprehension; however this makes life confusing
(to say the least), because these functions have a "T" representing the
temperature.

We know this is not ideal, and it will be fixed in V2.

.. _data_passing:

How do I pass numpy arrays to the C/CUDA pyJac functions from Python?
=====================================================================

In addition to the ordering considerations discussed above (:ref:`ordering`),
care must be taken to ensure correctness when using numpy with pyJac's
Python interface.

First, it is important to realize the difference between a numpy view_ and the
actual data order inside of numpy_.  A view is a simple user interface for
indexing numpy arrays but does *not* change the actual order in which the data
is
stored in memory. E.g.:

.. code-block:: python

    arr = arr.T

Returns a *view* of the transpose of "arr", but does not change the data order
in memory.
To do that, you would need a copy (among other ways):

.. code-block:: python

    arr = arr.T.copy()

This is important for pyJac, as once you pass a 2-D array to the underlying
C/CUDA code, it will be written / read from in C-contiguous order
(again, see `numpy`__). If you pass a non C-contiguous array to pyJac, you will
likely have difficulty intepreting the output.

.. _view: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.view.html
.. _numpy: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html
__ numpy_

Finally, we note that the CUDA functions expects 2-D arrays to be ordered such
that e.g., the temperatures for all the different states are contiguous in
memory, followed by the mass fractions, etc.  For :math:`N_{\text{state}}`
independent thermo-chemical states, this translates to:

.. math::
    T_{0}, T_{1}, \ldots T_{N_{\text{state}}}, Y_{0, 0}, Y_{0, 1}, \ldots
    Y_{0, N_{\text{state}}}, Y_{1, 0}, \ldots

where :math:`\Phi_{i, j}` corresponds to the *i*-th entry in the state vector, for the *j*-th stherm-chemical state.
