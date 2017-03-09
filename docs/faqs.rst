FAQs
####

====
FAQs
====

.. _ordering:

What state vector should I pass into pyJac?
===========================================

As described in :ref:`state_vec`, pyJac expectes state vector `Y` that
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
of N:sub:`2`, Ar or He found in the model.  The last species can
also be specified by the user, with the `-ls` command line option.

This species (say, N:sub:`2`) is taken out of its' original position in
the model and placed at the end of the model.  If you want to compare results
to Cantera, this can be confusing; say you have a Solution object `gas`, and
have selected N:sub:`2` as the last species.  The Solution object can be updated
as:

    import cantera as ct
    #create gas from original mechanism file `mech.cti`
    gas = ct.Solution('mech.cti')
    #reorder the gas to match pyJac
    n2_ind = gas.species_index('N2')
    gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
            species=specs[:n2_ind] + specs[n2_ind + 1:] + [specs[n2_ind]],
            reactions=gas.reactions())

What is in the Jacobian?
========================

As described in :ref:`jacobian_formulation`, the Jacobian consists of
temperature and mass fraction derivatives.  The order (described in full in
the paper_) consists of total temperature derivative in index 0

.. math::
    \frac{\partial \dot{T}}{\partial T}

followed by the temperature derivative WRT mass fractions:

.. math::
    \frac{\partial \dot{T}}{\partial Y_j}

for :math:`j = 0\ldots N_{\text{sp}} - 1}`, in indicies
:math:`1\ldots N_{\text{sp}}}`.

Following this is the species mass fraction derivative WRT temperature:

.. math::
    \frac{\partial Y_0}{\partial T}

in index :math:`N_{\text{sp} + 1}` and the mass fraction derivatives
WRT the other mass fractions:
.. math::
    \frac{\partial Y_0}{\partial Y_j}

in indicies :math:`N_{\text{sp}} + 1 \ldots 2 N_{\text{sp}}`, etc.
Note that the ordering issues disucssed in :ref:ordering apply here as well.
The resulting Jacobian is of length :math:`N_{\text{sp}} * N_{\text{sp}}`.

.. _paper: https://Niemeyer-Research-Group.github.io/pyJac-paper/

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

    arr = arr.T

Returns a *view* of the transpose of `arr`, but does not change the data order
in memory.
To do that, you would need a copy (among other ways):

    arr = arr.T.copy()

This is important for pyJac, as once you pass a 2-D array to the underlying
C/CUDA code, it will be written / read from in C-contiguous order
(again, see ordering_). If you pass a non C-contiguous array to pyJac, you will
likely have difficulty intepreting the output.

Finally, we note that the CUDA functions expects 2-D arrays to be ordered such
that e.g., the temperatures for all the different states are contiguous in
memory, followed by the mass fractions, etc.  For :math:`N_{\text{state}}`
independent thermo-chemical states, this translates to:

.. :math::
    T_{0}, T_{1}, \ldots T_{N_{\text{state}}}, Y_{0, 0}, Y_{0, 1}, \ldots
    Y_{0, N_{\text{state}}}, Y_{1, 0}, \ldots

.. _view: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.view.html
.. _numpy: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html
.. _ordering: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flags.html
