Overview
########

pyJac creates the C or CUDA source code files necessary to evaluate the
analytical Jacobian matrix for a constant-pressure reacting system.

.. _state_vec:

============
State Vector
============

Briefly, a thermochemical state is described using a composition vector:

.. math::
    \Phi = \left \lbrace T, Y_1, Y_2, \dotsc,
    Y_{N_{\text{sp}} - 1} \right \rbrace^{\text{T}}

where *T* is the temperature, :math:`Y_i` are the mass fractions, and
:math:`N_{\text{sp}}` is the number of species in the model. The mass fraction
of the final species is determined through conservation of mass:

.. math::
    Y_{N_{\text{sp}}} = 1 - \sum_{k=1}^{N_{\text{sp}} - 1} Y_k

.. _jacobian_formulation:

====================
Jacobian Formulation
====================

The governing equations of chemical kinetics include ordinary differential
equations for the rate of change of temperature and the species' mass fractions:

.. math::
    f &= \frac{\partial \Phi}{\partial t} \\
      &= \left \lbrace \frac{\partial T}{\partial t},
      \frac{\partial Y_1}{\partial t}, \frac{\partial Y_2}{\partial t},
      \dotsc, \frac{\partial Y_{N_{\text{sp}} - 1}}{\partial t}
      \right \rbrace^{\text{T}}

where

.. math::
    \frac{\partial T}{\partial t} &= \frac{-1}{\rho c_p}
    \sum_{k=1}^{N_{\text{sp}}} h_k W_k \dot{\omega}_k \\
    \frac{\partial Y_k}{\partial t} &= \frac{1}{\rho} W_k
    \dot{\omega}_k \quad k = 1, \dotsc, N_{\text{sp}} - 1

where :math:`c_p` is the mass-averaged constant-pressure specific heat,
:math:`h_k` is the specific enthalpy of species *k*, and :math:`\dot{\omega}_k`
is the overall production rate of species *k*.

The Jacobian matrix is then filled by the partial derivaties
:math:`\partial f / \partial \Phi`, such that

.. math::
    \mathcal{J}_{i,j} = \frac{\partial f_i}{\partial \Phi_j}

More details can be found in the paper fully describing version 1.0.2 of pyJac:
https://Niemeyer-Research-Group.github.io/pyJac-paper/