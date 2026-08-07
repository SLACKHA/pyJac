Validation
##########

pyJac generates source code, so two questions have to be answered separately:
does the generator still emit what it used to, and is what it emits correct?

The golden-output tests answer the first by comparing generated C and CUDA
byte-for-byte against recorded fixtures. They are what makes refactoring safe,
but they are silent on correctness: a wrong coefficient is just as stable as a
right one. The tests described here answer the second, by compiling the
generated code, evaluating it, and comparing against `Cantera`_.

.. _Cantera: https://cantera.org

.. _validation_reference:

===========================
Choosing a reference
===========================

The original pyJac paper :cite:`Niemeyer2017` validated the generated Jacobian
against automatic differentiation, using `Adept`_ to differentiate pyJac's own
generated source. That remains a sound check on the differentiation itself, and
pyJac can still build the ``adjacob`` variant. It has one structural limit:
because it differentiates pyJac's right-hand side, it cannot detect an error in
that right-hand side. A mistake in a rate expression appears identically on both
sides of the comparison and cancels.

Validating against Cantera removes that limit. Cantera is an independent
implementation of the same chemistry, so the comparison covers the rate
expressions and the thermodynamics as well as the derivatives. It also removes
the C++ Adept dependency from the validation path.

.. _Adept: https://www.met.reading.ac.uk/clouds/adept/

The same paper found that finite differencing is not accurate enough to
validate the full Jacobian. That finding still holds, and nothing below uses a
finite difference as the primary reference for the matrix. Finite differences
do appear, in two narrow roles where their accuracy is established rather than
assumed; :ref:`validation_finite_difference` sets out the reasoning.

.. _validation_rates:

=================
Validating rates
=================

``test/test_rate_validation.py`` generates, compiles and wraps a mechanism,
then compares three quantities against Cantera at several thermochemical
states:

* species concentrations,
* forward rates of progress, per reaction,
* net production rates.

Per-reaction forward rates matter more than the net rates: a net production
rate sums contributions that can cancel, so an error in one reaction can hide
in the total. Comparing reaction by reaction removes that shelter.

pyJac keeps the pressure-modification factor separate, while Cantera folds it
into the rate of progress, so it is applied before comparing. Which reactions
carry that factor has to be taken from pyJac's reading of the mechanism, not
Cantera's: for a reaction written with an explicit collider such as
``H+O2+O2<=>HO2+O2``, Cantera stores a three-body reaction while the Chemkin
parser folds the collider into the rate expression. Sizing the array from
Cantera's count leaves trailing zeros that silently zero out real rates.

Two composition details are load-bearing. Every species is given a non-zero
mole fraction, because most reactions need a radical and seeding only stable
species leaves them with a zero-concentration reactant --- every rate then
comes out zero and the comparison passes while proving nothing. And the
mechanisms are chosen to cover every supported reaction form between them:
``rxn_types.inp`` supplies SRI, PLOG and Chebyshev, which GRI-Mech 3.0 does not
contain, while GRI-Mech supplies Lindemann falloff and scale.

Agreement is at the level of accumulated floating-point difference,
roughly 1e-9, against a tolerance of 1e-8. pyJac evaluates its own
thermodynamic polynomials and orders the arithmetic differently from Cantera,
which is enough to account for it.

.. _validation_jacobian:

=======================
Validating the Jacobian
=======================

pyJac's Jacobian is the derivative of :math:`\Phi = \{T, Y_1, \dotsc,
Y_{N_{\text{sp}}-1}\}` with respect to itself at constant pressure, with one
species eliminated through :math:`Y_{N} = 1 - \sum_k Y_k`. Cantera exposes
kinetics derivatives with respect to temperature and species *concentrations*,
so ``test/jacobian_reference.py`` converts them with an explicit chain rule for
the constant-pressure constraint: raising one mass fraction lowers the
eliminated species by the same amount, which shifts the mean molecular weight
and therefore the density and every concentration.

Three details decide whether the comparison is meaningful at all.

**The eliminated species must match.** pyJac does not eliminate whichever
species happens to be listed last; it prefers an inert bulk species, N2 by
default, so that the closure error is absorbed by a species that barely
participates. For GRI-Mech 3.0 that is N2 at index 47, not the last-listed
species at index 52. Eliminating a different species produces a genuinely
different matrix, not a permutation of the same one, so the reference takes
both the eliminated species and the ordering from pyJac.

**Tolerances are scaled by the matrix, not by each entry.** Jacobian entries
pass through zero, and a relative comparison against a nearly cancelling entry
reports a large error for a difference that is negligible in context. Measured
elementwise, agreement looks like 1e-7; measured against the largest entry of
the matrix, it is 2e-10. The latter is the honest number.

**The temperature self-derivative needs its own check.**
:math:`\partial \dot{T} / \partial T` sums a contribution from every species,
including the eliminated one, whose running total is accumulated separately
from the matrix. An error confined to that accumulation moves exactly one entry
out of :math:`N_{\text{sp}}^2` and is invisible in any aggregate measure. This
is not hypothetical: it is how the defect described in
:ref:`validation_changes` was found and how a regression would be caught.

.. _validation_finite_difference:

=================================
Where finite differences are used
=================================

Two uses survive the objection in :cite:`Niemeyer2017`, both narrower than
differencing the full matrix.

The first checks the chain rule in ``jacobian_reference`` against a central
difference of the same right-hand side. Both sides evaluate the same
Cantera-based function, so the comparison tests the calculus rather than the
chemistry, and it is applied to the median entry rather than the worst.

The second supplies the reference for
:math:`\partial \dot{T} / \partial T`, where a Richardson-extrapolated central
difference is *more* accurate than Cantera's own derivative. Cantera has no
analytic temperature derivative for PLOG or Chebyshev rates and finite
differences them internally, which caps the analytic reference at a few parts
in :math:`10^7` for any mechanism carrying those forms. Tightening Cantera's
internal step does not fix it: the error is U-shaped in step size and its
minimum falls in a different place for each state.

Differencing directly is sound here for reasons that do not extend to the
species block. This is a single well-scaled scalar derivative of a smooth
function, not the stiff, ill-conditioned block that defeated finite differences
in the original study. Central differencing is second order, so evaluating at
two step sizes and extrapolating cancels the leading truncation term. The
result agrees with pyJac to 2--3e-10, the same floating-point floor the rest of
the matrix reaches, and two independent pairs of step sizes agree with each
other to between 1e-12 and 1e-10 --- so the accuracy is a property of the
method, not of a fortunate choice of step.

The distinction worth carrying forward is that a finite difference is
unsuitable as a *blanket* reference for an analytical Jacobian, not that it is
unusable. Where a single entry is smooth and well scaled, and where the step
independence is demonstrated rather than assumed, extrapolated differencing is
the most accurate reference available.

.. _validation_sensitivity:

=====================
Sensitivity
=====================

A validation test is only worth its tolerance if it would fail on a real
defect, so each threshold was checked against an injected error:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Injected error
     - Reported by
     - Tolerance
   * - One reaction rate scaled by :math:`1 + 10^{-6}`
     - 1.0e-6 (forward rates), 1.3e-7 (net rates)
     - 1e-8
   * - One NASA polynomial divisor, :math:`a_2/3 \rightarrow a_2/4`
     - 16% error against Cantera's thermodynamic data
     - 1e-10
   * - Eliminated species' derivative accumulation
     - 7.4e-3 on GRI-Mech 3.0 at 1800 K
     - 1e-8

.. _validation_changes:

=========================================
Changes affecting computed results
=========================================

Most of the modernization left generated source byte-identical, verified by the
golden fixtures. The exceptions below change computed values, so results from
this version will differ from 1.0.6. See ``CHANGELOG.md`` for the full record.

**Corrected defects.**

*The eliminated species' contribution to* :math:`\partial \dot{T} / \partial T`
*was overwritten rather than accumulated.* That species has no Jacobian entry,
so its running total is kept in a scratch variable, and the choice between
assignment and accumulation consulted a flag that was never set for it. Each
contribution overwrote the previous one and only the last survived --- 26
contributions collapsed to one for GRI-Mech 3.0. The error reached 0.74% at
1800 K while every other entry of the matrix remained correct to 2e-10, and
pyJac's :math:`\dot{T}` itself was correct throughout: the defect was confined
to the derivative. Only mechanisms whose eliminated species reacts are
affected, which with the default choice of N2 means any mechanism with NOx
chemistry. Mechanisms closed on a genuinely inert species such as Ar were
always correct, which is why the golden fixtures never registered it.

*A Chebyshev reaction with too few coefficients read past its fit.* With two or
fewer temperature coefficients the Jacobian read ``dot_prod`` out of bounds;
where another reaction had set a larger array size the read was in bounds but
returned that reaction's value, silently producing a wrong temperature
derivative. The rate expression had the same flaw for a fit with a single
temperature or pressure coefficient, which Cantera accepts.

*A negative pre-exponential with a negative whole-number temperature exponent
and no activation energy lost its temperature dependence entirely*, because the
repeated-multiplication path iterated over an empty range. No mechanism in the
test suite reaches this combination, so it is latent rather than observed.

**Deliberate changes.**

Atomic weights now come from Cantera rather than a hardcoded table from an
older IUPAC revision, shifting molecular weights and everything derived from
them by up to 6e-5 relative. The physical constants likewise come from Cantera
(2018 CODATA), shifting the gas constant by 6e-8 relative. Both were confirmed
to be the sole cause of their golden-output changes by regenerating with the
previous values and diffing.

==========
References
==========

.. bibliography::
   :filter: docname in docnames
