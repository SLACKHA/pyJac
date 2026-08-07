"""Unit tests for the pure expression-building helpers.

These functions return C/CUDA source as strings, so they can be tested
directly, without generating or compiling a mechanism. Where possible the
tests evaluate the emitted expression and check the *value* rather than
asserting on its text: the point is that the generated arithmetic is right,
not that it is spelled a particular way. The golden fixtures already pin
spelling.
"""

import math
import types

import pytest

from pyjac.core.rate_subs import get_nasa_arrays, rxn_rate_const

#: Temperatures at which emitted rate expressions are checked.
TEMPERATURES = [300.0, 1000.0, 2500.0]


def evaluate(expression, temperature):
    """Evaluate an emitted rate expression at a temperature.

    The generated C is also valid Python once ``T``, ``logT`` and ``exp`` are
    supplied, which lets these tests check the arithmetic without a compiler.
    """
    return eval(  # noqa: S307 - evaluating this module's own generated output
        expression,
        {'__builtins__': {}},
        {'T': temperature, 'logT': math.log(temperature), 'exp': math.exp},
    )


def arrhenius(pre_exponential, exponent, activation, temperature):
    """The rate constant the emitted expression is supposed to represent."""
    return pre_exponential * temperature**exponent * math.exp(-activation / temperature)


# Coefficients chosen to reach every branch of rxn_rate_const: positive and
# negative A, zero and non-zero b, whole-number and fractional b, zero and
# non-zero activation energy.
COEFFICIENTS = [
    pytest.param(1.0e13, 0.0, 0.0, id='A only'),
    pytest.param(1.0e13, 0.0, 5000.0, id='A and E'),
    pytest.param(1.0e13, 0.7, 0.0, id='fractional b, no E'),
    pytest.param(1.0e13, 0.7, 5000.0, id='fractional b and E'),
    pytest.param(1.0e13, -0.7, 0.0, id='negative fractional b, no E'),
    pytest.param(1.0e13, -0.7, 5000.0, id='negative fractional b and E'),
    pytest.param(1.0e13, 2.0, 0.0, id='whole b, no E'),
    pytest.param(1.0e13, 2.0, 5000.0, id='whole b and E'),
    pytest.param(1.0e13, -2.0, 0.0, id='negative whole b, no E'),
    pytest.param(1.0e13, -2.0, 5000.0, id='negative whole b and E'),
    pytest.param(-1.0e13, 0.0, 0.0, id='negative A only'),
    pytest.param(-1.0e13, 0.0, 5000.0, id='negative A and E'),
    pytest.param(-1.0e13, 0.7, 0.0, id='negative A, fractional b'),
    pytest.param(-1.0e13, 0.7, 5000.0, id='negative A, fractional b and E'),
    pytest.param(-1.0e13, 2.0, 0.0, id='negative A, whole b'),
    pytest.param(-1.0e13, 2.0, 5000.0, id='negative A, whole b and E'),
    pytest.param(-1.0e13, -2.0, 0.0, id='negative A, negative whole b'),
    pytest.param(-1.0e13, -2.0, 5000.0, id='negative A, negative whole b and E'),
]


@pytest.mark.parametrize(('pre_exponential', 'exponent', 'activation'), COEFFICIENTS)
@pytest.mark.parametrize('temperature', TEMPERATURES)
def test_emitted_rate_matches_arrhenius(
    pre_exponential, exponent, activation, temperature
):
    """Every emitted expression must evaluate to A * T**b * exp(-E/T)."""
    expression = rxn_rate_const(pre_exponential, exponent, activation)
    expected = arrhenius(pre_exponential, exponent, activation, temperature)
    assert evaluate(expression, temperature) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize('exponent', [1.0, 2.0, 3.0])
def test_whole_exponent_avoids_exp_and_log(exponent):
    """A whole-number exponent with no activation energy becomes a product.

    Repeated multiplication is both cheaper and more accurate than
    ``exp(log(A) + b * log(T))``, and the docstring documents this case, so it
    should apply regardless of the sign of A.
    """
    for pre_exponential in (1.0e13, -1.0e13):
        expression = rxn_rate_const(pre_exponential, exponent, 0.0)
        assert 'exp' not in expression, (
            f'A={pre_exponential:g}, b={exponent:g} should be a plain product, '
            f'got {expression}'
        )
        assert expression.count(' * T') == int(exponent)


def test_zero_pre_exponential_is_rejected():
    """log(0) is undefined, and a zero rate is meaningless, so refuse it."""
    with pytest.raises(NotImplementedError):
        rxn_rate_const(0.0, 1.0, 100.0)


def make_species(low, high):
    """A stand-in carrying just the two coefficient sets get_nasa_arrays reads."""
    return types.SimpleNamespace(lo=list(low), hi=list(high))


#: Distinct values so a transposition or wrong index cannot go unnoticed.
LOW_COEFFS = [2.0, 3.0, 6.0, 12.0, 20.0, 7.0, 11.0]
HIGH_COEFFS = [5.0, 9.0, 18.0, 36.0, 60.0, 13.0, 23.0]


def expected_arrays(coeffs, nu, factor):
    """The rearrangement get_nasa_arrays is specified to perform."""
    scale = nu * factor
    return [
        scale * (coeffs[6] - coeffs[0]),
        scale * (coeffs[0] - 1.0),
        scale * coeffs[1] / 2.0,
        scale * coeffs[2] / 6.0,
        scale * coeffs[3] / 12.0,
        scale * coeffs[4] / 20.0,
        scale * coeffs[5],
    ]


@pytest.mark.parametrize('factor', [1.0, -1.0])
@pytest.mark.parametrize('nu', [1.0, 2.0, 0.5])
def test_nasa_arrays_rearrange_both_ranges(nu, factor):
    """Both temperature ranges are scaled and rearranged the same way."""
    species = make_species(LOW_COEFFS, HIGH_COEFFS)
    low, high = get_nasa_arrays(species, nu, factor=factor)
    assert low == pytest.approx(expected_arrays(LOW_COEFFS, nu, factor))
    assert high == pytest.approx(expected_arrays(HIGH_COEFFS, nu, factor))


def test_nasa_arrays_are_linear_in_nu():
    """Doubling the stoichiometric coefficient doubles every term."""
    species = make_species(LOW_COEFFS, HIGH_COEFFS)
    single_low, single_high = get_nasa_arrays(species, 1.0)
    double_low, double_high = get_nasa_arrays(species, 2.0)
    assert double_low == pytest.approx([2.0 * x for x in single_low])
    assert double_high == pytest.approx([2.0 * x for x in single_high])


def test_nasa_arrays_factor_negates():
    """The reactant side is the product side with the opposite sign."""
    species = make_species(LOW_COEFFS, HIGH_COEFFS)
    product_low, _ = get_nasa_arrays(species, 1.5, factor=1.0)
    reactant_low, _ = get_nasa_arrays(species, 1.5, factor=-1.0)
    assert reactant_low == pytest.approx([-x for x in product_low])


def test_nasa_arrays_does_not_mutate_species():
    """The species' own coefficients must survive the call unchanged."""
    species = make_species(LOW_COEFFS, HIGH_COEFFS)
    get_nasa_arrays(species, 2.0, factor=-1.0)
    assert species.lo == LOW_COEFFS
    assert species.hi == HIGH_COEFFS


@pytest.mark.parametrize('temperature', TEMPERATURES)
def test_reader_supplied_exponents_are_floats(temperature):
    """Guards the branch selection against the type the readers actually pass.

    Both mechanism readers produce ``b`` as a float, so any branch keyed on
    ``isinstance(b, int)`` is unreachable in practice. Passing 2 and 2.0 must
    produce expressions with the same value.
    """
    as_int = rxn_rate_const(1.0e13, 2, 0.0)
    as_float = rxn_rate_const(1.0e13, 2.0, 0.0)
    assert evaluate(as_int, temperature) == pytest.approx(
        evaluate(as_float, temperature), rel=1e-12
    )
