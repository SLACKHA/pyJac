"""Numerical validation of the generated Jacobian against Cantera.

`jacobian_reference` builds the same quantity pyJac computes -- the derivative
of {T, Y_1 ... Y_{N-1}} with respect to itself at constant pressure -- from
Cantera's analytic kinetics derivatives. These tests generate, compile and
evaluate pyJac's Jacobian and compare against it.

As in the rate tests, each mechanism runs in its own subprocess because the
compiled extension is always named ``pyjacob``.
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from conftest import GOLDEN_MECHS, MECH_DIR

#: Agreement required, measured against the largest entry of the matrix.
#:
#: Scaling by the matrix rather than by each entry is deliberate. Individual
#: Jacobian entries pass through zero, and a relative comparison against an
#: entry that is nearly cancelling reports a large error for a difference that
#: is negligible in context. Against the matrix scale the observed
#: disagreement is ~2e-10 across every mechanism and state below.
JACOBIAN_RTOL = 1e-8

#: Agreement required for the temperature self-derivative alone.
#:
#: Compared against a Richardson-extrapolated finite difference rather than
#: `analytic_jacobian`, which is the less accurate reference for this entry --
#: see `temperature_derivative_by_extrapolation`. Against that, pyJac agrees to
#: ~3e-10 for every mechanism and state here, the same floor as the rest of the
#: matrix, so this can be as tight as JACOBIAN_RTOL.
TEMPERATURE_RTOL = 1e-8

STATES = [
    (800.0, 1.0),
    (1200.0, 1.0),
    (1800.0, 10.0),
]


def gri30_path():
    """Path to the gri30 mechanism bundled with Cantera."""
    ct = pytest.importorskip('cantera')
    path = pathlib.Path(ct.__file__).parent / 'data' / 'gri30.yaml'
    if not path.is_file():
        pytest.skip('cantera does not bundle gri30.yaml')
    return path


_COMPARE = textwrap.dedent("""
    import json, sys
    import cantera as ct
    import numpy as np
    from jacobian_reference import (
        analytic_jacobian,
        temperature_derivative_by_extrapolation,
    )
    from pyjac.functional_tester.test import cpyjac_evaluator

    mech, build_dir, states = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])

    gas = ct.Solution(mech)
    evaluator = cpyjac_evaluator(build_dir, gas)
    n_species = gas.n_species

    # pyJac may move the eliminated species to the end and renumber the rest.
    # The reference has to eliminate the same species and use the same order,
    # or the two matrices describe different state vectors.
    order = np.asarray(evaluator.fwd_spec_map)

    worst_matrix = 0.0
    worst_corner = 0.0

    for temperature, atm in states:
        pressure = atm * ct.one_atm
        gas.TPX = temperature, pressure, {
            s: 1.0 / n_species for s in gas.species_names
        }

        flat = np.zeros(n_species * n_species)
        evaluator.eval_jacobian(
            0, pressure, np.hstack((temperature, gas.Y)), flat
        )
        # pyJac flattens the Jacobian in column-major order.
        got = flat.reshape((n_species, n_species), order='F')

        partial = gas.Y[order[:-1]].copy()
        want = analytic_jacobian(gas, temperature, partial, pressure, order)

        # The temperature self-derivative gets its own, more accurate reference.
        extrapolated = temperature_derivative_by_extrapolation(
            gas, temperature, partial, pressure, order
        )
        corner = abs(got[0, 0] - extrapolated) / abs(extrapolated)

        difference = np.abs(got - want)
        difference[0, 0] = 0.0

        worst_matrix = max(worst_matrix, difference.max() / np.abs(want).max())
        worst_corner = max(worst_corner, corner)

    json.dump({'matrix': worst_matrix, 'temperature': worst_corner}, sys.stdout)
""")


def build_and_compare(source, cantera_yaml, work_dir):
    """Generate, compile, wrap and compare; return worst disagreements."""
    pytest.importorskip('Cython', reason='building the wrapper requires Cython')
    pytest.importorskip('setuptools', reason='building the wrapper requires setuptools')

    from pyjac.core.create_jacobian import create_jacobian
    from pyjac.pywrap import generate_wrapper

    build = work_dir / 'out'

    # generate_wrapper compiles into ./build relative to the working directory.
    previous = os.getcwd()
    os.chdir(work_dir)
    try:
        create_jacobian('c', mech_name=str(source), build_path=str(build))
        generate_wrapper('c', str(build), out_dir=str(work_dir))
    finally:
        os.chdir(previous)

    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(pathlib.Path(__file__).parent)

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            _COMPARE,
            str(cantera_yaml),
            str(build),
            json.dumps(STATES),
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, (
        f'comparison failed:\n{result.stdout}\n{result.stderr}'
    )
    return json.loads(result.stdout)


def to_yaml(chemkin_path, out_dir):
    """Convert a Chemkin mechanism to Cantera YAML alongside the build."""
    ck2yaml = pytest.importorskip('cantera.ck2yaml')
    out_name = out_dir / (pathlib.Path(chemkin_path).stem + '.yaml')
    ck2yaml.convert(
        str(chemkin_path), out_name=str(out_name), permissive=True, quiet=True
    )
    return out_name


@pytest.fixture(scope='module')
def h2o2_result(tmp_path_factory):
    """Built once; both assertions below read from it."""
    work = tmp_path_factory.mktemp('h2o2')
    chemkin = GOLDEN_MECHS['h2o2']
    return build_and_compare(chemkin, to_yaml(chemkin, work), work)


@pytest.fixture(scope='module')
def rxn_types_result(tmp_path_factory):
    work = tmp_path_factory.mktemp('rxn_types')
    chemkin = MECH_DIR / 'rxn_types.inp'
    return build_and_compare(chemkin, to_yaml(chemkin, work), work)


@pytest.fixture(scope='module')
def gri30_result(tmp_path_factory):
    work = tmp_path_factory.mktemp('gri30')
    mech = gri30_path()
    return build_and_compare(mech, mech, work)


#: How closely the reference must track a finite difference of the same RHS.
#:
#: Central differencing loses accuracy on stiff entries, so this compares the
#: median rather than the worst entry. It checks this repository's chain rule,
#: not pyJac -- both sides evaluate the same Cantera-based right-hand side.
REFERENCE_FD_RTOL = 1e-7


@pytest.mark.parametrize('mechanism', ['h2o2.yaml', 'gri30.yaml'])
@pytest.mark.parametrize('eliminate', [None, 'N2'], ids=['last species', 'reordered'])
def test_reference_chain_rule_agrees_with_finite_difference(mechanism, eliminate):
    """The reference's calculus is right, for either choice of eliminated species.

    pyJac moves the eliminated species to the end and renumbers the rest, so
    the reference has to handle an arbitrary choice, not just the last one.
    """
    ct = pytest.importorskip('cantera')
    import numpy as np

    from jacobian_reference import analytic_jacobian, finite_difference_jacobian

    gas = ct.Solution(mechanism)
    if eliminate is None:
        index = gas.n_species - 1
    elif eliminate in gas.species_names:
        index = gas.species_index(eliminate)
    else:
        pytest.skip(f'{mechanism} has no {eliminate}')
    order = np.array([i for i in range(gas.n_species) if i != index] + [index])

    for temperature, atm in STATES:
        pressure = atm * ct.one_atm
        gas.TPX = (
            temperature,
            pressure,
            dict.fromkeys(gas.species_names, 1.0 / gas.n_species),
        )
        partial = gas.Y[order[:-1]].copy()

        exact = analytic_jacobian(gas, temperature, partial, pressure, order)
        approx = finite_difference_jacobian(gas, temperature, partial, pressure, order)

        significant = np.abs(approx) > 1e-6 * np.abs(approx).max()
        error = np.abs(exact - approx)[significant] / np.abs(approx)[significant]
        assert np.median(error) < REFERENCE_FD_RTOL, (
            f'{mechanism} at {temperature:.0f} K: reference disagrees with '
            f'finite difference, median {np.median(error):.3e}'
        )


@pytest.mark.compiler
@pytest.mark.slow
@pytest.mark.parametrize(
    'result_name', ['h2o2_result', 'rxn_types_result', 'gri30_result']
)
def test_jacobian_matches_cantera(result_name, request):
    """Every entry but the temperature self-derivative matches the reference."""
    worst = request.getfixturevalue(result_name)['matrix']
    assert worst < JACOBIAN_RTOL, (
        f'generated Jacobian disagrees with Cantera by {worst:.3e} '
        f'relative to the largest entry'
    )


@pytest.mark.compiler
@pytest.mark.slow
@pytest.mark.parametrize(
    'result_name', ['h2o2_result', 'rxn_types_result', 'gri30_result']
)
def test_temperature_self_derivative_matches_cantera(result_name, request):
    """The temperature row's temperature derivative matches the reference.

    This entry sums a contribution from every species, including the
    eliminated one, whose running total is accumulated separately. Getting
    that accumulation wrong moves only this entry, so it needs its own check:
    the other N*N-1 entries stayed correct to 2e-10 while this one was off by
    0.7% on gri30.
    """
    worst = request.getfixturevalue(result_name)['temperature']
    assert worst < TEMPERATURE_RTOL, (
        f'd(dT/dt)/dT disagrees with Cantera by {worst:.3e}'
    )
