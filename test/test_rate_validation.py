"""Numerical validation of generated rate subroutines against Cantera.

The golden fixtures prove generated source is *stable*; they say nothing about
it being *correct*. These tests close that gap: they generate C, compile it,
build the Python wrapper, evaluate it at several thermochemical states, and
compare against Cantera evaluating the same mechanism.

Each mechanism is exercised in its own subprocess. The compiled extension is
always named ``pyjacob``, and a CPython process can only import one module of
a given name, so several mechanisms cannot share a process.
"""

import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

from conftest import GOLDEN_MECHS, MECH_DIR

#: Agreement required between pyJac and Cantera, relative.
#:
#: Observed worst case across the mechanisms below is ~1e-9, which is
#: accumulated floating-point difference: pyJac evaluates its own thermodynamic
#: polynomials and orders the arithmetic differently from Cantera. This leaves
#: roughly a decade of headroom while staying far tighter than any real defect
#: would produce.
RATE_RTOL = 1e-8

#: States at which to compare. Every species is given a non-zero mole fraction
#: on purpose: most reactions need a radical, and seeding only stable species
#: leaves them with a zero-concentration reactant, so every rate comes out zero
#: and the comparison passes while proving nothing.
STATES = [
    (800.0, 1.0),
    (1200.0, 1.0),
    (1800.0, 10.0),
]


def _gri30_path():
    """Path to the gri30 mechanism bundled with Cantera."""
    ct = pytest.importorskip('cantera')
    path = pathlib.Path(ct.__file__).parent / 'data' / 'gri30.yaml'
    if not path.is_file():
        pytest.skip('cantera does not bundle gri30.yaml')
    return path


# Runs in a subprocess: builds the evaluator against the compiled module and
# reports the worst relative disagreement for each quantity as JSON.
_COMPARE = textwrap.dedent("""
    import json, sys
    import cantera as ct
    import numpy as np
    from pyjac.core.mech_interpret import read_mech, read_mech_ct
    from pyjac.functional_tester.test import cpyjac_evaluator

    mech, build_dir, states = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    source = sys.argv[4]

    gas = ct.Solution(mech)
    ev = cpyjac_evaluator(build_dir, gas)

    # Which reactions carry a separate pressure-modification factor has to come
    # from pyJac's reading of the mechanism, not Cantera's. The two disagree
    # for a reaction written with an explicit collider, such as
    # H+O2+O2<=>HO2+O2: Cantera stores a three-body reaction, while the Chemkin
    # reader folds the collider into the rate expression. Sizing these arrays
    # from Cantera leaves trailing zeros that silently zero out real rates.
    if source.endswith(('.yaml', '.yml')):
        _, _, pyjac_reacs = read_mech_ct(source)
    else:
        _, _, pyjac_reacs = read_mech(source, None)

    composition = {s: 1.0 / gas.n_species for s in gas.species_names}
    idx_pmod = [i for i, r in enumerate(pyjac_reacs) if r.thd_body or r.pdep]
    n_rev = len([r for r in pyjac_reacs if r.rev])

    worst = {}

    def record(label, got, want):
        got, want = np.asarray(got, float), np.asarray(want, float)
        nonzero = np.abs(want) > 0
        if not nonzero.any():
            return
        err = np.max(np.abs((got[nonzero] - want[nonzero]) / want[nonzero]))
        worst[label] = max(worst.get(label, 0.0), float(err))

    for temperature, atm in states:
        pressure = atm * ct.one_atm
        gas.TPX = temperature, pressure, composition

        conc = np.zeros(gas.n_species)
        ev.eval_conc(temperature, pressure, gas.Y, conc)
        record('concentrations', conc, gas.concentrations)

        fwd = np.zeros(gas.n_reactions)
        rev = np.zeros(n_rev)
        ev.eval_rxn_rates(temperature, pressure, conc, fwd, rev)

        pmod = np.zeros(len(idx_pmod))
        ev.get_rxn_pres_mod(temperature, pressure, conc, pmod)

        spec = np.zeros(gas.n_species)
        ev.eval_spec_rates(fwd, rev, pmod, spec)
        record('net production rates', spec, gas.net_production_rates)

        # pyJac keeps the pressure modification separate; Cantera folds it into
        # the rate of progress, so apply it before comparing.
        fwd[idx_pmod] *= pmod
        record('forward rates of progress', fwd, gas.forward_rates_of_progress)

    json.dump(worst, sys.stdout)
""")


def _build_and_compare(chemkin, cantera_yaml, tmp_path, monkeypatch, states=STATES):
    """Generate, compile, wrap, and compare; return worst errors by quantity."""
    pytest.importorskip('Cython', reason='building the wrapper requires Cython')
    pytest.importorskip('setuptools', reason='building the wrapper requires setuptools')

    from pyjac.core.create_jacobian import create_jacobian
    from pyjac.pywrap import generate_wrapper

    source = str(chemkin if chemkin is not None else cantera_yaml)
    build = tmp_path / 'out'

    # generate_wrapper compiles into ./build, relative to the working
    # directory. Without this the object files land in the repository, and
    # every mechanism shares one build tree.
    monkeypatch.chdir(tmp_path)

    create_jacobian('c', mech_name=source, build_path=str(build))
    generate_wrapper('c', str(build), out_dir=str(tmp_path))

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            _COMPARE,
            str(cantera_yaml),
            str(build),
            json.dumps(states),
            source,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'comparison failed:\n{result.stdout}\n{result.stderr}'
    )
    return json.loads(result.stdout)


def _assert_agrees(worst, expected_quantities):
    assert set(worst) == set(expected_quantities), (
        f'expected to compare {sorted(expected_quantities)}, got {sorted(worst)}'
    )
    bad = {k: v for k, v in worst.items() if v > RATE_RTOL}
    assert not bad, (
        'generated rates disagree with Cantera beyond '
        f'{RATE_RTOL:g}: ' + ', '.join(f'{k} {v:.3e}' for k, v in sorted(bad.items()))
    )


QUANTITIES = ('concentrations', 'net production rates', 'forward rates of progress')


@pytest.mark.compiler
@pytest.mark.slow
def test_h2o2_rates_match_cantera(tmp_path, monkeypatch, to_cantera_yaml):
    """A small Chemkin mechanism, generated through the Chemkin reader."""
    chemkin = GOLDEN_MECHS['h2o2']
    worst = _build_and_compare(chemkin, to_cantera_yaml(chemkin), tmp_path, monkeypatch)
    _assert_agrees(worst, QUANTITIES)


@pytest.mark.compiler
@pytest.mark.slow
def test_all_reaction_types_match_cantera(tmp_path, monkeypatch, to_cantera_yaml):
    """Every reaction form pyJac supports, in one mechanism.

    Covers third-body, Troe falloff, SRI falloff, PLOG, Chebyshev, duplicate,
    irreversible and explicit-reverse reactions. gri30 below has none of the
    SRI, PLOG or Chebyshev forms, so this fixture is what exercises them.
    """
    chemkin = MECH_DIR / 'rxn_types.inp'
    worst = _build_and_compare(chemkin, to_cantera_yaml(chemkin), tmp_path, monkeypatch)
    _assert_agrees(worst, QUANTITIES)


@pytest.mark.compiler
@pytest.mark.slow
def test_gri30_rates_match_cantera(tmp_path, monkeypatch):
    """A realistic mechanism, generated through the Cantera reader.

    53 species and 325 reactions, read from the YAML Cantera bundles rather
    than a copy in this repository. This is the only rate test that goes
    through ``read_mech_ct`` rather than the Chemkin parser, and the only one
    covering Lindemann falloff.
    """
    mech = _gri30_path()
    worst = _build_and_compare(None, mech, tmp_path, monkeypatch)
    _assert_agrees(worst, QUANTITIES)
