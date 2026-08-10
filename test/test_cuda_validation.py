"""Numerical validation of generated CUDA against Cantera, on a real GPU.

Everything else holding the CUDA backend to account checks that it *compiles*:
the golden fixtures pin the emitted source and CI builds it for several
architectures. Nothing evaluates a kernel. These tests do, comparing the same
quantities as the C validation against the same Cantera reference, so the two
backends are held to one standard.

Skipped unless nvcc and a GPU are both present, so the suite is unaffected
where there is neither. Run them explicitly with ``-m cuda``.

The CUDA build is compiled for whatever architecture the installed GPU
reports. Note that a V100 is sm_70, which CUDA 13 cannot target at all, so
that pairing needs a 12.x toolkit.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

import pytest

from conftest import GOLDEN_MECHS, MECH_DIR, read_comparison

#: Agreement required against Cantera, matching the C validation's bar.
RATE_RTOL = 1e-8

#: Agreement required for the Jacobian, relative to the largest entry.
JACOBIAN_RTOL = 1e-8

#: Agreement required for the temperature self-derivative, against a
#: Richardson-extrapolated finite difference. See `test_jacobian_validation`.
TEMPERATURE_RTOL = 1e-8

STATES = [
    (800.0, 1.0),
    (1200.0, 1.0),
    (1800.0, 10.0),
]


def compute_capability():
    """Returns the installed GPU's architecture as an ``sm_XX`` string."""
    if shutil.which('nvidia-smi') is None:
        return None
    probe = subprocess.run(
        ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return None
    first = probe.stdout.strip().splitlines()[0].strip()
    return 'sm_' + first.replace('.', '')


@pytest.fixture(scope='session')
def cuda_arch():
    """The architecture to build for, or a skip if there is nothing to build for."""
    if shutil.which('nvcc') is None:
        pytest.skip('nvcc not on PATH')
    arch = compute_capability()
    if arch is None:
        pytest.skip('no GPU visible to nvidia-smi')
    return arch


_COMPARE = textwrap.dedent("""
    import json, sys
    import cantera as ct
    import numpy as np
    from jacobian_reference import (
        analytic_jacobian,
        temperature_derivative_by_extrapolation,
    )
    from pyjac.core.mech_interpret import read_mech, read_mech_ct
    from pyjac.functional_tester.test import cupyjac_evaluator

    mech, build_dir = sys.argv[1], sys.argv[2]
    states, source, out_path = json.loads(sys.argv[3]), sys.argv[4], sys.argv[5]

    gas = ct.Solution(mech)
    n_species = gas.n_species

    # cupyjac_evaluator evaluates every condition up front, so the whole sweep
    # is handed over at construction. Column 0 is unused; the evaluator drops
    # it and reads temperature, pressure, then mass fractions.
    state_data = np.zeros((len(states), 3 + n_species))
    for row, (temperature, atm) in enumerate(states):
        gas.TPX = temperature, atm * ct.one_atm, dict.fromkeys(
            gas.species_names, 1.0 / n_species
        )
        state_data[row, 1] = temperature
        state_data[row, 2] = gas.P
        state_data[row, 3:] = gas.Y

    evaluator = cupyjac_evaluator(build_dir, gas, state_data)
    order = np.asarray(evaluator.fwd_spec_map)

    # Which reactions carry a separate pressure-modification factor comes from
    # pyJac's reading of the mechanism, not Cantera's; the two disagree for a
    # reaction written with an explicit collider.
    if source.endswith(('.yaml', '.yml')):
        _, _, pyjac_reacs = read_mech_ct(source)
    else:
        _, _, pyjac_reacs = read_mech(source, None)
    n_rev = len([r for r in pyjac_reacs if r.rev])
    n_pmod = len([r for r in pyjac_reacs if r.thd_body or r.pdep])

    worst = {}

    def record(label, got, want):
        got, want = np.asarray(got, float), np.asarray(want, float)
        nonzero = np.abs(want) > 0
        if not nonzero.any():
            return
        err = np.max(np.abs((got[nonzero] - want[nonzero]) / want[nonzero]))
        worst[label] = max(worst.get(label, 0.0), float(err))

    worst_matrix = 0.0
    worst_corner = 0.0

    for index, (temperature, atm) in enumerate(states):
        pressure = atm * ct.one_atm
        gas.TPX = temperature, pressure, dict.fromkeys(
            gas.species_names, 1.0 / n_species
        )
        evaluator.update(index)

        conc = np.zeros(n_species)
        evaluator.eval_conc(temperature, pressure, gas.Y, conc)
        record('concentrations', conc, gas.concentrations)

        spec = np.zeros(n_species)
        fwd = np.zeros(gas.n_reactions)
        rev = np.zeros(n_rev)
        pmod = np.zeros(n_pmod)
        evaluator.eval_rxn_rates(temperature, pressure, conc, fwd, rev)
        evaluator.get_rxn_pres_mod(temperature, pressure, conc, pmod)
        evaluator.eval_spec_rates(fwd, rev, pmod, spec)
        record('net production rates', spec, gas.net_production_rates)

        flat = np.zeros(n_species * n_species)
        evaluator.eval_jacobian(0, pressure, np.hstack((temperature, gas.Y)), flat)
        got = flat.reshape((n_species, n_species), order='F')

        partial = gas.Y[order[:-1]].copy()
        want = analytic_jacobian(gas, temperature, partial, pressure, order)
        extrapolated = temperature_derivative_by_extrapolation(
            gas, temperature, partial, pressure, order
        )

        worst_corner = max(
            worst_corner, abs(got[0, 0] - extrapolated) / abs(extrapolated)
        )
        difference = np.abs(got - want)
        difference[0, 0] = 0.0
        worst_matrix = max(worst_matrix, difference.max() / np.abs(want).max())

    evaluator.clean()

    worst['jacobian'] = worst_matrix
    worst['temperature derivative'] = worst_corner
    with open(out_path, 'w') as handle:
        json.dump(worst, handle)
""")


def build_and_compare(source, cantera_yaml, work_dir, arch):
    """Generate CUDA, build it for ``arch``, run it, and compare."""
    pytest.importorskip('Cython', reason='building the wrapper requires Cython')
    pytest.importorskip('setuptools', reason='building the wrapper requires setuptools')

    from pyjac.core.create_jacobian import create_jacobian
    from pyjac.pywrap import generate_wrapper

    build = work_dir / 'out'

    previous = os.getcwd()
    os.chdir(work_dir)
    try:
        create_jacobian('cuda', mech_name=str(source), build_path=str(build))
        generate_wrapper('cuda', str(build), out_dir=str(work_dir), cuda_arch=arch)
    finally:
        os.chdir(previous)

    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(pathlib.Path(__file__).parent)

    written = work_dir / 'comparison.json'
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            _COMPARE,
            str(cantera_yaml),
            str(build),
            json.dumps(STATES),
            str(source),
            str(written),
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=environment,
    )
    return read_comparison(written, result)


def to_yaml(chemkin_path, out_dir):
    """Convert a Chemkin mechanism to Cantera YAML alongside the build."""
    ck2yaml = pytest.importorskip('cantera.ck2yaml')
    out_name = out_dir / (pathlib.Path(chemkin_path).stem + '.yaml')
    ck2yaml.convert(
        str(chemkin_path), out_name=str(out_name), permissive=True, quiet=True
    )
    return out_name


def assert_agrees(worst):
    """Hold each quantity to the same bar as the C validation."""
    bad = []
    for label, value in sorted(worst.items()):
        limit = {
            'jacobian': JACOBIAN_RTOL,
            'temperature derivative': TEMPERATURE_RTOL,
        }.get(label, RATE_RTOL)
        if value > limit:
            bad.append(f'{label} {value:.3e} (limit {limit:g})')
    assert not bad, 'generated CUDA disagrees with Cantera: ' + ', '.join(bad)
    assert worst, 'nothing was compared'


@pytest.mark.cuda
@pytest.mark.compiler
@pytest.mark.slow
@pytest.mark.parametrize('name', ['h2o2', 'rxn_types'])
def test_cuda_matches_cantera(name, tmp_path, cuda_arch):
    """Generated CUDA reproduces Cantera as closely as the C backend does.

    rxn_types carries SRI, PLOG and Chebyshev, so between the two mechanisms
    every supported reaction form is evaluated on the GPU.
    """
    chemkin = GOLDEN_MECHS['h2o2'] if name == 'h2o2' else MECH_DIR / 'rxn_types.inp'
    worst = build_and_compare(chemkin, to_yaml(chemkin, tmp_path), tmp_path, cuda_arch)
    assert_agrees(worst)


@pytest.mark.cuda
@pytest.mark.compiler
@pytest.mark.slow
def test_cuda_matches_cantera_at_scale(tmp_path, cuda_arch):
    """A realistic mechanism, read through the Cantera reader.

    Separate from the small mechanisms because nvcc takes considerably longer
    over 53 species and 325 reactions than gcc does.
    """
    ct = pytest.importorskip('cantera')
    mech = pathlib.Path(ct.__file__).parent / 'data' / 'gri30.yaml'
    if not mech.is_file():
        pytest.skip('cantera does not bundle gri30.yaml')
    worst = build_and_compare(mech, mech, tmp_path, cuda_arch)
    assert_agrees(worst)
