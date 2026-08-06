"""Regression tests for Chebyshev parsing and Jacobian generation.

Both bugs below predate 1.0.6 and went unnoticed because real Chebyshev fits
usually carry four or more temperature coefficients, and because ``.todo`` has
listed "Test Jacobian with PLOG and CHEB reactions" as never done.
"""

import re
import subprocess

import pytest

from conftest import MECH_DIR
from pyjac.core.create_jacobian import create_jacobian
from pyjac.core.mech_interpret import read_mech

CHEB_SMALL = MECH_DIR / 'cheb_small.inp'


def test_standalone_tcheb_line_parses():
    """A TCHEB/ / line not followed by PCHEB on the same line must parse.

    The parser indexed ``line_split[3]`` unconditionally when looking for
    limits sharing the line, raising IndexError for the standalone form.
    """
    _, _, reacs = read_mech(str(CHEB_SMALL), None)

    cheb = [r for r in reacs if r.cheb]
    assert len(cheb) == 1, 'fixture should hold exactly one Chebyshev reaction'
    rxn = cheb[0]
    assert rxn.cheb_tlim == [300.0, 2500.0]
    assert rxn.cheb_n_temp == 2
    assert rxn.cheb_n_pres == 3
    # PCHEB values are given in atm and converted to Pa.
    assert rxn.cheb_plim == pytest.approx([0.01 * 101325.0, 100.0 * 101325.0])


def test_two_temperature_coefficients_stay_in_bounds(tmp_path):
    """dot_prod must not be read past cheb_n_temp - 1.

    With two temperature coefficients dot_prod holds a single usable entry, but
    the temperature-derivative expression always emitted ``dot_prod[2]``. That
    overran the array when this reaction set the mechanism-wide size, and read
    another reaction's stale value when it did not.
    """
    create_jacobian('c', mech_name=str(CHEB_SMALL), build_path=str(tmp_path))

    jacob = (tmp_path / 'jacob.c').read_text()

    declared = {int(n) for n in re.findall(r'double\s+dot_prod\[(\d+)\]', jacob)}
    assert declared, 'no dot_prod declaration found in generated Jacobian'
    assert len(declared) == 1, f'inconsistent dot_prod sizes: {declared}'
    size = declared.pop()

    used = {
        int(n) for n in re.findall(r'(?<!double )(?<!double  )dot_prod\[(\d+)\]', jacob)
    }
    out_of_bounds = {i for i in used if i >= size}
    assert not out_of_bounds, (
        f'generated Jacobian indexes dot_prod at {sorted(out_of_bounds)} '
        f'but the array is declared with size {size}'
    )
    assert 'kf = dot_prod[1]' in jacob


@pytest.mark.compiler
def test_two_temperature_chebyshev_compiles_without_warnings(tmp_path, c_compiler):
    """The generated C compiles clean; -Warray-bounds caught the original bug."""
    create_jacobian('c', mech_name=str(CHEB_SMALL), build_path=str(tmp_path))

    failures = []
    for source in sorted(tmp_path.glob('*.c')):
        result = subprocess.run(
            [
                c_compiler,
                '-std=c99',
                '-O2',
                '-fPIC',
                '-Wall',
                '-Wextra',
                '-Warray-bounds',
                '-Wno-unused-parameter',
                '-I',
                str(tmp_path),
                '-c',
                str(source),
                '-o',
                str(tmp_path / (source.stem + '.o')),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or result.stderr.strip():
            failures.append(f'{source.name}:\n{result.stderr}')

    assert not failures, 'generated C did not compile cleanly:\n' + '\n'.join(failures)
