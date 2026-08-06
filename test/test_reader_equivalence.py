"""The Chemkin and Cantera readers must describe the same mechanism.

``read_mech`` (Chemkin) and ``read_mech_ct`` (Cantera) build the same internal
`SpecInfo`/`ReacInfo` representation from the same chemistry, so parsing one
mechanism both ways and comparing the results checks the Cantera reader against
a Chemkin reader that is already covered by the golden fixtures.

This is the safety net for the Cantera 3.x port.

Several representational differences are expected and normalised away. They
are all artefacts of how the two input formats describe identical chemistry,
not disagreements about the chemistry itself:

* Element names differ in case (``AR`` from Chemkin, ``Ar`` from Cantera).
* Where a species' high- and low-temperature polynomials are identical,
  ``ck2yaml`` collapses them to a single range by setting the midpoint to the
  maximum temperature. The same polynomial is evaluated either way.
* Chemkin writes a reaction with a single specific collider by naming it on
  both sides (``H+O2+AR<=>HO2+AR``); Cantera stores it as a third-body
  reaction whose only non-zero efficiency is that species. The rate expression
  is the same, so the Cantera form is folded back into the explicit form.
* Chemkin keeps the ``A``/``b``/``E`` written on a PLOG or Chebyshev reaction
  line; Cantera zeroes them. Neither reader's values are used -- generation
  reads ``plog_par``/``cheb_par`` for these, and the only two references to
  ``rxn.A`` in ``rate_subs`` are guarded by ``not (rxn.cheb or rxn.plog)``.

Molecular weights agree exactly, because `chem_utilities.get_elem_wt` sources
atomic weights from Cantera.
"""

import copy

import numpy as np
import pytest

from conftest import GOLDEN_MECHS, MECH_DIR
from pyjac import utils
from pyjac.core.create_jacobian import create_jacobian
from pyjac.core.mech_interpret import read_mech, read_mech_ct

#: Both readers now draw atomic weights from Cantera, so molecular weights,
#: like the rate parameters, must agree to machine precision.
RATE_RTOL = 1e-12

SCALAR_RATE_FIELDS = ('A', 'b', 'E')
FLAG_FIELDS = ('rev', 'dup', 'thd_body', 'pdep', 'troe', 'sri', 'cheb', 'plog')
LIST_RATE_FIELDS = ('low', 'high', 'troe_par', 'sri_par', 'rev_par')


def _approx(value, rel):
    return pytest.approx(value, rel=rel, abs=1e-30)


def _normalise_composition(elem):
    """Elemental composition as {element: count}, case- and type-normalised."""
    return {str(name).lower(): float(count) for name, count in elem}


def _compare_species(a, b, label):
    """Return a list of human-readable differences between two `SpecInfo`."""
    diffs = []
    if a.name != b.name:
        diffs.append(f'{label}: name {a.name!r} != {b.name!r}')
        return diffs

    if _normalise_composition(a.elem) != _normalise_composition(b.elem):
        diffs.append(f'{label} {a.name}: composition {a.elem} != {b.elem}')

    if b.mw != _approx(a.mw, RATE_RTOL):
        diffs.append(f'{label} {a.name}: mw {a.mw} != {b.mw}')

    single_range = list(a.lo) == pytest.approx(list(a.hi), rel=1e-12)
    trange_a, trange_b = list(a.Trange), list(b.Trange)
    if single_range:
        # the midpoint is irrelevant when both polynomials are the same
        trange_a, trange_b = [trange_a[0], trange_a[2]], [trange_b[0], trange_b[2]]
    if trange_b != _approx(trange_a, 1e-10):
        diffs.append(f'{label} {a.name}: Trange {a.Trange} != {b.Trange}')

    for field in ('lo', 'hi'):
        av, bv = list(getattr(a, field)), list(getattr(b, field))
        if len(av) != len(bv):
            diffs.append(f'{label} {a.name}: {field} length {len(av)} != {len(bv)}')
        elif bv != _approx(av, 1e-10):
            diffs.append(f'{label} {a.name}: {field} coefficients differ')
    return diffs


def _fold_single_collider(reac):
    """Return a copy of ``reac`` with a lone third-body collider made explicit.

    Cantera describes ``H+O2+AR<=>HO2+AR`` as a third-body reaction whose only
    non-zero efficiency is AR. Chemkin names the collider on both sides
    instead. Rewriting the former into the latter lets the two be compared.
    """
    if not reac.thd_body or not reac.thd_body_eff:
        return reac

    nonzero = [(sp, eff) for sp, eff in reac.thd_body_eff if eff != 0.0]
    if len(nonzero) != 1 or nonzero[0][1] != 1.0:
        return reac

    collider = nonzero[0][0]
    folded = copy.deepcopy(reac)
    folded.thd_body = False
    folded.thd_body_eff = []
    for names, nus in ((folded.reac, folded.reac_nu), (folded.prod, folded.prod_nu)):
        if collider in names:
            nus[names.index(collider)] += 1
        else:
            names.append(collider)
            nus.append(1)
    return folded


def _compare_reactions(a, b, label):
    """Return a list of human-readable differences between two `ReacInfo`."""
    diffs = []
    a, b = _fold_single_collider(a), _fold_single_collider(b)

    for field in FLAG_FIELDS:
        if getattr(a, field) != getattr(b, field):
            diffs.append(f'{label}: {field} {getattr(a, field)} != {getattr(b, field)}')

    for field in ('reac', 'prod'):
        if sorted(getattr(a, field)) != sorted(getattr(b, field)):
            diffs.append(f'{label}: {field} {getattr(a, field)} != {getattr(b, field)}')

    for field in ('reac_nu', 'prod_nu'):
        av = [float(v) for v in getattr(a, field)]
        bv = [float(v) for v in getattr(b, field)]
        if sorted(av) != sorted(bv):
            diffs.append(f'{label}: {field} {av} != {bv}')

    if not (a.plog or a.cheb):
        for field in SCALAR_RATE_FIELDS:
            av, bv = getattr(a, field), getattr(b, field)
            if bv != _approx(av, RATE_RTOL):
                diffs.append(f'{label}: {field} {av} != {bv}')

    for field in LIST_RATE_FIELDS:
        av = [float(v) for v in getattr(a, field)]
        bv = [float(v) for v in getattr(b, field)]
        if len(av) != len(bv):
            diffs.append(f'{label}: {field} length {len(av)} != {len(bv)}')
        elif av and bv != _approx(av, RATE_RTOL):
            diffs.append(f'{label}: {field} {av} != {bv}')

    a_thd = {str(sp).lower(): float(eff) for sp, eff in a.thd_body_eff}
    b_thd = {str(sp).lower(): float(eff) for sp, eff in b.thd_body_eff}
    if a_thd.keys() != b_thd.keys():
        diffs.append(f'{label}: third-body species {sorted(a_thd)} != {sorted(b_thd)}')
    else:
        for sp in a_thd:
            if b_thd[sp] != _approx(a_thd[sp], RATE_RTOL):
                diffs.append(
                    f'{label}: third-body efficiency for {sp} '
                    f'{a_thd[sp]} != {b_thd[sp]}'
                )

    if a.pdep_sp != b.pdep_sp:
        diffs.append(f'{label}: pdep_sp {a.pdep_sp!r} != {b.pdep_sp!r}')

    if a.plog:
        # plog_par is a list of [pressure, A, b, E] rows, so compare row by row;
        # pytest.approx does not descend into nested sequences.
        av = [[float(x) for x in row] for row in (a.plog_par or [])]
        bv = [[float(x) for x in row] for row in (b.plog_par or [])]
        if len(av) != len(bv):
            diffs.append(f'{label}: plog has {len(av)} pressures != {len(bv)}')
        else:
            for j, (arow, brow) in enumerate(zip(av, bv, strict=True)):
                if brow != _approx(arow, RATE_RTOL):
                    diffs.append(f'{label}: plog row {j} {arow} != {brow}')

    if a.cheb:
        for field in ('cheb_n_temp', 'cheb_n_pres'):
            if getattr(a, field) != getattr(b, field):
                diffs.append(
                    f'{label}: {field} {getattr(a, field)} != {getattr(b, field)}'
                )
        for field in ('cheb_plim', 'cheb_tlim'):
            av = [float(v) for v in getattr(a, field)]
            bv = [float(v) for v in getattr(b, field)]
            if bv != _approx(av, 1e-10):
                diffs.append(f'{label}: {field} {av} != {bv}')

        # cheb_par is a 2-D numpy array of fit coefficients
        apar = np.asarray(a.cheb_par, dtype=float)
        bpar = np.asarray(b.cheb_par, dtype=float)
        if apar.shape != bpar.shape:
            diffs.append(f'{label}: cheb_par shape {apar.shape} != {bpar.shape}')
        elif not np.allclose(apar, bpar, rtol=RATE_RTOL, atol=0.0):
            diffs.append(f'{label}: Chebyshev coefficients differ')
    return diffs


def compare_mechanisms(first, second):
    """Compare two ``(elems, specs, reacs)`` triples, returning differences.

    An empty list means the two readers describe the same chemistry.
    """
    elems_a, specs_a, reacs_a = first
    elems_b, specs_b, reacs_b = second
    diffs = []

    if sorted(e.lower() for e in elems_a) != sorted(e.lower() for e in elems_b):
        diffs.append(f'elements {sorted(elems_a)} != {sorted(elems_b)}')

    if len(specs_a) != len(specs_b):
        diffs.append(f'species count {len(specs_a)} != {len(specs_b)}')
    else:
        for i, (sa, sb) in enumerate(zip(specs_a, specs_b, strict=True)):
            diffs.extend(_compare_species(sa, sb, f'species[{i}]'))

    if len(reacs_a) != len(reacs_b):
        diffs.append(f'reaction count {len(reacs_a)} != {len(reacs_b)}')
    else:
        for i, (ra, rb) in enumerate(zip(reacs_a, reacs_b, strict=True)):
            diffs.extend(_compare_reactions(ra, rb, f'reaction[{i}]'))
    return diffs


# --------------------------------------------------------------------------
# The comparator itself must be neither vacuous nor over-sensitive.
# --------------------------------------------------------------------------


@pytest.mark.parametrize('mech', sorted(GOLDEN_MECHS))
def test_comparator_finds_no_differences_against_itself(mech):
    """A mechanism compared with itself reports nothing."""
    parsed = read_mech(str(GOLDEN_MECHS[mech]), None)
    assert compare_mechanisms(parsed, copy.deepcopy(parsed)) == []


@pytest.mark.parametrize(
    'field,value',
    [
        ('A', 1.5),
        ('b', 99.0),
        ('E', 1234.0),
        ('rev', None),
        ('dup', None),
    ],
)
def test_comparator_detects_a_changed_reaction_field(field, value):
    """Planting a difference in one reaction is reported."""
    parsed = read_mech(str(GOLDEN_MECHS['h2o2']), None)
    mutated = copy.deepcopy(parsed)
    original = getattr(mutated[2][3], field)
    setattr(mutated[2][3], field, not original if value is None else value)

    diffs = compare_mechanisms(parsed, mutated)
    assert diffs, f'comparator missed a changed {field}'
    assert any('reaction[3]' in d and field in d for d in diffs), diffs


def test_comparator_detects_a_changed_species():
    """Planting a difference in one species is reported."""
    parsed = read_mech(str(GOLDEN_MECHS['h2o2']), None)
    mutated = copy.deepcopy(parsed)
    mutated[1][2].mw *= 1.01

    diffs = compare_mechanisms(parsed, mutated)
    assert any('species[2]' in d and 'mw' in d for d in diffs), diffs


def test_chemkin_molecular_weights_match_cantera(to_cantera_yaml):
    """The Chemkin reader's molecular weights match Cantera's exactly.

    ``get_elem_wt`` sources atomic weights from ``cantera.Element`` precisely so
    that the two input paths cannot describe different species masses.
    """
    ct = pytest.importorskip('cantera')
    yaml_path = to_cantera_yaml(GOLDEN_MECHS['h2o2'])
    gas = ct.Solution(str(yaml_path))

    _, specs, _ = read_mech(str(GOLDEN_MECHS['h2o2']), None)
    assert [s.name for s in specs] == list(gas.species_names)
    for spec, expected in zip(specs, gas.molecular_weights, strict=True):
        assert spec.mw == expected, f'{spec.name}: {spec.mw} != {expected}'


def test_element_weight_overrides_do_not_leak():
    """A mechanism's own atomic weights must not affect later reads.

    ``get_elem_wt`` returns a fresh dict per call; the Chemkin ELEMENTS block
    overwrites entries in place.
    """
    from pyjac.core.chem_utilities import get_elem_wt

    first = get_elem_wt()
    first['h'] = 999.0
    assert get_elem_wt()['h'] != 999.0


# --------------------------------------------------------------------------
# The comparison this all exists for.
# --------------------------------------------------------------------------


@pytest.mark.parametrize('mech', sorted(GOLDEN_MECHS))
def test_readers_describe_the_same_mechanism(mech, to_cantera_yaml):
    """Chemkin and Cantera readers agree on the same source mechanism."""
    chemkin_path = GOLDEN_MECHS[mech]
    yaml_path = to_cantera_yaml(chemkin_path)

    from_chemkin = read_mech(str(chemkin_path), None)
    from_cantera = read_mech_ct(str(yaml_path))

    diffs = compare_mechanisms(from_chemkin, from_cantera)
    assert not diffs, (
        f'{mech}: Chemkin and Cantera readers disagree:\n  ' + '\n  '.join(diffs[:20])
    )


# --------------------------------------------------------------------------
# Rate types pyJac has no formulation for must be refused, not mistranslated.
# --------------------------------------------------------------------------


def test_unsupported_rate_type_is_rejected():
    """A Blowers-Masel reaction raises rather than being silently mishandled."""
    mech = MECH_DIR / 'blowers_masel.yaml'
    with pytest.raises(NotImplementedError) as excinfo:
        read_mech_ct(str(mech))

    message = str(excinfo.value)
    assert 'BlowersMaselRate' in message
    assert 'Chebyshev' in message, 'error should list the supported rate types'


def test_unsupported_rate_types_are_named_consistently():
    """Every name in the reject list is a real Cantera rate class."""
    ct = pytest.importorskip('cantera')
    for name in utils.unsupported_rate_types:
        assert hasattr(ct, name), f'{name} is not a cantera rate class'


def test_supported_rate_types_are_not_in_the_reject_list():
    """The five forms pyJac does implement must never be rejected."""
    ct = pytest.importorskip('cantera')
    supported = (
        'ArrheniusRate',
        'LindemannRate',
        'TroeRate',
        'SriRate',
        'PlogRate',
        'ChebyshevRate',
    )
    for name in supported:
        assert hasattr(ct, name)
        assert name not in utils.unsupported_rate_types


def test_legacy_cantera_formats_are_refused(tmp_path):
    """.cti and .xml were removed in Cantera 3.0; say so instead of failing late."""
    for suffix in ('.cti', '.xml'):
        legacy = tmp_path / f'mech{suffix}'
        legacy.write_text('')
        with pytest.raises(NotImplementedError) as excinfo:
            create_jacobian('c', mech_name=str(legacy), build_path=str(tmp_path))
        assert 'cti2yaml' in str(excinfo.value) or 'ctml2yaml' in str(excinfo.value)
