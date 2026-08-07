"""Tests for pyjac.utils."""

import types

import pytest

from pyjac import utils


def test_public_api_is_exported():
    for name in utils.__all__:
        assert hasattr(utils, name), f'{name} listed in __all__ but missing'


def test_language_tables_cover_every_language():
    for table in (utils.comment, utils.file_ext, utils.line_end, utils.array_chars):
        assert set(table) >= set(utils.langs)


def test_supported_languages_have_header_extensions():
    """Only the implemented backends need header extensions.

    Fortran and Matlab are absent from header_ext, which is what made
    generation die with a KeyError before they were rejected up front.
    """
    assert set(utils.header_ext) == set(utils.supported_langs)


@pytest.mark.parametrize('num_specs', range(1, 9))
def test_species_mappings_are_inverse_permutations(num_specs):
    """The two maps must invert each other for every choice of last species.

    pyJac moves one species to the end of the state vector and recovers its
    mass fraction from the others. Both directions are needed, and a mapping
    that is not a clean inverse silently attributes rates to the wrong species.
    """
    for last_species in range(num_specs):
        forward, backward = utils.get_species_mappings(num_specs, last_species)

        assert sorted(forward) == list(range(num_specs)), 'forward map lost an index'
        assert sorted(backward) == list(range(num_specs)), 'backward map lost an index'
        assert forward[-1] == last_species, 'chosen species is not last'
        for new_index in range(num_specs):
            assert backward[forward[new_index]] == new_index


def make_reaction(reac, reac_nu, prod, prod_nu):
    """A stand-in carrying only the stoichiometry get_nu reads."""
    return types.SimpleNamespace(
        reac=list(reac), reac_nu=list(reac_nu), prod=list(prod), prod_nu=list(prod_nu)
    )


@pytest.mark.parametrize(
    ('isp', 'expected'),
    [
        pytest.param(0, -2, id='reactant only'),
        pytest.param(1, 3, id='product only'),
        pytest.param(2, 1, id='both sides, net positive'),
        pytest.param(3, -1, id='both sides, net negative'),
        pytest.param(4, 0, id='both sides, cancels'),
        pytest.param(5, 0, id='not in the reaction'),
    ],
)
def test_net_stoichiometric_coefficient(isp, expected):
    """Net nu is production minus consumption, and zero when absent."""
    rxn = make_reaction(
        reac=[0, 2, 3, 4], reac_nu=[2, 1, 2, 1], prod=[1, 2, 3, 4], prod_nu=[3, 2, 1, 1]
    )
    assert utils.get_nu(isp, rxn) == expected


@pytest.mark.parametrize(
    ('text', 'sep', 'expected'),
    [
        ('1.0 2.0 3.0', None, [1.0, 2.0, 3.0]),
        ('  1.0   2.0  ', None, [1.0, 2.0]),
        ('1.0,2.0,3.0', ',', [1.0, 2.0, 3.0]),
        ('1.0E+03 -2.5e-2', None, [1000.0, -0.025]),
    ],
)
def test_read_str_num(text, sep, expected):
    assert utils.read_str_num(text, sep) == pytest.approx(expected)


@pytest.mark.parametrize(
    ('seq', 'length', 'expected'),
    [
        ('abcdef', 2, ['ab', 'cd', 'ef']),
        ('abcdefg', 3, ['abc', 'def', 'g']),
        ('abc', 5, ['abc']),
        ('', 3, []),
    ],
)
def test_split_str_keeps_every_character(seq, length, expected):
    """Trailing partial chunks are kept; Chemkin thermo lines rely on it."""
    assert utils.split_str(seq, length) == expected
    assert ''.join(utils.split_str(seq, length)) == seq


@pytest.mark.parametrize(
    ('lang', 'expected'),
    [
        ('c', 'conc[3]'),
        # CUDA routes every access through the INDEX macro so that neighbouring
        # threads read neighbouring addresses.
        ('cuda', 'conc[INDEX(3)]'),
        ('fortran', 'conc(4)'),
        ('matlab', 'conc(4)'),
    ],
)
def test_get_array_indexing_base(lang, expected):
    """C and CUDA index from zero; Fortran and Matlab from one."""
    assert utils.get_array(lang, 'conc', 3) == expected


def test_get_array_without_index_returns_bare_name():
    """A None index is the probe used to ask whether a name is in shared memory."""
    assert utils.get_array('c', 'conc', None) == 'conc'


@pytest.mark.parametrize('lang', ['fortran', 'matlab'])
def test_get_array_two_dimensional(lang):
    assert utils.get_array(lang, 'jac', 2, twod=5) == 'jac(3, 6)'


@pytest.mark.parametrize(
    ('lang', 'expected'), [('c', '7'), ('cuda', '7'), ('fortran', '8'), ('matlab', '8')]
)
def test_get_index_matches_array_base(lang, expected):
    assert utils.get_index(lang, 7) == expected


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (3, True),
        (-3, True),
        (0, True),
        (3.0, True),
        (-3.0, True),
        (3.5, False),
        (-0.5, False),
    ],
)
def test_is_integer(value, expected):
    """Whole floats count as integers; the readers only ever supply floats."""
    assert utils.is_integer(value) is expected
