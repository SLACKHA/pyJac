"""Tests for the pyjac.core modules."""

import pytest

from pyjac.core import (
    CUDAParams,
    cache_optimizer,
    chem_utilities,
    create_jacobian,
    mech_auxiliary,
    mech_interpret,
    rate_subs,
    shared_memory,
)


@pytest.mark.parametrize(
    'module,attribute',
    [
        (cache_optimizer, 'optimize_cache'),
        (chem_utilities, 'get_elem_wt'),
        (create_jacobian, 'create_jacobian'),
        (mech_auxiliary, 'write_mechanism_initializers'),
        (mech_interpret, 'read_mech'),
        (mech_interpret, 'read_mech_ct'),
        (rate_subs, 'write_rxn_rates'),
        (shared_memory, 'shared_memory_manager'),
        (CUDAParams, 'write_launch_bounds'),
    ],
)
def test_expected_entry_points_exist(module, attribute):
    assert hasattr(module, attribute), f'{module.__name__} lost {attribute}'


def test_physical_constants_are_consistent():
    """RU_JOUL is RU expressed per mole rather than per kilomole."""
    assert chem_utilities.RU_JOUL == pytest.approx(chem_utilities.RU / 1000.0)
    assert chem_utilities.RUC == pytest.approx(chem_utilities.RU / 4.184)
