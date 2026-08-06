"""Tests for pyjac.performance_tester.

The module imports again now that ``cantera.ck2cti`` and the unmaintained
``optionloop`` dependency are gone. It is not yet fully functional: ``is_pdep``
still dispatches on Cantera reaction classes removed in 3.0, which is part of
the Cantera 3.x port rather than this work.
"""

import sys

from pyjac.performance_tester import performance_tester


def test_performance_tester_imported():
    assert 'pyjac.performance_tester.performance_tester' in sys.modules


def test_option_cases_expands_a_single_set():
    """List values sweep; scalars stay fixed."""
    cases = list(
        performance_tester.option_cases(
            {'lang': 'c', 'finite_diffs': [False, True], 'threads': [1, 2]}
        )
    )
    assert len(cases) == 4
    assert all(case['lang'] == 'c' for case in cases)
    assert {(c['finite_diffs'], c['threads']) for c in cases} == {
        (False, 1),
        (False, 2),
        (True, 1),
        (True, 2),
    }


def test_option_cases_concatenates_sets():
    """Sets are visited in order, reproducing optionloop's ``+``."""
    cases = list(
        performance_tester.option_cases(
            {'lang': 'c', 'finite_diffs': [False, True]},
            {'lang': 'tchem', 'threads': [1]},
        )
    )
    assert [case['lang'] for case in cases] == ['c', 'c', 'tchem']


def test_option_cases_defaults_missing_options_to_false():
    """An option absent from a set reads back False, as optionloop did."""
    cases = list(
        performance_tester.option_cases(
            {'lang': 'c', 'finite_diffs': [True]},
            {'lang': 'cuda', 'shared': [True]},
        )
    )
    c_case, cuda_case = cases
    assert c_case['shared'] is False
    assert cuda_case['finite_diffs'] is False


def test_option_cases_sweeps_thread_counts_for_c_only():
    """C sweeps thread counts; CUDA has none and must fall back to -1.

    ``num_threads`` reaches the test binary as an argv string, so a case
    without a thread count has to yield the -1 sentinel rather than the False
    that a missing option reads back as.
    """
    c_params = {'lang': 'c', 'finite_diffs': [False], 'num_threads': [1, 2, 4]}
    cuda_params = {'lang': 'cuda', 'shared': [False]}

    cases = list(performance_tester.option_cases(c_params, cuda_params))
    c_cases = [c for c in cases if c['lang'] == 'c']
    cuda_cases = [c for c in cases if c['lang'] == 'cuda']

    assert sorted(c['num_threads'] for c in c_cases) == [1, 2, 4]
    for case in cuda_cases:
        assert (case['num_threads'] or -1) == -1


def test_option_cases_skips_empty_sets():
    """An unavailable backend contributes nothing."""
    cases = list(
        performance_tester.option_cases(
            {'lang': 'c', 'finite_diffs': [False]},
            {},
        )
    )
    assert len(cases) == 1
    assert cases[0]['lang'] == 'c'
