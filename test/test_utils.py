"""Tests for pyjac.utils."""

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
