# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

Modernization work in progress. Generated source output is unchanged: the
characterization tests in `test/test_golden_output.py` assert byte-identical
C and CUDA output against fixtures recorded from 1.0.6.

### Added
- Golden-output fixtures and characterization tests covering C and CUDA
  generation, generator determinism, and warning-free compilation of the
  generated C (`test/fixtures/`, `test/test_golden_output.py`)
- `test/fixtures/mechanisms/rxn_types.inp`, a fixture mechanism exercising
  third-body, Troe falloff, SRI falloff, PLOG, Chebyshev, duplicate,
  irreversible, and explicit-reverse reactions
- `test/regenerate_golden.py` to re-record fixtures when generated output
  changes intentionally
- `pyproject.toml` with PEP 621 metadata, optional-dependency extras
  (`pywrap`, `cache-opt`, `test`, `docs`), and ruff/pytest/coverage config,
  built with hatchling
- `.pre-commit-config.yaml` (ruff hooks staged but disabled pending the
  one-time lint cleanup)

### Changed
- Replaces CodeMeta files with `CITATION.cff`
- Moved the package to a `src/` layout (`pyjac/` -> `src/pyjac/`) and the test
  suite out of the package to a top-level `test/` directory
- Tests now import `pyjac` absolutely, so they exercise the installed package
- `requires-python` is now `>=3.10`, the floor set by Cantera 3.x
- `_version.py` exposes `__version__` as a literal so build backends can read
  it without importing the package

### Fixed
- Cantera version check rejected every 3.x release and called `sys.exit(1)` at
  import time, making `import pyjac` fail outright with modern Cantera. It now
  compares versions as a tuple and warns instead of exiting.

### Removed
- `setup.py`, `setup.cfg`, `MANIFEST.in` (superseded by `pyproject.toml`)
- `conda.recipe/` and `test-environment.yaml`; distribution is now PyPI-only
- The source distribution no longer carries the test suite, example mechanisms,
  or documentation sources. Those are development content and remain in the git
  repository; the sdist now holds only what is needed to install and run pyJac,
  which took it from 355 KB to 101 KB. The wheel payload is unchanged.

### Known issues (not yet addressed)
- `pyjac.pywrap.parallel_compiler` and the `pywrap` setup templates still
  import `distutils`, removed from the stdlib in Python 3.12
- `pyjac.functional_tester` and `pyjac.performance_tester` still import
  `cantera.ck2cti`, removed in Cantera 3.0
- `read_mech_ct` still dispatches on Cantera reaction classes removed in 3.0
- Fortran and Matlab generation raise `KeyError` on the first write, as
  `utils.header_ext` defines only `c` and `cuda`; this predates 1.0.6
- A Chebyshev reaction with two or fewer temperature coefficients generates an
  out-of-bounds read on `dot_prod` in `jacob.c`
- A standalone `TCHEB/ ... /` line (not followed by `PCHEB` on the same line)
  raises `IndexError` in the Chemkin parser

## [1.0.6] - 2018-02-21
### Added
- DOI for 1.0.4

### Fixed
- Syntax errors in readme.md
- Conda install instructions in install.md
- Corrected TRange columns in parser
- Minor documentation fixes

### Added
- Add check to reactions to test that all species exist
- Duplicate warning from falloff->chemically-activated TROE reactions for zero-parameters
- Add handling of non-unity default third body efficiency

### Changed
- Bump internal version to 1.0.5.c

## [1.0.5.b0] - 2017-06-02
### Added
- Added usergroup info to README and documentation

### Fixed

### Changed
- Now strip whitespace from mechanism file lines prior to parsing keywords

### Removed
- Removed plotting scripts specific to first paper on pyJac

## [1.0.4] - 2017-04-18
### Added
 - Adds Travis config for automatic PyPI and conda builds
 - Adds minimal unittest test suite for module imports
 - Adds code of conduct

### Changed
 - Changed README back to Markdown for simplicity
 - Updated citation instructions

## [1.0.3] - 2017-04-01
### Fixed
 - Fix for SRI Falloff functions with non-default third bodies ([issue #12](https://github.com/SLACKHA/pyJac/issues/12))
 - Fixed removal of jac/rate lists before libgen of functional_tester
 - Fixed pywrap module import

### Changed
 - Issue warning in Cantera parsing if the installed version doesn't have access to species thermo properties.

### Added
 - Added significantly more documentation and examples for data ordering,
 the state vector / Jacobian, and using the python interface

## [1.0.2] - 2017-01-18
### Added
 - Added CHANGELOG
 - Added documentation for libgen / pywrap features

### Changed
 - Minor compilation fixes for including OpenMP
 - Updated github links to point to SLACKHA / Niemeyer Research Group

### Deprecated
 - Shared library creation for CUDA disabled, as CUDA does not allow linkage of SO's into another CUDA kernel

### Fixed
 - Explicitly conserve mass in PaSR
 - Minor path fixes
 - Division by zero in some TROE parameter cases

## [1.0.1] - 2016-05-25
### Added
 - Added GPU macros, e.g., THREAD_ID, GRID_SIZE

### Changed
 - Much better handling of removal of files created during testing

### Fixed
 - Bugfix that generates data.bin files correctly from .npy files for performance testing (**important**)
 - Explicit setting of OpenMP # threads for performance testing

## [1.0] - 2016-05-07
### Added
 - pyJac is now a Python package
 - pyJac can now create a static/shared library for a mechanism (for external linkage)
 - Added documentation
 - Added examples

### Changed
 - Handles CUDA compilation better via Cython
 - pointers are now restricted where appropriate
 - better Python3 compatibility

### Fixed
 - other minor bugfixes

## [0.9.1-beta] - 2015-10-29
### Changed
 - Implemented the strict mass conservation formulation
 - Updated CUDA implementation such that it is testable vs. pyJac c-version (and Cantera where applicable)
 - More robust build folder management
 - More robust mapping for strict mass conservation

## 0.9-beta - 2015-10-02
### Added
 - First working / tested version of pyJac


[Unreleased]: https://github.com/slackha/pyJac/compare/v1.0.4...HEAD
[1.0.4]: https://github.com/slackha/pyJac/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/slackha/pyJac/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/slackha/pyJac/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/slackha/pyJac/compare/v1.0...v1.0.1
[1.0]: https://github.com/slackha/pyJac/compare/v0.9.1-beta...v1.0
[0.9.1-beta]: https://github.com/slackha/pyJac/compare/v0.9-beta...v0.9.1-beta
