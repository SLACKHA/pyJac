# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

Modernization work in progress. The characterization tests in
`test/test_golden_output.py` assert byte-identical C and CUDA output against
recorded fixtures, so every refactor here is verified to leave generated source
untouched. The one deliberate exception is the atomic weight change described
below, whose effect was confirmed to be limited to molecular weights by
regenerating with the old table and diffing.

### Added
- Reader-equivalence tests comparing the mechanism the Chemkin parser builds
  against the one Cantera builds from the same source, field by field. These
  are the safety net for the Cantera 3.x port and are currently strict xfails.
- Golden-output fixtures and characterization tests covering C and CUDA
  generation, the Adept autodifferentiation variant, CUDA without the
  shared-memory manager, generator determinism, and warning-free compilation
  of the generated C: 152 recorded files across 8 variants
- `test/fixtures/mechanisms/rxn_types.inp`, a fixture mechanism exercising
  third-body, Troe falloff, SRI falloff, PLOG, Chebyshev, duplicate,
  irreversible, and explicit-reverse reactions
- `test/regenerate_golden.py` to re-record fixtures when generated output
  changes intentionally
- Smoke-and-compile test for the cache-optimizer path. Its output is not
  byte-compared: `cache_optimizer` is a randomized greedy search on unseeded
  `np.random`, so successive runs legitimately differ.
- Regression tests for Chebyshev parsing and generation, the CLI, the
  unsupported-language guard, and `option_cases`
- `pyproject.toml` with PEP 621 metadata, optional-dependency extras
  (`pywrap`, `cache-opt`, `test`, `docs`), and ruff/pytest/coverage config,
  built with hatchling
- `.pre-commit-config.yaml` (ruff hooks staged but disabled pending the
  one-time lint cleanup)
- `--version` / `-v` flag reporting the pyJac version
- The CLI validates before doing any work: unsupported languages, a missing
  mechanism file, and a missing thermodynamic database all exit through the
  parser with a plain message and status 2, rather than a traceback from deep
  inside generation

### Changed
- Ported `read_mech_ct` to the Cantera 3.x API. Dispatch now keys on the
  reaction's `ReactionRate` type and its `ThirdBody` rather than on the
  `Reaction` subclasses removed in 3.0, and the reader-equivalence tests hold
  it to producing the same mechanism as the Chemkin parser.
- Consolidated the duplicated `is_pdep` predicates and the four inline
  reaction-type checks into `utils.is_pdep` and `utils.is_plog_or_cheb`.
- Cantera YAML (`.yaml`/`.yml`) is now the recognised Cantera input format.
  `.cti` and `.xml` raise `NotImplementedError` pointing at `cti2yaml` and
  `ctml2yaml`; both formats were removed in Cantera 3.0.
- Unsupported rate types (Blowers-Masel, Linear-Burke, Tsang, plasma, surface 
  and user-supplied rates)---now raise `NotImplementedError` naming the type.
- Atomic weights now come from `cantera.Element` rather than a hardcoded table
  taken from an older IUPAC revision, so a mechanism read through the Chemkin
  parser and through Cantera describes identical species masses. **This changes
  generated source**: molecular weights and quantities derived from them shift
  by up to ~6e-5 relative (for example H2 from 2.01588 to 2.016).
- The physical constants `RU`, `RU_JOUL`, `RUC` and `PA` now come from Cantera
  (2018 CODATA) rather than being hardcoded. **This also changes generated
  source**: the gas constant shifts by ~6e-8 relative, from 8314.4621 to
  8314.462618.
- Moved the package to a `src/` layout (`pyjac/` -> `src/pyjac/`) and the test
  suite out of the package to a top-level `test/` directory
- Tests now import `pyjac` absolutely, so they exercise the installed package
- `requires-python` is now `>=3.10`, the floor set by Cantera 3.x
- `_version.py` exposes `__version__` as a literal so build backends can read
  it without importing the package
- Replaced the CodeMeta files with `CITATION.cff`
- Moved the argument parser from `pyjac.utils` to `pyjac.__main__`.
- Fortran and Matlab are now explicitly unsupported, and raise
  `NotImplementedError` before any work is done, and the CLI reports the
  error and exits with status 2.
- Replaced `cantera.ck2cti` with `cantera.ck2yaml`; Chemkin input now converts
  to YAML, as the CTI format and its converter were removed in Cantera 3.0
- Ported `pywrap.parallel_compiler` and the four `*_setup.py.in` templates
  from `distutils` to `setuptools._distutils`
- Dropped the `optionloop` dependency in favour of a small
  `itertools.product` helper (`performance_tester.option_cases`)
- Removed Python 2 compatibility imports, converted all 589 `str.format()`
  calls to f-strings, fixed invalid escape sequences,
  replaced `is` comparisons against string literals with `==`, and
  switched `logging.warn` to `logging.warning`.
- Path parsing now uses `pathlib` rather than `os.path`

### Fixed
- `pywrap.generate_wrapper` built the wrapper with a reconstructed `pythonX.Y`
  name resolved against `PATH`, escaping the active environment and its
  Cython, NumPy and setuptools. It now uses `sys.executable`.
- The wrapper build wrote the filled-in setup script and Cython's generated
  `.c` into the package directory, which is read-only in a normal install.
  Both now go to the build directory.
- The Cantera version check rejected every 3.x release and called
  `sys.exit(1)` at import time, making `import pyjac` fail outright
  with modern Cantera. It now compares versions as a tuple and warns
  instead of exiting.
- A Chebyshev reaction with two or fewer temperature coefficients generated an
  out-of-bounds read on `dot_prod` in `jacob.c`. Where another reaction set a
  larger array size, the read was in bounds but returned that reaction's
  value, silently producing a wrong temperature derivative.
- A standalone `TCHEB/ ... /` or `PCHEB/ ... /` line, not followed by
  the other on the same line, raised `IndexError` in the Chemkin parser
- `cache_optimizer.optimize_cache` unconditionally called a debug `plot()`
  helper. Removed both the call and the helper.
- `libgen.libgen` referenced an undefined `args` when a compiler was missing,
  raising `NameError` instead of reporting the missing compiler
- `rate_subs` referenced an undefined `sp` when estimating shared-memory usage
  for third-body species, raising `NameError` whenever that branch was reached
- `performance_tester` compared a list against an int to size its
  thread sweep, raising `TypeError` on Python 3
- `read_thermo` looped forever at end of file: it tested `line is None`, but
  `readline()` returns `''` when exhausted. Any non-Chemkin input fed to the
  Chemkin parser hung instead of erroring.
- `__main__.main()` ignored a supplied `args` namespace: the entire body sat
  inside `if args is None`, so `main(args)` was a silent no-op. It now returns
  an exit status.

### Removed
- `setup.py`, `setup.cfg`, `MANIFEST.in` (superseded by `pyproject.toml`)
- `conda.recipe/` and `test-environment.yaml`; distribution is now PyPI-only
- The vestigial `__main__` blocks in `core.create_jacobian` and
  `core.rate_subs`; the latter called an imported module as a function
- The source distribution no longer carries the test suite,
  example mechanisms, or documentation sources. The wheel payload is
  unchanged.

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
