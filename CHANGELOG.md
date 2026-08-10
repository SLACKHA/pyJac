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
- Numerical validation of the generated code against Cantera, described in
  `docs/validation.rst`. `test/test_rate_validation.py` compiles the generated
  rate subroutines and compares concentrations, per-reaction forward rates of
  progress, and net production rates; `test/test_jacobian_validation.py` does
  the same for the Jacobian, against a reference built in
  `test/jacobian_reference.py` from Cantera's analytic kinetics derivatives
  plus the constant-pressure chain rule. Together they cover h2o2, the
  all-reaction-types fixture, and GRI-Mech 3.0, and agree to ~1e-9 and ~2e-10
  respectively. This checks correctness rather than stability.
- `test/test_cuda_validation.py`, which evaluates generated CUDA kernels on a
  GPU and compares against the same Cantera reference the C validation uses,
  at the same tolerances. Skipped unless both nvcc and a GPU are present. The
  CUDA backend had only ever been checked for compiling.
- A test that builds the CUDA Python wrapper, which needs nvcc but not a GPU,
  run by the CUDA jobs in CI. Those jobs previously built only through
  `pyjac.libgen`, leaving the wrapper templates uncovered.
- Unit tests for the expression-building helpers, in
  `test/test_rate_expressions.py`. Emitted rate expressions are evaluated and
  compared against `A T**b exp(-E/T)` rather than matched as text, and
  `get_thermo_expression` is checked against Cantera's thermodynamic data for
  every GRI-Mech species over both temperature ranges.
- Tests for the `pyjac.utils` helpers, including a property test that the
  species mappings are inverse permutations for every species count and every
  choice of eliminated species.
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
- A docs workflow that builds with warnings as errors and deploys to
  GitHub Pages after the test suite passes on main. Released docs are
  published at the site root and development docs at `/dev`.
  Pull requests build the docs but never deploy.
- GitHub Actions workflows: a test matrix over Python 3.10-3.14 on Linux,
  macOS and Windows; a full-suite job including the slow cache-optimizer
  tests; lint via pre-commit; a build job that checks the sdist can rebuild
  the wheel; a CUDA job that compiles generated sources through
  `pyjac.libgen` against both CUDA 12 and 13; and PyPI publishing on release
  via trusted publishing.
- `pyproject.toml` with PEP 621 metadata, optional-dependency extras
  (`pywrap`, `cache-opt`, `test`, `docs`), and ruff/pytest/coverage config,
  built with hatchling
- `.pre-commit-config.yaml`, including the ruff lint and format hooks
- `--version` / `-v` flag reporting the pyJac version
- The CLI validates before doing any work: unsupported languages, a missing
  mechanism file, and a missing thermodynamic database all exit through the
  parser with a plain message and status 2, rather than a traceback from deep
  inside generation

### Changed
- Consolidated the four duplicated species loops in `write_chem_utils` into
  `_write_thermo_loop` and a pure `get_thermo_expression`, which builds
  enthalpy, internal energy, cv and cp from two flags rather than four
  near-identical blocks, and the `eval_conc`/`eval_conc_rho` bodies into
  `_write_conc_body`, parameterised on which of density and pressure is
  supplied. Both reject an unknown property or quantity rather than falling
  through to a default.
- The CUDA target architecture is configurable instead of hardcoded to
  `sm_20`. Fermi support was removed in CUDA 9 (2017), so the CUDA backend
  could not compile on any current toolkit. It now defaults to `sm_75` and is
  settable with `--cuda-arch` on `pyjac.libgen` and `pyjac.pywrap`, or the
  `cuda_arch` argument to `generate_library` and `generate_wrapper`.
  Turing is the oldest architecture that compiles offline on every tested
  toolkit: CUDA 13 dropped offline compilation for Maxwell, Pascal and Volta,
  so `sm_70` builds under CUDA 12.6 but fails under 13.3. Targets older than
  the default remain available through `--cuda-arch` on a toolkit that still
  accepts them.
- CUDA register limits updated from Fermi's values to those of compute
  capability 5.0 and later: 65536 registers per multiprocessor rather than
  32768, and a 255-register per-thread cap rather than 63. The emitted
  `regcount` rises from 63 to 128 at the default launch geometry.
  Shared memory stays at 48 KB, which is still the portable limit for
  statically declared shared memory on every current architecture.
- Applied the ruff cleanup and enabled the ruff pre-commit hooks. Every rule in
  the selected set now passes with no ignores beyond `E501`.
- Narrowed all 18 bare `except:` clauses to the exceptions they are actually
  guarding, and required `zip()` calls to state their strictness.
- The partially stirred reactor now rejects an odd particle count. Particles are
  mixed in non-overlapping pairs, so an odd count silently left one unmixed.
- The functional and performance testers now take Cantera YAML. Both refuse
  `.cti`/`.xml` with a pointer to Cantera's converters, and the functional
  tester logs when it converts a Chemkin mechanism. `performance_tester`
  discovers mechanisms by `.yaml`/`.yml` rather than `.cti`, which it would
  never have found.
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
- Rewrote documentation configuration to work with current Sphinx.
- Documentation and README updated for the current package: Cantera YAML
  rather than `.cti` in every example, `pip install` rather than
  `python setup.py install`, the optional-dependency extras explained, and the
  retired conda channel removed.

### Fixed
- The CUDA Python wrapper could not be built on any toolkit released in the
  last several years. Four independent causes, each hidden behind the one
  before it: the setup template located and required the toolkit's `samples`
  directory, dropped by NVIDIA after CUDA 11.5 and needed only for the
  `helper_cuda.h` include already removed from generated code; host compiler
  flags from Python's build configuration, which now include
  `-fno-strict-overflow` and `-Wsign-compare`, were passed straight to nvcc,
  which rejects rather than ignores them; only `linker_so` was rewritten for
  nvcc, while distutils builds the command for a C++ extension from
  `linker_so_cxx`, added later and left holding raw host flags; and the
  interpreter's own `-Wl,--rpath=...` needs `-Xlinker`, since nvcc splits a
  `-Xcompiler` argument on commas and gcc understands neither `-Wl` nor
  `--rpath=...` alone. Nothing covered this path: the CUDA job in CI built
  through `pyjac.libgen`, which compiles generated sources directly and never
  touches the wrapper templates.
- Generated CUDA declared `dot_prod` twice in `eval_jacob` for any mechanism
  with a Chebyshev reaction, once to pass into `eval_rxn_rates` and once for
  the Jacobian's own use, so it did not compile. **This changes generated
  CUDA**: the duplicate declaration is gone. The C backend is unaffected,
  where the second declaration is a local array and the first is never
  emitted.
- The eliminated species' contribution to `d(dT/dt)/dT` was overwritten rather
  than accumulated. That species has no Jacobian entry, so its running total is
  kept in a scratch variable, and the choice between assignment and
  accumulation consulted a flag that was never set for it; each contribution
  overwrote the previous one and only the last survived. **This changes
  computed Jacobians.** On GRI-Mech 3.0, 26 contributions collapsed to one and
  the entry was wrong by 0.74% at 1800 K, while every other entry of the matrix
  was correct to 2e-10 and `dT/dt` itself was correct throughout. Only
  mechanisms whose eliminated species reacts are affected, which with the
  default choice of N2 means any mechanism with NOx chemistry; mechanisms
  closed on an actual inert species such as Ar were always correct.
- `rxn_rate_const` dropped the temperature dependence of a rate with a negative
  pre-exponential, a negative whole-number temperature exponent and no
  activation energy, emitting `A` where `A T**b` was meant, because the
  repeated-multiplication path iterated over an empty range. The two sign
  branches also disagreed on how to detect a whole exponent. No mechanism in
  the test suite reaches this combination, so no generated output changes.
- `get_cheb_rate` read past the end of a Chebyshev fit carrying a single
  temperature or pressure coefficient, which Cantera accepts: it emitted a
  `Tred * dot_prod[1]` term for a one-row fit, producing an out-of-bounds read
  in the generated code, and raised `IndexError` on a one-column fit.
- `get_sri_dt` described Troe falloff in its docstring rather than SRI.
- Removes unused include of `helper_cuda.h` from generated CUDA code,
  which was removed from the toolkit.
- `libgen.compiler` called `sys.exit` from inside a `multiprocessing.Pool`
  worker when the compiler was missing, so `generate_library` hung waiting on
  a result that never arrived rather than reporting the missing compiler.
- Three test modules asserted on `sys.modules` without importing what they
  checked, so they passed only when another module had imported it first and
  failed when run alone.
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
- Five bare `except:` clauses in `mech_auxiliary` caught the `SystemExit` raised
  by their own `sys.exit(1)`, so a malformed initial-conditions string reported
  "not comma separated" regardless of what was actually wrong with it.
- `get_register_count` returned a float once the register budget stopped
  being clamped by an integer literal, so `regcount` would have been written as
  `128.0` and rejected by nvcc's `-maxrregcount`.
- `read_thermo` looped forever at end of file: it tested `line is None`, but
  `readline()` returns `''` when exhausted. Any non-Chemkin input fed to the
  Chemkin parser hung instead of erroring.
- `__main__.main()` ignored a supplied `args` namespace: the entire body sat
  inside `if args is None`, so `main(args)` was a silent no-op. It now returns
  an exit status.

### Removed
- `.travis.yml` and `appveyor.yml`, replaced by GitHub Actions
- `data/h2o2.cti` and `data/h2o2_performance/h2o2.cti`, replaced by YAML
  equivalents; Cantera 3.x cannot read the CTI format
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
