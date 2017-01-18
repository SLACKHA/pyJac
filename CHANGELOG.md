# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.0.2] - 2017.01.18
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


[Unreleased]: https://github.com/slackha/pyJac/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/slackha/pyJac/compare/v1.0...v1.0.1
[1.0]: https://github.com/slackha/pyJac/compare/v0.9.1-beta...v1.0
[0.9.1-beta]: https://github.com/slackha/pyJac/compare/v0.9-beta...v0.9.1-beta
