# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/) 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]
### Added

### Changed
- Minor compilation fixes for including OpenMP
### Depreciated
- Shared library creation for CUDA disabled, as CUDA does not allow linkage of SO's into another CUDA kernel
### Fixed
- Explicitly conserve mass in PaSR
- Minor path fixes

## [1.0.1] - 2016-05-25
### Added
- Added GPU macros, e.g. THREAD_ID, GRID_SIZE
### Changed
- Much better handling of removal of files created during testing
### Fixed
- Bugfix that generates data.bin's correctly from npy files for performance testing (**important**)
- Explicit setting of omp # threads for performance testing

## [1.0] - 2016-05-07
### Added
- pyJac is now a python package
- pyJac now posesses the ability to create a static/shared library for a mechanism (for external linkage)
- documentation added
- examples added
### Changed
- much better handling of CUDA compilation via cython
- pointers are now restricted where appropriate
- better python3 compatiblitity
### Fixed
- other minor bugfixes

## [0.9.1-beta] - 2015-10-29
###Changed
- Implemented the strict mass conservation formulation
- Updated CUDA implementation such that it is testable v.s. pyJac c-version (and Cantera where applicable)
- More robust build folder management
- More robust mapping for strict mass consv. testing 

## 0.9-beta - 2015-10-02
###Added
- First working / tested version of pyJac


[Unreleased]: https://github.com/kyleniemeyer/pyJac/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/kyleniemeyer/pyJac/compare/v1.0...v1.0.1
[1.0]: https://github.com/kyleniemeyer/pyJac/compare/v0.9.1-beta...v1.0
[0.9.1-beta]: https://github.com/kyleniemeyer/pyJac/compare/v0.9-beta...v0.9.1-beta
