pyJac
===============

This utility creates source code to calculate the Jacobian matrix analytically for a chemical reaction mechanism.

Documentation
-------------

The full documentation for pyJac can be found at http://kyleniemeyer.github.io/pyJac/.

Installation
------------

Detailed installation instructions can be found in the [full documentation](http://kyleniemeyer.github.io/pyJac/). However, pyJac can be installed as a Python module:

    python setup.py install

Usage
-----

pyJac can be run as a python module:

    python -m pyjac [options]

The generated source code is placed within the `out` (by default) directory, which is created if it doesn't exist initially.


Options
-------

In the above, `[options]` indicates where command line options should be specified. The options available can be seen using `-h` or `--help`, or below:

    -h, --help            show this help message and exit
    -l {c,cuda,fortran,matlab}, --lang {c,cuda,fortran,matlab}
                          Programming language for output source files.
    -i INPUT, --input INPUT
                          Input mechanism filename (e.g., mech.dat).
    -t THERMO, --thermo THERMO
                          Thermodynamic database filename (e.g., therm.dat), or
                          nothing if in mechanism.
    -b, --build_path
                          The folder to generate the Jacobian and rate subroutines in
    -ls SPECIES NAME, --last_species SPECIES NAME
                          The species to move to the end of the mechanism to force mass conservation.  
                          Defaults to (in order) N2, Ar, He.
    -ic STRING, --initial-conditions STRING
                          A comma separated list of initial conditions to set in the
                          set_same_initial_conditions method.
                          Expected Form: T,P,Species1=...,Species2=...,...
                              Temperature in K
                              Pressure in Atm
                              Species in moles
    -nco, --no-cache-optimizer
                          Turn off the cache optimization functionality
    -nosmem, --no-shared-memory
                          Turn off shared memory usage for CUDA
    -pshare, --prefer-shared
                          Prefer a larger shared memory size (compared to L1 cache size) for CUDA.  
                          (Not recommended)
    -nb, --num_blocks
                          The target number of blocks per streaming multiprocessor for CUDA.  
                          Default is 8
    -nt, --num_threads
                          The number of threads per block for CUDA.  
                          Default is 64
    -mt, --multi-threaded
                          The number of threads to use during cache-optimization.
                          Default is the CPU count returned by multiprocessing
    -fopt, --force-opt
                          Force re-running of cache optimizer.  
                          By default cache optimizer will turn off if it detects a previous optimization
                          for the same mechanism
    -ad, --auto-diff
                          Generate code for Adept autodifferentiation library (used for validation)
    -sj, --skip_jac
                          Do not generate Jacobian subroutine

License
-------

`pyJac` is released under the MIT license; see the [`LICENSE`](https://github.com/kyleniemeyer/pyJac/blob/master/LICENSE) for details.

If you use this package as part of a scholarly publication, please see the `CITATION.md` for the appropriate citation(s).

Contributing
------------

We welcome contributions to pyJac! Please see the guide to making contributions in the [`CONTRIBUTING.md`](https://github.com/kyleniemeyer/pyJac/blob/master/CONTRIBUTING.md) file.


Authors
-------

Created by [Kyle Niemeyer](http://kyleniemeyer.com) ([kyle.niemeyer@gmail.com](mailto:kyle.niemeyer@gmail.com)) and Nicholas Curtis ([nicholas.curtis@uconn.edu](mailto:nicholas.curtis@uconn.edu))
