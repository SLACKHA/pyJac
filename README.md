# pyJac

This utility creates source code to calculate the Jacobian matrix analytically for a chemical reaction mechanism.

## Documentation

The full documentation for pyJac can be found at http://kyleniemeyer.github.io/pyJac/.

## Installation

Detailed installation instructions can be found in the [full documentation](http://kyleniemeyer.github.io/pyJac/). However, pyJac can be installed as a Python module:

    python setup.py install

or from PyPI using pip:

    pip install pyjac

## Usage

pyJac can be run as a python module:

    python -m pyjac [options]

The generated source code is placed within the `out` (by default) directory, which is created if it doesn't exist initially. See the documentation or use `python pyjac -h` for the full list of options.

## License

`pyJac` is released under the MIT license; see the [`LICENSE`](https://github.com/kyleniemeyer/pyJac/blob/master/LICENSE) for details.

If you use this package as part of a scholarly publication, please see the `CITATION.md` for the appropriate citation(s).

## Contributing

We welcome contributions to pyJac! Please see the guide to making contributions in the [`CONTRIBUTING.md`](https://github.com/kyleniemeyer/pyJac/blob/master/CONTRIBUTING.md) file.


## Authors

Created by [Kyle Niemeyer](http://kyleniemeyer.com) ([kyle.niemeyer@gmail.com](mailto:kyle.niemeyer@gmail.com)) and Nicholas Curtis ([nicholas.curtis@uconn.edu](mailto:nicholas.curtis@uconn.edu))
