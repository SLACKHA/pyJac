# pyJac

[![DOI](https://zenodo.org/badge/19829533.svg)](https://zenodo.org/badge/latestdoi/19829533)
[![PyPI](https://badge.fury.io/py/pyJac.svg)](https://badge.fury.io/py/pyJac)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This utility creates source code to calculate the Jacobian matrix analytically
for a chemical reaction mechanism.

## Documentation

The full documentation for pyJac can be found at <http://slackha.github.io/pyJac/>.

## Installation

Detailed installation instructions can be found in the
[full documentation](http://slackha.github.io/pyJac/). However, pyJac can be
installed as a Python module:

   python setup.py install

or from PyPI using pip:

   pip install pyjac

## Usage

pyJac can be run as a python module:

   python -m pyjac [options]

The generated source code is placed within the `out` (by default) directory,
which is created if it doesn't exist initially.
See the documentation or use `python pyjac -h` for the full list of options.

## Theory

Theory, derivations, validation and performance testing can be found in the paper
fully describing version 1.0.2 of pyJac: <https://niemeyer-research-group.github.io/pyJac-paper/>,
now published via <https://doi.org/10.1016/j.cpc.2017.02.004> and available
openly via [`arXiv:1605.03262 [physics.comp-ph]`](https://arxiv.org/abs/1605.03262).

## License

pyJac is released under the MIT license; see the
[LICENSE](https://github.com/slackha/pyJac/blob/master/LICENSE)> for details.

If you use this package as part of a scholarly publication, please see
[CITATION.md](https://github.com/slackha/pyJac/blob/master/CITATION.md)
for the appropriate citation(s).

## Contributing

We welcome contributions to pyJac! Please see the guide to making contributions
in the [CONTRIBUTING.md](https://github.com/slackha/pyJac/blob/master/CONTRIBUTING.md)
file.

## Authors

Created by [Kyle Niemeyer](http://kyleniemeyer.com) (<kyle.niemeyer@gmail.com>) and
Nicholas Curtis (<nicholas.curtis@uconn.edu>)
