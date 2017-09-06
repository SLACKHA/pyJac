# pyJac

[![DOI](https://zenodo.org/badge/19829533.svg)](https://zenodo.org/badge/latestdoi/19829533)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-contributor%20covenant-green.svg)](http://contributor-covenant.org/version/1/4/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://badge.fury.io/py/pyJac.svg)](https://badge.fury.io/py/pyJac)
[![Anaconda](https://anaconda.org/slackha/pyjac/badges/version.svg)](https://anaconda.org/slackha/pyjac)

This utility creates source code to calculate the Jacobian matrix analytically
for a chemical reaction mechanism.

## Documentation

The full documentation for pyJac can be found at <http://slackha.github.io/pyJac/>.

## User Group

Further support can be found at our [user group](https://groups.io/g/slackha-users),
or by [opening an issue](https://github.com/SLACKHA/pyJac/issues) on our github repo.

## Installation

Detailed installation instructions can be found in the
[full documentation](http://slackha.github.io/pyJac/).
The easiest way to install pyJac is via `conda`. You can install to your
environment with
```
> conda install -c slackha pyjac
```

pyJac can also be installed from PyPI using pip:
```
pip install pyjac
```
or, using the downloaded source code, installed as a Python module:
```
> python setup.py install
```

## Usage

pyJac can be run as a python module:
```
> python -m pyjac [options]
```

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
[LICENSE](https://github.com/slackha/pyJac/blob/master/LICENSE) for details.

If you use this package as part of a scholarly publication, please see
[CITATION.md](https://github.com/slackha/pyJac/blob/master/CITATION.md)
for the appropriate citation(s).

## Contributing

We welcome contributions to pyJac! Please see the guide to making contributions
in the [CONTRIBUTING.md](https://github.com/slackha/pyJac/blob/master/CONTRIBUTING.md)
file.

## Code of Conduct

In order to have a more open and welcoming community, pyJac adheres to a code of conduct adapted from the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

Please adhere to this code of conduct in any interactions you have in the pyJac community. It is strictly enforced on all official pyJac repositories, websites, and resources. If you encounter someone violating these terms, please let a maintainer ([@kyleniemeyer](https://github.com/kyleniemeyer) or [@arghdos](https://github.com/arghdos), via email at <slackha@googlegroups.com>) know and we will address it as soon as possible.

## Authors

Created by [Kyle Niemeyer](http://kyleniemeyer.com) (<kyle.niemeyer@gmail.com>) and
Nicholas Curtis (<arghdos@gmail.com>)
