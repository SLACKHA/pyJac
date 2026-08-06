# pyJac

[![DOI](https://zenodo.org/badge/19829533.svg)](https://zenodo.org/badge/latestdoi/19829533)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-contributor%20covenant-green.svg)](http://contributor-covenant.org/version/1/4/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/pyjac)](https://pypi.org/project/pyjac/)

This utility creates source code to calculate the Jacobian matrix analytically
for a chemical reaction mechanism.

## Documentation

The full documentation for pyJac can be found at <http://slackha.github.io/pyJac/>.

## User Group

Further support can be found by [opening an issue](https://github.com/SLACKHA/pyJac/issues) on our GitHub repo.

## Installation

pyJac requires Python 3.10 or newer. Install from PyPI with pip:
```
pip install pyjac
```

Building the Python wrapper around generated code needs an extra:
```
pip install 'pyjac[pywrap]'
```

Detailed instructions are in the
[full documentation](https://slackha.github.io/pyJac/).

## Usage

pyJac installs a `pyjac` command, and can also be run as a module:
```
pyjac --lang c --input mech.yaml
python -m pyjac --lang c --input mech.yaml
```

Mechanisms may be in Chemkin format or Cantera's YAML format. The generated
source is placed in `out` by default, created if it does not exist.
Use `pyjac --help` for the full list of options.

## Theory

Theory, derivations, validation and performance testing can be found in the paper
fully describing version 1.0.2 of pyJac,
now published via <https://doi.org/10.1016/j.cpc.2017.02.004> and available
openly via [`arXiv:1605.03262 [physics.comp-ph]`](https://arxiv.org/abs/1605.03262).

## License

pyJac is released under the MIT license; see the
[LICENSE](https://github.com/slackha/pyJac/blob/main/LICENSE) for details.

If you use this package as part of a scholarly publication, please see
[CITATION.md](https://github.com/slackha/pyJac/blob/main/CITATION.md)
for the appropriate citation(s).

## Contributing

We welcome contributions to pyJac! Please see the guide to making contributions
in the [CONTRIBUTING.md](https://github.com/slackha/pyJac/blob/main/CONTRIBUTING.md)
file.

## Code of Conduct

In order to have a more open and welcoming community, pyJac adheres to a code of conduct adapted from the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

Please adhere to this code of conduct in any interactions you have in the pyJac community. It is strictly enforced on all official pyJac repositories, websites, and resources. If you encounter someone violating these terms, please let a maintainer know via email to <slackha@googlegroups.com>) and we will address it as soon as possible.

## Maintainers

Maintained by [Kyle Niemeyer](https://kyleniemeyer.com) ([@kyleniemeyer](https://github.com/kyleniemeyer)) (<kyle.niemeyer@oregonstate.edu>).
