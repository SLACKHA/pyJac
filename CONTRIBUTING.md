# Contributing

We welcome contributions in the form of bug reports, bug fixes, improvements to the documentation, ideas for enhancements (or the enhancements themselves!).

You can find a [list of current issues](https://github.com/slackha/pyJac/issues) in the project's GitHub repo. Feel free to tackle any existing bugs or enhancement ideas by submitting a [pull request](https://github.com/slackha/pyJac/pulls).

## Bug Reports

 * Please include a short (but detailed) Python snippet or explanation for reproducing the problem. Attach or include a link to any input files (e.g., reaction mechanism) that will be needed to reproduce the error.
 * Explain the behavior you expected, and how what you got differed.

## Pull Requests

 * Please reference relevant GitHub issues in your commit message using `GH123` or `#123`.
 * Changes should be [PEP8](http://www.python.org/dev/peps/pep-0008/) compatible.
 * Keep style fixes to a separate commit to make your pull request more readable.
 * Docstrings are required and should follow the [NumPy style](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy).
 * When you start working on a pull request, start by creating a new branch pointing at the latest commit on [GitHub master](https://github.com/slackha/pyJac/tree/master).
 * The pyJac copyright policy is detailed in the [`LICENSE`](https://github.com/slackha/pyJac/blob/master/LICENSE).

## Tests

Since pyJac (unfortunately) does not currently employ unit tests, functional testing is somewhat involved. However, the necessary framework is available in `pyjac.functional_tester`, and we ask that pull requests demonstrate that the changes do not invalidate these tests.

We also enthusiastically welcome contributions in the form of unit tests!

## Meta

Thanks to the useful [contributing guide of pyrk](https://github.com/pyrk/pyrk/blob/master/CONTRIBUTING.md), which served as an inspiration and starting point for this guide.
