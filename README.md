# The official repository has moved!

The official Wyrm repository has a new home at the [BBCI github account][bbci].
Please use the new one for forks, issues, etc.

  [bbci] https://github.com/bbci/wyrm


# Wyrm

Wyrm is a Brain Computer Interface (BCI) toolbox written in Python. Wyrm is
suitable for running on-line BCI experiments as well as off-line analysis of EEG
data.

Online documentation is available [here][wyrmdoc].

  [wyrmdoc]: http://bbci.github.io/wyrm


## Installation

### Using git

Use distutils to install Wyrm into your `PYTHONPATH`:

```bash
$ git clone http://github.com/bbci/wyrm
$ cd wyrm
$ python setup.py install --user
```

this will always give you the latest development version of Wyrm.


### Using PyPI

Wyrm is also available on the [Python Package Index (PyPI)][pypi] and can be
easily installed via:

```bash
$ pip install wyrm
```

  [pypi]: https://pypi.python.org/pypi/Wyrm


## Examples

In the `examples` directory, you'll find, among others, examples for various BCI
tasks using publicly available BCI datasets from the [BCI Competition][bcicomp].

* An example for classification of motor imagery in ECoG recordings. For that
  example the [BCI Competition3, Data Set 1][bcicomp3ds1] was used.

* An example for classification with a P300 Matrix Speller in EEG recordings.
  The [BCI Competition 3, Data Set 2][bcicomp3ds2] was used for that example.

You can follow those examples by downloading the data and copying the files to
the appropriate places.


  [bcicomp]: http://www.bbci.de/competition
  [bcicomp3ds1]: http://www.bbci.de/competition/iii/#data_set_i
  [bcicomp3ds2]: http://www.bbci.de/competition/iii/#data_set_ii


## Python 3 Support

Wyrm is mainly developed under Python 2.7, however since people will eventually
move on to Python 3 we try to be forward compatible by writing the code in a way
that it runs on Python 2 and -3.

[![Build Status](https://travis-ci.org/venthur/wyrm.png)](https://travis-ci.org/venthur/wyrm)

Whenever a new version of Wyrm is pushed to github, the [Travis continuous
integration service][travisci] will run Wyrm's whole test suite with Python 2.7,
3.3, and 3.4. If and only if all three test suites pass, the build is shown as
"passing".

  [travisci]: https://travis-ci.org/venthur/wyrm


## Related Software

For a complete BCI system written in Python use Wyrm together with
[Mushu][mushu] and [Pyff][pyff]. Mushu is a BCI signal acquisition and Pyff a
BCI feedback and -stimulus framework.

  [pyff]: http://github.com/venthur/pyff
  [mushu]: http://github.com/venthur/mushu

