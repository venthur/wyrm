# Wyrm

Wyrm is a Brain Computer Interface (BCI) toolbox written in Python. Wyrm is
suitable for running on-line BCI experiments as well as off-line analysis of EEG
data.

Online documentation is available [here][wyrmdoc].

  [wyrmdoc]: http://venthur.github.io/wyrm


[![Build Status](https://travis-ci.org/venthur/wyrm.png)](https://travis-ci.org/venthur/wyrm)


## Installation

Use distutils to install Wyrm into your `PYTHONPATH`:

```bash
$ git clone http://github.com/venthur/wyrm
$ cd wyrm
$ python setyp.py install --user
```

this will always give you the latest development version of Wyrm. A slightly
outdated but maybe more stable version can be easily installed via:

```bash
$ pip install wyrm
```


## Examples

In the `examples` directory, you'll find, among others, examples for various BCI
tasks using publicly available BCI datasets from the [BCI Competition][bcicomp].

* An example for classification of motor imagery in ECoG recordings. For that
  example the [BCI Competition3, Data Set 1][bcicompds1] was used.

* An example for classification with a P300 Matrix Speller in EEG recordings.
  The [BCI Competition 3, Data Set 2][bcicomp3ds2] was used for that example.

You can follow those examples by downloading the data and copying the files to
the appropriate places.


  [bcicomp]: http://www.bbci.de/competition
  [bcicomp3ds1]: http://www.bbci.de/competition/iii/#data_set_i
  [bcicomp3ds2]: http://www.bbci.de/competition/iii/#data_set_ii


## Related Software

For a complete BCI system written in Python use Wyrm together with
[Mushu][mushu] and [Pyff][pyff]. Mushu is a BCI signal acquisition and Pyff a
BCI feedback and -stimulus framework.

  [pyff]: http://github.com/venthur/pyff
  [mushu]: http://github.com/venthur/mushu

