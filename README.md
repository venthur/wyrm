Wyrm
====

Wyrm is a Brain Computer Interface (BCI) toolbox written in Python. Wyrm is
suitable for running on-line BCI experiments as well as off-line analysis of EEG
data.

Online documentation is available [here][wyrmdoc].

  [wyrmdoc]: http://venthur.github.io/wyrm


[![Build Status](https://travis-ci.org/venthur/wyrm.png)](https://travis-ci.org/venthur/wyrm)


Installation
------------

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


Examples
--------

In `wyrm/examples` you'll find an example for classification with a P300 Matrix
Speller. The data used for that experiment is from the [BCI Competition 3, Data
Set 2][bcicomp3ds2]. If you want to run that experiment on your machine, you
need to download the data, and follow the instructions in the example script.

  [bcicomp3ds2]: http://www.bbci.de/competition/iii/#data_set_ii


Related Software
----------------

For a complete BCI system written in Python use Wyrm together with
[Mushu][mushu] and [Pyff][pyff]. Mushu is a BCI signal acquisition and Pyff a
BCI feedback and -stimulus framework.

  [pyff]: http://github.com/venthur/pyff
  [mushu]: http://github.com/venthur/mushu

