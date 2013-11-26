Wyrm
====

[![Build Status](https://travis-ci.org/venthur/wyrm.png)](https://travis-ci.org/venthur/wyrm)

About
=====

Wyrm is a Brain Computer Interface (BCI) toolbox written in Python.

Using it
========

Installation
------------

Use distutils to install Wyrm into your `PYTHONPATH`:

```bash
$ git clone http://github.com/venthur/wyrm
$ cd wyrm
$ python setyp.py install --user

this will always give you the latest development version of Wyrm. A slightly
outdated but maybe more stable version can be easily installed via:

```bash
$ pip install wyrm
```

Online Documentation
--------------------

Online documentation is available [here][wyrmdoc].

  [wyrmdoc]: http://venthur.github.io/wyrm

Running it
----------

Please refer to the examples in wyrm/examples.


Related Software
================

For a complete BCI system written in Python use Wyrm together with
[Mushu][mushu] and [Pyff][pyff]. Mushu is a BCI signal acquisition and Pyff a
BCI feedback and -stimulus framework.

  [pyff]: http://github.com/venthur/pyff
  [mushu]: http://github.com/venthur/mushu

