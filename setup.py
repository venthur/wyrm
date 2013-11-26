#!/usr/bin/env python


from distutils.core import setup

import wyrm


setup(
    name='Wyrm',
    version=wyrm.__version__,
    description='Toolbox for Brain Computer Interfacing Experiments.',
    author='Bastian Venthur',
    author_email='bastian.venthur@tu-berlin.de',
    url='http://github.com/venthur/wyrm/',
    packages=['wyrm'],
)
