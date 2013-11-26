#!/usr/bin/env python


from distutils.core import setup

import wyrm


setup(
    name='Wyrm',
    version=wyrm.__version__,
    description='Toolbox for Brain Computer Interfacing Experiments.',
    long_description='A Python toolbox of on-line BCI experiments and off-line BCI data analysis.',
    author='Bastian Venthur',
    author_email='bastian.venthur@tu-berlin.de',
    url='http://github.com/venthur/wyrm/',
    download_url='http://github.com/venthur/wyrm/',
    license='GPL2',
    platform='any',
    packages=['wyrm'],
)
