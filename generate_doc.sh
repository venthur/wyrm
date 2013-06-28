#!/bin/sh

DOCDIR=doc/
APIDOCDIR=$DOCDIR/api
SRCDIR=wyrm/


rm -rf $APIDOCDIR
mkdir -p $APIDOCDIR
sphinx-apidoc -o $APIDOCDIR $SRCDIR
cd $DOCDIR
make html

