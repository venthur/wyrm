#!/bin/sh


pprint() {
    echo
    echo ============================================================
    echo $@
    echo ============================================================
    echo
}

pprint Running Pyflakes
PYFLAKES_NODOCTEST=1 pyflakes .

pprint Running Unittests
coverage run --branch --source=wyrm --module unittest discover test

pprint Coverage Report
coverage report
