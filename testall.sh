#!/bin/sh


pprint() {
    echo
    echo ============================================================
    echo $@
    echo ============================================================
    echo
}

pprint Running Pyflakes
pyflakes .

pprint Running Unittests
coverage run --source=wyrm --module unittest discover test

pprint Coverage Report
coverage report
