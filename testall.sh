#!/bin/sh



if command -v coverage >/dev/null
then 
    COVERAGE=ccoverage
elif command -v python-coverage >/dev/null
then
    COVERAGE=python-coverage
else
    echo "No Python Coverage found, aborting."
    exit 1
fi

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
$COVERAGE run --branch --source=wyrm --module unittest discover test

pprint Coverage Report
$COVERAGE report

