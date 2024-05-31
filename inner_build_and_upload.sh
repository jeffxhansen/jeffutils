#!/bin/bash

# this shell script builds the package and uploads it to PyPI

rm -rf dist
python3 -m build
twine upload dist/* --verbose