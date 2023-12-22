#! /bin/bash

# Redirect stdout to stderr so we can see verbose output:
py.test --cov-report=xml:test-reports/coverage.xml --cov=. 1>&2 || exit 1

# Cat xml file to stdout so the caller can save the output to a file on host system:
cat test-reports/coverage.xml
