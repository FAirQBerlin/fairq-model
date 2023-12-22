#! /bin/bash

# Redirect stdout to stderr so we can see verbose output:
pytest --verbose --durations=0 --junit-xml test-reports/results.xml 1>&2 || exit 1

# Cat xml file to stdout so the caller can save the output to a file on host system:
cat test-reports/results.xml
