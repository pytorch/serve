#!/bin/bash

ARTIFACTS_DIR='run_artifacts'
RESULT_DIR=$ARTIFACTS_DIR'/report/performance/'
JMETER_PATH='/opt/apache-jmeter-5.3/bin/jmeter'

cd test/performance

# Set LANG env variable required by click
# https://click.palletsprojects.com/en/5.x/python3/
export LANG=C.UTF-8

# Install dependencies
pip install -r requirements.txt
pip install bzt

# Execute performance test suite and store exit code
python run_performance_suite.py -j $JMETER_PATH -e xlarge -x example* --no-compare-local --no-monit
EXIT_CODE=$?

# Collect and store test results in result directory to be picked up by CircleCI
mkdir -p $RESULT_DIR
cp $ARTIFACTS_DIR/**/performance_results.xml $RESULT_DIR

# Exit with the same error code as that of test execution
exit $EXIT_CODE
