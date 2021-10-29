#!/bin/bash

cd model-archiver/

# Lint test
pylint -rn --rcfile=./model_archiver/tests/pylintrc model_archiver/.
PY_LINT_EXIT_CODE=$?

# Execute python unit tests
python -m pytest --cov-report html:result_units --cov=./ model_archiver/tests/unit_tests
PY_UNITS_EXIT_CODE=$?

# Execute integration tests
python -m pytest model_archiver/tests/integ_tests # ToDo - Report for Integration tests ?
PY_INTEG_EXIT_CODE=$?

# If any one of the steps fail, exit with error
if [ "$PY_LINT_EXIT_CODE" -ne 0 ] || [ "$PY_UNITS_EXIT_CODE" -ne 0 ] || [ "$PY_INTEG_EXIT_CODE" -ne 0 ]
then exit 1
fi