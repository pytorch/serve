#!/bin/bash

# Lint Test
pylint -rn --rcfile=./ts/tests/pylintrc ts/.
PY_LINT_EXIT_CODE=$?

# Execute python tests
python -m pytest --cov-report html:result_units --cov=ts/ ts/tests/unit_tests/
PYTEST_EXIT_CODE=$?

# If any one of the tests fail, exit with error
if [ "$PY_LINT_EXIT_CODE" -ne 0 ] || [ "$PYTEST_EXIT_CODE" -ne 0 ]
then exit 1
fi