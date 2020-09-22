import os
import sys

# Lint Test
TS_DIR = os.path.join("ts", ".")
RC_FILE_PATH = os.path.join(".", "ts", "tests", "pylintrc")
PY_LINT_CMD = f"pylint -rn --rcfile={RC_FILE_PATH} {TS_DIR}"
PY_LINT_EXIT_CODE = os.system(PY_LINT_CMD)

# Execute python tests
TEST_DIR = os.path.join("ts", "tests", "unit_tests")
COV_DIR = os.path.join("ts")
RESULTS_DIR_NAME = "result_units"
PY_TEST_CMD = f"python -m pytest --cov-report html:{RESULTS_DIR_NAME} --cov={COV_DIR} {TEST_DIR}"
PY_EXIT_CODE = os.system(PY_TEST_CMD)

# If any one of the steps fail, exit with error
if any( EXIT_CODE != 0 for EXIT_CODE in [PY_LINT_EXIT_CODE, PY_EXIT_CODE]):
    sys.exit("TorchServe Python Tests Failed")