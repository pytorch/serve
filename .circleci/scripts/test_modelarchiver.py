import os
import sys

BASE_DIR = os.getcwd()

os.chdir("model-archiver")

# Lint test
MA_DIR = os.path.join("model_archiver", ".")
RC_FILE_PATH = os.path.join(".", "model_archiver", "tests", "pylintrc")
PY_LINT_CMD = f"pylint -rn --rcfile={RC_FILE_PATH} {MA_DIR}"
PY_LINT_EXIT_CODE = os.system(PY_LINT_CMD)

# Execute python unit tests
UT_DIR = os.path.join("model_archiver", "tests", "unit_tests")
COV_DIR = os.path.join(".")
RESULTS_DIR_NAME = "result_units"
PY_UNITS_CMD = f"python -m pytest --cov-report html:{RESULTS_DIR_NAME} --cov={COV_DIR} {UT_DIR}"
PY_UNITS_EXIT_CODE = os.system(PY_UNITS_CMD)

# Execute integration tests
IT_DIR = os.path.join("model_archiver", "tests", "integ_tests")
PY_INTEG_CMD = f"python -m pytest {IT_DIR}" # ToDo - Report for Integration tests ?
PY_INTEG_EXIT_CODE = os.system(PY_INTEG_CMD)

# If any one of the steps fail, exit with error
if any( EXIT_CODE != 0 for EXIT_CODE in [PY_LINT_EXIT_CODE, PY_UNITS_EXIT_CODE, PY_INTEG_EXIT_CODE]):
    sys.exit("Model Archiver Tests Failed")