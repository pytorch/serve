import os
import sys


REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)


def test_modelarchiver():
    os.chdir("model-archiver")

    # Lint test
    print("## Started model archiver linting")
    ma_dir = os.path.join("model_archiver", ".")
    rc_file_path = os.path.join(".", "model_archiver", "tests", "pylintrc")
    py_lint_cmd = f"pylint -rn --rcfile={rc_file_path} {ma_dir}"
    print(f"## In directory: {os.getcwd()} | Executing command: {py_lint_cmd}")
    py_lint_exit_code = os.system(py_lint_cmd)

    # Execute python unit tests
    print("## Started model archiver pytests - unit tests")
    ut_dir = os.path.join("model_archiver", "tests", "unit_tests")
    coverage_dir = os.path.join(".")
    results_dir_name = "result_units"
    py_units_cmd = f"python -m pytest --cov-report html:{results_dir_name} --cov={coverage_dir} {ut_dir}"
    print(f"## In directory: {os.getcwd()} | Executing command: {py_units_cmd}")
    py_units_exit_code = os.system(py_units_cmd)

    # Execute integration tests
    print("## Started model archiver pytests - integration tests")
    it_dir = os.path.join("model_archiver", "tests", "integ_tests")
    py_integ_cmd = f"python -m pytest {it_dir}" # ToDo - Report for Integration tests ?
    print(f"## In directory: {os.getcwd()} | Executing command: {py_integ_cmd}")
    py_integ_exit_code = os.system(py_integ_cmd)

    # If any one of the steps fail, exit with error
    if py_lint_exit_code != 0:
        sys.exit("## Model archiver Linting Failed !")
    if py_units_exit_code != 0:
        sys.exit("## Model archiver Unit Pytests Failed !")
    if py_integ_exit_code != 0:
        sys.exit("## Model archiver Integration Pytests Failed !")

    os.chdir(REPO_ROOT)
