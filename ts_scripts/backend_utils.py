import os
import sys


def test_torchserve():
    # Lint Test
    print("## Started torchserve linting")
    ts_dir = os.path.join("ts", ".")
    rc_file_path = os.path.join(".", "ts", "tests", "pylintrc")
    py_lint_cmd = f"pylint -rn --rcfile={rc_file_path} {ts_dir}"
    print(f"## In directory: {os.getcwd()} | Executing command: {py_lint_cmd}")
    py_lint_exit_code = os.system(py_lint_cmd)

    # Execute python tests
    print("## Started torchserve pytests")
    test_dir = os.path.join("ts", "tests", "unit_tests")
    coverage_dir = os.path.join("ts")
    results_dir_name = "result_units"
    py_test_cmd = f"python -m pytest --cov-report html:{results_dir_name} --cov={coverage_dir} {test_dir}"
    print(f"## In directory: {os.getcwd()} | Executing command: {py_test_cmd}")
    py_test_exit_code = os.system(py_test_cmd)

    # If any one of the steps fail, exit with error
    if py_lint_exit_code != 0:
        sys.exit("## TorchServe Linting Failed !")
    if py_test_exit_code != 0:
        sys.exit("## TorchServe Pytests Failed !")
