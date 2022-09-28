import os
import sys

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)


def test_modelarchiver():
    os.chdir("model-archiver")

    # Execute python unit tests
    print("## Started model archiver pytests - unit tests")
    ut_dir = os.path.join("model_archiver", "tests", "unit_tests")
    coverage_dir = os.path.join(".")
    report_output_dir = os.path.join(ut_dir, "coverage.xml")

    py_units_cmd = f"python -m pytest --cov-report xml:{report_output_dir} --cov={coverage_dir} {ut_dir}"
    print(f"## In directory: {os.getcwd()} | Executing command: {py_units_cmd}")
    py_units_exit_code = os.system(py_units_cmd)

    # Execute integration tests
    print("## Started model archiver pytests - integration tests")
    it_dir = os.path.join("model_archiver", "tests", "integ_tests")
    report_output_dir = os.path.join(it_dir, "coverage.xml")

    py_integ_cmd = f"python -m pytest --cov-report xml:{report_output_dir} --cov={coverage_dir} {it_dir}"
    print(f"## In directory: {os.getcwd()} | Executing command: {py_integ_cmd}")
    py_integ_exit_code = os.system(py_integ_cmd)

    if py_units_exit_code != 0:
        sys.exit("## Model archiver Unit Pytests Failed !")
    if py_integ_exit_code != 0:
        sys.exit("## Model archiver Integration Pytests Failed !")

    os.chdir(REPO_ROOT)
