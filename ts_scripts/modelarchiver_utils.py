import os
import sys
import subprocess

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)


def test_modelarchiver():
    os.chdir("model-archiver")

    # Execute python unit tests
    print("## Started model archiver pytests - unit tests")
    ut_dir = os.path.join("model_archiver", "tests", "unit_tests")
    coverage_dir = os.path.join(".")
    report_output_dir = os.path.join(ut_dir, "coverage.xml")

    py_units_cmd = [
        "python", "-m", "pytest", 
        f"--cov-report=xml:{report_output_dir}", 
        f"--cov={coverage_dir}", 
        ut_dir
    ]
    print(f"## In directory: {os.getcwd()} | Executing command: {py_units_cmd}")
    try:
        subprocess.run(
            py_units_cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        sys.exit("## Model archiver Unit Pytests Failed !")

    # Execute integration tests
    print("## Started model archiver pytests - integration tests")
    it_dir = os.path.join("model_archiver", "tests", "integ_tests")
    report_output_dir = os.path.join(it_dir, "coverage.xml")

    py_integ_cmd = [
        "python", "-m", "pytest", 
        f"--cov-report=xml:{report_output_dir}", 
        f"--cov={coverage_dir}", 
        it_dir
    ]
    print(f"## In directory: {os.getcwd()} | Executing command: {py_integ_cmd}")
    try:
        subprocess.run(
            py_integ_cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        sys.exit("## Model archiver Integration Pytests Failed !")

    os.chdir(REPO_ROOT)
