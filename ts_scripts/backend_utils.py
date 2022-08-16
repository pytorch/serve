import os
import sys


def test_torchserve():
    # Lint Test
    print("## Started torchserve linting")
    ts_dir = os.path.join("ts", ".")

    # Execute python tests
    print("## Started torchserve pytests")
    test_dir = os.path.join("ts", "tests", "unit_tests")
    coverage_dir = os.path.join("ts")
    report_output_dir = os.path.join(test_dir, "coverage.xml")

    ts_test_cmd = f"python -m pytest --cov-report xml:{report_output_dir} --cov={coverage_dir} {test_dir}"
    print(f"## In directory: {os.getcwd()} | Executing command: {ts_test_cmd}")
    ts_test_error_code = os.system(ts_test_cmd)

    if ts_test_error_code != 0:
        sys.exit("## TorchServe Pytests Failed !")
