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
    results_dir_name = "result_units"
    ts_test_cmd = f"python -m pytest --cov-report xml:{results_dir_name}.xml --cov={coverage_dir} {test_dir}"
    print(f"## In directory: {os.getcwd()} | Executing command: {ts_test_cmd}")
    ts_test_cmd = os.system(ts_test_cmd)

    if ts_test_cmd != 0:
        sys.exit("## TorchServe Pytests Failed !")
