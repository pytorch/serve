import os
import subprocess
import sys


def test_torchserve():
    # Lint Test
    print("## Started torchserve linting")
    ts_dir = os.path.join("ts", ".")

    # Execute python tests
    print("## Started torchserve pytests")
    test_dir = os.path.join("ts", "tests", "unit_tests")
    handler_test_dir = os.path.join("ts", "torch_handler", "unit_tests")
    coverage_dir = os.path.join("ts")
    report_output_dir = os.path.join(test_dir, "coverage.xml")

    ts_test_cmd = [
        "python", "-m", "pytest",
        "--cov-report", f"xml:{report_output_dir}",
        "--cov", coverage_dir,
        test_dir,
        handler_test_dir
    ]
    print(f"## In directory: {os.getcwd()} | Executing command: {ts_test_cmd}")
    try:
        subprocess.run(ts_test_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

