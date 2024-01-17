import os
import subprocess
import sys
from pathlib import Path

import pytest

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parents[1]


@pytest.mark.skipif(os.system("ab -V") != 0, reason="Dependency not found: ab tool")
def test_benchmark_e2e():
    report_file = Path("/tmp/benchmark/ab_report.csv")

    if report_file.exists():
        report_file.unlink()

    sys.path.append((REPO_ROOT_DIR / "benchmarks").as_posix())

    os.chdir(REPO_ROOT_DIR / "benchmarks")

    cmd = subprocess.Popen(
        f"{sys.executable} ./benchmark-ab.py --concurrency 1 --requests 10",
        shell=True,
        stdout=subprocess.PIPE,
    )
    output_lines = list(cmd.stdout)

    assert output_lines[-1].decode("utf-8") == "Test suite execution complete.\n"

    assert len(output_lines) == 67

    report = report_file.read_text()

    assert report.count(",") == 58
