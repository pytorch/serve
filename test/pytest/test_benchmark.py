import os
import subprocess
import sys
from pathlib import Path

import pytest

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parents[1]


def dependencies_available():
    try:
        import click  # noqa
        import locust  # noqa
        import locust_plugins  # noqa
    except:
        return False
    return os.system("ab -V") == 0 and os.system("locust -V") == 0


@pytest.mark.skipif(
    not dependencies_available(), reason="Dependency not found: ab tool"
)
@pytest.mark.parametrize(
    "backend",
    (
        "ab",
        "locust",
    ),
)
def test_benchmark_e2e(backend):
    report_file = Path("/tmp/benchmark/ab_report.csv")

    if report_file.exists():
        report_file.unlink()

    sys.path.append((REPO_ROOT_DIR / "benchmarks").as_posix())

    os.chdir(REPO_ROOT_DIR / "benchmarks")

    cmd = subprocess.Popen(
        f"{sys.executable} ./benchmark-ab.py --concurrency 1 --requests 10 -bb {backend} --generate_graphs True",
        shell=True,
        stdout=subprocess.PIPE,
    )
    output_lines = list(cmd.stdout)

    assert output_lines[-1].decode("utf-8") == "Test suite execution complete.\n"

    assert len(output_lines) == 71

    report = report_file.read_text()

    assert report.count(",") == 58

    report.split(",")[7] == "Requests"
    report.split("\n")[1].split(",")[7] == "10"
