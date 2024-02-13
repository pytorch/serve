import os
import subprocess
import sys
from pathlib import Path

import pytest
from model_archiver import ModelArchiverConfig

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


@pytest.fixture
def mar_file(tmp_path, model_archiver):
    model_name = "gpt2"
    mar_file_path = tmp_path / model_name

    model_config_yaml = CURR_FILE_PATH.joinpath(
        "test_data", "streaming", "model_config.yaml"
    ).as_posix()
    model_file = CURR_FILE_PATH.joinpath(
        "test_data", "streaming", "fake_streaming_model.py"
    ).as_posix()
    handler_file = CURR_FILE_PATH.joinpath(
        "test_data", "streaming", "stream_handler.py"
    ).as_posix()

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        model_file=model_file,
        handler=handler_file,
        serialized_file=None,
        export_path=tmp_path.as_posix(),
        requirements_file=None,
        runtime="python",
        force=False,
        config_file=model_config_yaml,
        extra_files=None,
        archive_format="no-archive",
    )

    model_archiver.generate_model_archive(config)

    assert mar_file_path.exists()

    print(mar_file_path)

    yield mar_file_path.as_posix()

    # Clean up files
    mar_file_path.unlink()


def test_llm_benchmark_e2e(mar_file, tmp_path):
    report_file = Path("/tmp/benchmark/llm_report.csv")

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Hello, ")

    if report_file.exists():
        report_file.unlink()

    sys.path.append((REPO_ROOT_DIR / "benchmarks").as_posix())

    os.chdir(REPO_ROOT_DIR / "benchmarks")

    cmd = subprocess.Popen(
        f"{sys.executable} ./benchmark-ab.py --concurrency 10 --requests 10 --llm_mode -u {mar_file} -i {prompt_file.as_posix()} ",
        shell=True,
        stdout=subprocess.PIPE,
    )

    output_lines = list(cmd.stdout)

    assert output_lines[-1].decode("utf-8") == "Test suite execution complete.\n"

    report = report_file.read_text()

    assert report.count(",") == 54

    report.split(",")[12] == "Requests"
    report.split("\n")[1].split(",")[12] == "10"

    print(output_lines)
