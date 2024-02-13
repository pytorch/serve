import os
import shutil
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
def test_benchmark_e2e(backend, tmp_path):
    report_file = Path(f"/{tmp_path}/benchmark/ab_report.csv")

    sys.path.append((REPO_ROOT_DIR / "benchmarks").as_posix())

    os.chdir(REPO_ROOT_DIR / "benchmarks")

    cmd = subprocess.Popen(
        f"{sys.executable} ./benchmark-ab.py --concurrency 1 --requests 10 -bb {backend} --generate_graphs True --report_location {tmp_path} --tmp_dir {tmp_path}",
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

    yield mar_file_path.as_posix()

    # Clean up files
    shutil.rmtree(mar_file_path)


def test_llm_benchmark_e2e(mar_file, tmp_path):
    benchmark_dir = tmp_path / "benchmark"

    report_file = Path(f"/{benchmark_dir}/benchmark/llm_report.csv")

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Hello, ")

    if report_file.exists():
        report_file.unlink()

    sys.path.append((REPO_ROOT_DIR / "benchmarks").as_posix())

    os.chdir(REPO_ROOT_DIR / "benchmarks")

    cmd = subprocess.Popen(
        f"{sys.executable} ./benchmark-ab.py --concurrency 10 --requests 10 --llm_mode -u {mar_file} -i {prompt_file.as_posix()}  --report_location {benchmark_dir} --tmp_dir {benchmark_dir}",
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
