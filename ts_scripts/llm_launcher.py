import contextlib
import importlib
import os
import shutil
import sys
import time
from pathlib import Path

import yaml
from model_archiver import ModelArchiverConfig

from ts.launcher import start, stop

model_name = "SOME_MODEL"
model_store = "/home/ubuntu/serve/model_store"
work_dir = "/home/ubuntu/serve/data"

model_config = {
    "minWorkers": 1,
    "maxWorkers": 1,
    "maxBatchDelay": 100,
    "responseTimeout": 1200,
    "deviceType": "gpu",
    "asyncCommunication": True,
    "handler": {
        "model_path": "model/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/",
        "vllm_engine_config": {
            "enable_lora": True,
            "max_loras": 4,
            "max_cpu_loras": 4,
            "max_num_seqs": 16,
            "max_model_len": 250,
        },
        "adapters": {
            "adapter_1": "adapters/model/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/",
        },
    },
}


@contextlib.contextmanager
def model_archiver():
    loader = importlib.machinery.SourceFileLoader(
        "archiver",
        os.path.join(
            "/home/ubuntu/serve/",
            "model-archiver",
            "model_archiver",
            "model_packaging.py",
        ),
    )
    spec = importlib.util.spec_from_loader("archiver", loader)
    archiver = importlib.util.module_from_spec(spec)

    sys.modules["archiver"] = archiver

    loader.exec_module(archiver)

    yield archiver

    del sys.modules["archiver"]


@contextlib.contextmanager
def create_mar_file():
    mar_file_path = Path(model_store).joinpath(model_name)

    model_config_yaml = Path(model_store) / "model-config.yaml"
    with model_config_yaml.open("w") as f:
        yaml.dump(model_config, f)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        handler="vllm_handler",
        serialized_file=None,
        export_path=model_store,
        requirements_file=None,
        runtime="python",
        force=False,
        config_file=model_config_yaml.as_posix(),
        archive_format="no-archive",
    )

    with model_archiver() as ma:
        ma.generate_model_archive(config)

    model_config_yaml.unlink()

    assert mar_file_path.exists()

    yield mar_file_path.as_posix()

    shutil.rmtree(mar_file_path)


def main():
    """
    Register the model in torchserve
    """

    params = (
        ("model_name", model_name),
        ("url", Path(model_store) / model_name),
        ("initial_workers", "1"),
        ("synchronous", "true"),
        ("batch_size", "1"),
    )

    try:
        with create_mar_file():
            start(model_store=model_store, no_config_snapshots=True, models=model_name)

        time.sleep(10)

    finally:
        stop()


if __name__ == "__main__":
    main()
