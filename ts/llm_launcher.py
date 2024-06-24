import argparse
import contextlib
import shutil
from pathlib import Path
from signal import pause

import torch
import yaml
from model_archiver import ModelArchiverConfig
from model_archiver.model_packaging import generate_model_archive

from ts.launcher import start, stop


def get_model_config(args):
    download_dir = getattr(args, "vllm_engine.download_dir")
    download_dir = (
        Path(download_dir).resolve().as_posix() if download_dir else download_dir
    )

    model_config = {
        "minWorkers": 1,
        "maxWorkers": 1,
        "batchSize": 1,
        "maxBatchDelay": 100,
        "responseTimeout": 1200,
        "deviceType": "gpu",
        "asyncCommunication": True,
        "parallelLevel": torch.cuda.device_count() if torch.cuda.is_available else 1,
        "handler": {
            "model_path": args.model_id,
            "vllm_engine_config": {
                "enable_lora": True,
                "max_loras": 4,
                "max_cpu_loras": 4,
                "max_num_seqs": getattr(args, "vllm_engine.max_num_seqs"),
                "max_model_len": getattr(args, "vllm_engine.max_model_len"),
                "download_dir": download_dir,
            }
            # ,
            # "adapters": {
            #     "adapter_1": "adapters/model/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/",
            # },
        },
    }
    return model_config


@contextlib.contextmanager
def create_mar_file(args):
    model_store_path = Path(args.model_store)
    model_store_path.mkdir(parents=True, exist_ok=True)

    mar_file_path = model_store_path / args.model_name

    model_config_yaml = Path(args.model_store) / "model-config.yaml"
    with model_config_yaml.open("w") as f:
        yaml.dump(get_model_config(args), f)

    config = ModelArchiverConfig(
        model_name=args.model_name,
        version="1.0",
        handler="vllm_handler",
        serialized_file=None,
        export_path=args.model_store,
        requirements_file=None,
        runtime="python",
        force=False,
        config_file=model_config_yaml.as_posix(),
        archive_format="no-archive",
    )

    generate_model_archive(config)

    model_config_yaml.unlink()

    assert mar_file_path.exists()

    yield mar_file_path.as_posix()

    shutil.rmtree(mar_file_path)


def main(args):
    """
    Register the model in torchserve
    """

    with create_mar_file(args):
        try:
            start(
                model_store=args.model_store,
                no_config_snapshots=True,
                models=args.model_name,
            )

            pause()

        except KeyboardInterrupt:
            pass
        finally:
            stop(wait=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="model",
        help="Model name",
    )

    parser.add_argument(
        "--model_store",
        type=str,
        default="model_store",
        help="Model store",
    )

    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model id",
    )

    parser.add_argument(
        "--vllm_engine.max_num_seqs",
        type=int,
        default=16,
        help="Max sequences in vllm engine",
    )

    parser.add_argument(
        "--vllm_engine.max_model_len",
        type=int,
        default=None,
        help="Model context length",
    )

    parser.add_argument(
        "--vllm_engine.download_dir",
        type=str,
        default="/data",
        help="Cache dir",
    )

    args = parser.parse_args()

    main(args)
