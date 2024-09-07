import argparse
import contextlib
import os
import shutil
import subprocess
from pathlib import Path
from signal import pause

import torch
import yaml
from model_archiver import ModelArchiverConfig
from model_archiver.model_packaging import generate_model_archive

from ts.launcher import start, stop
from ts.utils.hf_utils import download_model


def create_tensorrt_llm_engine(model_store, model_name, dtype, snapshot_path):
    if not os.path.exists("TensorRT-LLM"):
        subprocess.run(["git", "clone", "https://github.com/NVIDIA/TensorRT-LLM.git"])
    if not os.path.exists(f"{model_store}/{model_name}/tllm_checkpoint_1gpu_bf16"):
        subprocess.run(
            [
                "python",
                "TensorRT-LLM/examples/llama/convert_checkpoint.py",
                "--model_dir",
                snapshot_path,
                "--output_dir",
                f"{model_store}/{model_name}/tllm_checkpoint_1gpu_bf16",
                "--dtype",
                dtype,
            ]
        )
    if not os.path.exists(f"{model_store}/{model_name}/{model_name}-engine"):
        subprocess.run(
            [
                "trtllm-build",
                "--checkpoint_dir",
                f"{model_store}/{model_name}/tllm_checkpoint_1gpu_bf16",
                "--gemm_plugin",
                dtype,
                "--gpt_attention_plugin",
                dtype,
                "--output_dir",
                f"{model_store}/{model_name}/{model_name}-engine",
            ]
        )


def get_model_config(args, model_snapshot_path=None):
    model_config = {
        "minWorkers": 1,
        "maxWorkers": 1,
        "batchSize": 1,
        "maxBatchDelay": 100,
        "responseTimeout": 1200,
        "deviceType": "gpu",
        "asyncCommunication": True,
    }

    if args.engine == "vllm":
        download_dir = getattr(args, "vllm_engine.download_dir")
        download_dir = (
            Path(download_dir).resolve().as_posix() if download_dir else download_dir
        )

        model_config.update(
            {
                "parallelLevel": torch.cuda.device_count()
                if torch.cuda.is_available
                else 1,
                "handler": {
                    "model_path": args.model_id,
                    "vllm_engine_config": {
                        "max_num_seqs": getattr(args, "vllm_engine.max_num_seqs"),
                        "max_model_len": getattr(args, "vllm_engine.max_model_len"),
                        "download_dir": download_dir,
                        "tensor_parallel_size": torch.cuda.device_count()
                        if torch.cuda.is_available
                        else 1,
                    },
                },
            }
        )

        if hasattr(args, "lora_adapter_ids"):
            raise NotImplementedError("Lora setting needs to be implemented")
            lora_adapter_ids = args.lora_adapter_ids.split(";")

            model_config["handler"]["vllm_engine_config"].update(
                {
                    "enable_lora": True,
                }
            )

    elif args.engine == "trt_llm":
        model_config.update(
            {
                "handler": {
                    "tokenizer_dir": os.path.join(os.getcwd(), model_snapshot_path),
                    "trt_llm_engine_config": {
                        "engine_dir": f"{args.model_name}-engine",
                    },
                }
            }
        )
    else:
        raise RuntimeError("Unsupported LLM Engine")

    return model_config


@contextlib.contextmanager
def create_mar_file(args, model_snapshot_path=None):
    mar_file_path = Path(args.model_store) / args.model_name

    model_config_yaml = Path(args.model_store) / "model-config.yaml"
    with model_config_yaml.open("w") as f:
        yaml.dump(get_model_config(args, model_snapshot_path), f)

    config = ModelArchiverConfig(
        model_name=args.model_name,
        version="1.0",
        handler=f"{args.engine}_handler",
        serialized_file=None,
        export_path=args.model_store,
        requirements_file=None,
        runtime="python",
        force=True,
        config_file=model_config_yaml.as_posix(),
        archive_format="no-archive",
    )

    if not os.path.exists(mar_file_path):
        generate_model_archive(config)

    model_config_yaml.unlink()

    assert mar_file_path.exists()

    yield mar_file_path.as_posix()

    if args.engine == "vllm":
        shutil.rmtree(mar_file_path)


def main(args):
    """
    Register the model in torchserve
    """

    model_store_path = Path(args.model_store)
    model_store_path.mkdir(parents=True, exist_ok=True)
    if args.engine == "trt_llm":
        model_snapshot_path = download_model(args.model_id)

    with create_mar_file(args, model_snapshot_path):
        if args.engine == "trt_llm":
            create_tensorrt_llm_engine(
                args.model_store, args.model_name, args.dtype, model_snapshot_path
            )
        try:
            start(
                model_store=args.model_store,
                no_config_snapshots=True,
                models=args.model_name,
                disable_token=args.disable_token_auth,
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
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model id",
    )

    parser.add_argument(
        "--disable_token_auth",
        action="store_true",
        help="Disable token authentication",
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
        default=None,
        help="Cache dir",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        help="LLM engine",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type",
    )
    args = parser.parse_args()

    main(args)
