import json
import logging
import os
from pathlib import Path

import deepspeed

from ts.context import Context


def create_checkpoints_json(model_path, checkpoints_json):
    checkpoint_files = file_list = [
        str(entry)
        for entry in Path(model_path).rglob("*.[bp][it][n]")
        if entry.is_file()
    ]
    data = {"type": "ds_model", "checkpoints": checkpoint_files, "version": 1.0}
    print(f"Creating deepspeed checkpoint file {checkpoints_json}")
    json.dump(data, open(checkpoints_json, "w"))


def get_ds_engine(model, ctx: Context):
    model_dir = ctx.system_properties.get("model_dir")
    ds_config, checkpoint = None, None
    model_path = ctx.model_yaml_config["handler"]["model_path"]

    if "deepspeed" in ctx.model_yaml_config:
        # config: the deepspeed config json file path.
        # deepspeed config parameters:
        # https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/inference/config.py
        if "config" in ctx.model_yaml_config["deepspeed"]:
            ds_config = os.path.join(
                model_dir, ctx.model_yaml_config["deepspeed"]["config"]
            )
            if not os.path.exists(ds_config):
                raise ValueError(
                    f"{ctx.model_name} has no deepspeed config file {ds_config}"
                )

        if "checkpoint" in ctx.model_yaml_config["deepspeed"]:
            checkpoint = os.path.join(
                model_dir, ctx.model_yaml_config["deepspeed"]["checkpoint"]
            )
            create_checkpoints_json(model_path, checkpoint)

        logging.debug("Creating DeepSpeed engine")
        ds_engine = deepspeed.init_inference(
            model,
            config=ds_config,
            base_dir=model_path,
            checkpoint=checkpoint,
        )
        return ds_engine
    else:
        raise ValueError(
            f"{ctx.model_name} has no deepspeed config in model config yaml file"
        )
