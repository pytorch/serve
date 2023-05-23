import logging
import os

import deepspeed

from ts.context import Context


def get_ds_engine(model, ctx: Context):
    model_dir = ctx.system_properties.get("model_dir")
    ds_config = None
    checkpoint = None
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

        if "checkpoint" in ctx.model_yaml_config:
            checkpoint = os.path.join(
                model_dir, ctx.model_yaml_config["deepspeed"]["checkpoint"]
            )
            if not os.path.exists(checkpoint):
                raise ValueError(
                    f"{ctx.model_name} has no deepspeed checkpoint file {checkpoint}"
                )
        logging.debug("Creating DeepSpeed engine")
        ds_engine = deepspeed.init_inference(
            model, config=ds_config, checkpoint=checkpoint
        )
        return ds_engine
    else:
        raise ValueError(
            f"{ctx.model_name} has no deepspeed config in model config yaml file"
        )
