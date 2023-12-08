import importlib.resources as pkg_resources
import os

from nvidia.dali.pipeline import Pipeline


def get_dali_pipeline(ctx):
    model_dir = ctx.system_properties.get("model_dir")

    if "dali" in ctx.model_yaml_config:
        batch_size = ctx.model_yaml_config["dali"]["batch_size"]
        num_threads = ctx.model_yaml_config["dali"]["num_threads"]
        device_id = ctx.model_yaml_config["dali"]["device_id"]
        seed = ctx.model_yaml_config["dali"]["seed"]
        if "pipeline_file" in ctx.model_yaml_config["dali"]:
            pipeline_filename = ctx.model_yaml_config["dali"]["pipeline_file"]
            pipeline_filepath = os.path.join(model_dir, pipeline_filename)
        else:
            with pkg_resources.path(
                "ts.handler_utils.preprocess.built-in", "default.dali"
            ) as file:
                pipeline_filepath = file
        if not os.path.exists(pipeline_filepath):
            raise RuntimeError("Missing dali pipeline file.")
        pipeline = Pipeline.deserialize(
            filename=pipeline_filepath,
            batch_size=batch_size,
            num_threads=num_threads,
            prefetch_queue_depth=1,
            device_id=device_id,
            seed=seed,
        )
        # pylint: disable=protected-access
        pipeline._max_batch_size = batch_size
        pipeline._num_threads = num_threads
        pipeline._device_id = device_id
        return pipeline
    else:
        raise ValueError(
            f"{ctx.model_name} has no dali config in model config yaml file"
        )
