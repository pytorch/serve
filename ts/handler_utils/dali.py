import os

from nvidia.dali.pipeline import Pipeline


def get_dali_pipeline(self, ctx):
    self.manifest = ctx.manifest
    self.model_dir = properties.get("model_dir")
    properties = ctx.system_properties
    self.device = self.local_rank

    if "dali" in ctx.model_yaml_config:
        self.batch_size = ctx.model_yaml_config["dali"]["batch_size"]
        self.num_threads = ctx.model_yaml_config["dali"]["num_threads"]
        self.device_id = ctx.model_yaml_config["dali"]["device_id"]
        self.seed = ctx.model_yaml_config["dali"]["seed"]
        pipeline_filename = ctx.model_yaml_config["dali"]["pipeline_file"]
        pipeline_filepath = os.path.join(self.model_dir, pipeline_filename)
        if not os.path.exists(pipeline_filepath):
            raise RuntimeError("Missing dali pipeline file.")
        self.pipeline = Pipeline.deserialize(
            filename=pipeline_filepath,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            prefetch_queue_depth=1,
            device_id=self.device_id,
            seed=self.seed,
        )
        # pylint: disable=protected-access
        self.pipeline._max_batch_size = self.batch_size
        self.pipeline._num_threads = self.num_threads
        self.pipeline._device_id = self.device_id
        return self.pipeline
    else:
        raise ValueError(
            f"{ctx.model_name} has no dali config in model config yaml file"
        )
