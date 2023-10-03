# This readme covers how the default dali_image_classifier pipeline is built

Torchserve comes with a built-in pre-processing pipeline for image classification example.

Step 1: The pre-processing stages are defined in the `dali_pipeline_generation.py` file within the function wrapped with
`@dali.pipeline_def` decorator.

Step 2: Run the file

```bash
python dali_pipeline_generation.py --batch_size 2 --num_thread 1 --device_id 0 --save default.dali
```

This will generate a `default.dali` file which is a serialized dali pipeline file

:warning: The default pipeline has pre-processing stage for image_classifier example.

To add additional built-in pipelines,

1. Modify the pre-processing stages defined in the `@dali.pipeline_def` decorator function.
2. run `dali_pipeline_generation.py` and save the dali file
3. Additional keys to `model-config.yaml` and modify [dali.py](../dali.py) such that it
can load the new serialized dali file.
