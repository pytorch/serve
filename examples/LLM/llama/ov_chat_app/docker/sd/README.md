
# Accelerating StableDiffusionXL model with torch.compile OpenVINO backend

[Stable Diffusion XL ](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl) is a image generation model that is geared towards generating more photorealistic images to its predecessors. This guide details the process of enhancing model performance using the torch.compile with the OpenVINO backend, specifically tested on Intel Xeon Platinum 8469 CPU and Intel GPU Flex 170.


### Prerequisites
- `PyTorch >= 2.1.0`
- `OpenVINO >= 2024.1.0` . Install the latest version as shown below:

```bash
cd examples/pt2/torch_compile_openvino/stable_diffusion
pip install -r requirements.txt
```

## Workflow
1. Configure torch.compile.
1. Create Model Archive.
1. Start TorchServe.
1. Run Inference.
1. Stop TorchServe.
1. Measure and Compare Performance with different backends and devices.

### 1. Configure torch.compile

`torch.compile` allows various configurations that can influence performance outcomes. Explore different options in the [official PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.compile.html) and the [OpenVINO backend documentation](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html).

In this example, we utilize the configuration defined in [model-config.yaml](./model-config.yaml). Specifically, the OpenVINO backend is enabled to optimize performance, as shown in the configuration snippet:
```yaml
pt2: {backend: 'openvino'}
```

#### Additional Configuration Options:
- If you want to measure the handler `preprocess`, `inference`, `postprocess` times, include `profile: true` in the handler section of the config:

```bash
echo "    profile: true" >> model-config.yaml
```

- `torch.compile` OpenVINO backend supports additional configurations for model caching, device selection, and other OpenVINO specific options. Refer to the [torch.compile OpenVINO options documentation](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html#options). For example, see the following configuration sample:

```yaml
pt2: {backend: 'openvino', options: {'model_caching' : True, 'device': 'GPU'}}
```

- `model_caching`: Enables caching the model after the initial run, reducing the first inference latency for subsequent runs of the same model.
- `device`: Specifies the hardware device for running the application.

### 2. Create Model Archive

Download Stable diffusion model and prepare the Model Archive using [model-config.yaml](./model-config.yaml) configuration.

```bash
# Download the Stable diffusion model. Saves it to the Base_Diffusion_model directory.
python Download_model.py

# Create model archive
torch-model-archiver --model-name diffusion_fast --version 1.0 --handler stable_diffusion_handler.py \
    --config-file model-config.yaml --extra-files "./pipeline_utils.py" --archive-format no-archive

mv Base_Diffusion_model diffusion_fast/

# Add the model archive to model store
mkdir model_store
mv diffusion_fast model_store
```

### 3. Start torchserve

Start the TorchServe server using the following command:

```bash
torchserve --start --ts-config config.properties --model-store model_store --models diffusion_fast
```

### 4. Run inference

Execute the model using the following command to generate an image based on your specified prompt:

```bash
python query.py --url "http://localhost:8080/predictions/diffusion_fast" --prompt "a photo of an astronaut riding a horse on mars"
```

By default, the generated image is saved to a file named `output-<timestamp>.jpg`. You can customize the output filename by using the `--filename` parameter in the `query.py` script.


### 5. Stop the server
Stop TorchServe with the following command:

```bash
torchserve --stop
```

### 6. Measure and Compare Performance with different backends

Following the steps outlined in the previous section, you can compare the inference times for Inductor backend and OpenVINO backend:

1. Update model-config.yaml by adding `profile: true` under the `handler` section.
1. Create a new model archive using torch-model-archiver with the updated configuration.
1. Start TorchServe and run inference.
1. Analyze the TorchServe logs for metrics like `ts_handler_preprocess.Milliseconds`, `ts_handler_inference.Milliseconds`, and `ts_handler_postprocess.Milliseconds`. These metrics represent the time taken for pre-processing, inference, and post-processing steps, respectively, for each inference request.


#### 6.1. Measure inference time with `torch.compile` Inductor backend

Update the `model-config.yaml` file to specify the Inductor backend:

```yaml
pt2: {backend: inductor, mode: reduce-overhead}
```
Make sure that profiling is enabled

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` requires a warm-up phase to reach optimal performance. Ensure you run at least as many inferences as the `maxWorkers` specified before measuring performance.

After a few iterations of warmup, we see the following

```bash
2024-04-25T07:21:31,722 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:0.0054836273193359375|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029691,10ca6d02-5895-4af3-a052-6b5d409ca676, pattern=[METRICS]
2024-04-25T07:22:31,405 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:59682.70015716553|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029751,10ca6d02-5895-4af3-a052-6b5d409ca676, pattern=[METRICS]
2024-04-25T07:22:31,947 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:542.2341823577881|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029751,10ca6d02-5895-4af3-a052-6b5d409ca676, pattern=[METRICS]
```

#### 6.2. Measure inference time with `torch.compile` OpenVINO backend

Update the `model-config.yaml` file to specify the OpenVINO backend:

```yaml
pt2: {backend: openvino}
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` requires a warm-up phase to reach optimal performance. Ensure you run at least as many inferences as the `maxWorkers` specified before measuring performance.

After a few iterations of warmup, we see the following:

```bash
2024-04-25T07:12:36,276 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:0.0045299530029296875|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029156,2d8c54ac-1c6f-43d7-93b0-bb205a9a06ee, pattern=[METRICS]
2024-04-25T07:12:51,667 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:15391.06822013855|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029171,2d8c54ac-1c6f-43d7-93b0-bb205a9a06ee, pattern=[METRICS]
2024-04-25T07:12:51,955 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:287.31536865234375|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029171,2d8c54ac-1c6f-43d7-93b0-bb205a9a06ee, pattern=[METRICS]
```

#### 6.3. Measure inference time with `torch.compile` OpenVINO backend on Intel Discrete GPU

Update the `model-config.yaml` file to specify the OpenVINO backend and with Intel GPU device:

```yaml
pt2: {backend: 'openvino', options: {'model_caching' : True, 'device': 'GPU'}}
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` requires a warm-up phase to reach optimal performance. Ensure you run at least as many inferences as the `maxWorkers` specified before measuring performance.

After a few iterations of warmup, we see the following:

```bash
2024-04-25T07:28:32,662 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:0.0050067901611328125|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714030112,579edbf3-5d78-40aa-b49c-480796b4d3b1, pattern=[METRICS]
2024-04-25T07:28:39,887 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:7225.085020065308|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714030119,579edbf3-5d78-40aa-b49c-480796b4d3b1, pattern=[METRICS]
2024-04-25T07:28:40,174 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:286.96274757385254|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714030120,579edbf3-5d78-40aa-b49c-480796b4d3b1, pattern=[METRICS]
```

### Conclusion

Using `torch.compile` with the OpenVINO backend significantly enhances the performance of the StableDiffusionXL model. When comparing backends:

- Using the **Inductor backend**, the inference time on a CPU (Intel Xeon Platinum 8469) is around 59 seconds.
- Switching to the **OpenVINO backend** on the same CPU (Intel Xeon Platinum 8469) reduces the inference time to approximately 15 seconds.
- Furthermore, employing an Intel Discrete GPU (Intel GPU Flex 170) with the OpenVINO backend reduces the inference time even more dramatically, to about 7 seconds.

The actual performance gains may vary depending on your hardware, model complexity, and workload. Consider exploring more advanced `torch.compile` [configurations](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html) for further optimization based on your specific use case.
