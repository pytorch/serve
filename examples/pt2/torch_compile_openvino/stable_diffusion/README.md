
## Accelerating StableDiffusionXL model with torch.compile OpenVINO backend

[Stable Diffusion XL ](https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl) is a image generation model that is geared towards generating more photorealistic images compared to previous SD models.

This example has been tested on Intel Xeon Platinum 8469 CPU with Intel GPU Flex 170.


#### Pre-requisites
Install the latest OpenVINO package from Pypi
```
pip install openvino
```

`cd` to the example folder `examples/pt2/torch_compile_openvino/stable_diffusion`

### Step 1: Download the Stable diffusion model

```bash
python Download_model.py
```
This saves the model in `Base_Diffusion_model`

### Step 1: Generate model archive
At this stage we're creating the model archive which includes the configuration of our model in [model_config.yaml](./model_config.yaml).

### Enable OpenVINO backend
To use torch.compile with OpenVINO backend, add the following line to the model_config.yaml file

```
echo "pt2 : {backend: "openvino"} >> model_config.yaml
```

We can also add additional options like model_caching, device and many more OpenVINO specific configs as well in options. For more information, refer to https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html#options

#### For Example:
```
pt2: {backend: "openvino", options: {"model_caching" : True, "device": "GPU"}}
```

Create the model archive

```
torch-model-archiver --model-name diffusion_fast --version 1.0 --handler stable_diffusion_handler.py --config-file model_config.yaml --extra-files "./pipeline_utils.py" --archive-format no-archive
mv Base_Diffusion_model diffusion_fast/
```

### Step 2: Add the model archive to model store

```
mkdir model_store
mv diffusion_fast model_store
```

### Step 3: Start torchserve

```
torchserve --start --ts-config config.properties --model-store model_store --models diffusion_fast
```

### Step 4: Run inference

```
python query.py --url "http://localhost:8080/predictions/diffusion_fast" --prompt "a photo of an astronaut riding a horse on mars"
```
The image generated will be written to a file `output-<>.jpg`

### Performance improvement from using `torch.compile` with OpenVINO backend

To measure the handler `prepocess`, `inference`, `postprocess` times, make sure profiling is enabled by adding the following to the model_config.yaml file.

```
echo "  profile: true" >> model-config.yaml
```

#### Measure inference time with `torch.compile` Inductor backend


Remove the following line from the model_config.yaml to run inference with torch.compile Inductor backend

```
pt2 : {backend: "openvino"}
```
Make sure that profiling is enabled

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.

`torch.compile` requires a few iterations of warmup, after which we see the following


```
2024-04-25T07:21:31,722 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:0.0054836273193359375|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029691,10ca6d02-5895-4af3-a052-6b5d409ca676, pattern=[METRICS]
2024-04-25T07:22:31,405 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:59682.70015716553|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029751,10ca6d02-5895-4af3-a052-6b5d409ca676, pattern=[METRICS]
2024-04-25T07:22:31,947 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:542.2341823577881|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029751,10ca6d02-5895-4af3-a052-6b5d409ca676, pattern=[METRICS]
```

#### Measure inference time with `torch.compile` OpenVINO backend

```
echo "pt2 : {backend: "openvino"} >> model_config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.

After a few iterations of warmup, we see the following

```
2024-04-25T07:12:36,276 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:0.0045299530029296875|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029156,2d8c54ac-1c6f-43d7-93b0-bb205a9a06ee, pattern=[METRICS]
2024-04-25T07:12:51,667 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:15391.06822013855|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029171,2d8c54ac-1c6f-43d7-93b0-bb205a9a06ee, pattern=[METRICS]
2024-04-25T07:12:51,955 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:287.31536865234375|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714029171,2d8c54ac-1c6f-43d7-93b0-bb205a9a06ee, pattern=[METRICS]
```

#### Measure inference time with `torch.compile` OpenVINO backend on dGPU

To run inference with `torch.compile` OpenVINO backend on a machine enabled with Intel Discrete GPUs, run the following

```
echo "pt2 : {backend: "openvino", options : {"device": "GPU"}} >> model_config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.

After a few iterations of warmup, we see the following

```
2024-04-25T07:28:32,662 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:0.0050067901611328125|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714030112,579edbf3-5d78-40aa-b49c-480796b4d3b1, pattern=[METRICS]
2024-04-25T07:28:39,887 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:7225.085020065308|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714030119,579edbf3-5d78-40aa-b49c-480796b4d3b1, pattern=[METRICS]
2024-04-25T07:28:40,174 [INFO ] W-9000-diffusion_fast_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:286.96274757385254|#ModelName:diffusion_fast,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714030120,579edbf3-5d78-40aa-b49c-480796b4d3b1, pattern=[METRICS]
```

### Conclusion

`torch.compile` OpenVINO backend reduces the infer time from 59s to 15s when compared to Inductor backend and further reduces the infer time to 7s if a Intel Discrete GPU is used.