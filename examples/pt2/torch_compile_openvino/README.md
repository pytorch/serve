
# TorchServe inference with torch.compile with OpenVINO backend of vgg16 model

This example shows how to take eager model of `vgg16`, configure TorchServe to use `torch.compile` and run inference using `torch.compile` with [OpenVINO backend](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html) and compare the performance

This example has been tested on Intel Xeon Platinum 8469 CPU.

### Pre-requisites

- `PyTorch >= 2.0`

Change directory to the examples directory
Ex:  `cd  examples/pt2/torch_compile_openvino`


### torch.compile config

`torch.compile` supports a variety of config and the performance you get can vary based on the config. You can find the various options [here](https://pytorch.org/docs/stable/generated/torch.compile.html) and see [here](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html) for OpenVINO backend options.

In this example , we use the following config

```
echo "minWorkers: 1
maxWorkers: 2
pt2: {backend: openvino}" > model-config.yaml
```

### Create model archive

```
wget https://download.pytorch.org/models/vgg16-397923af.pth
mkdir model_store
torch-model-archiver --model-name vgg16 --version 1.0 --model-file model.py --serialized-file vgg16-397923af.pth --export-path model_store --extra-files ../../image_classifier/index_to_name.json --handler image_classifier --config-file model-config.yaml
```

#### Start TorchServe
```
torchserve --start --ncs --model-store model_store --models vgg16.mar
```

#### Run Inference

**NOTE**: `torch.compile` needs a few inferences to warmup. Once warmed up, you can observe the speedup.
(Number of warmup runs >= num workers)


```
# Open a new terminal
cd  examples/pt2/torch_compile_openvino
curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}
```

#### Stop the server

```
torchserve --stop

```


### Performance improvement from using `torch.compile` with OpenVINO backend

To measure the handler `preprocess`, `inference`, `postprocess` times, run the following

#### Measure inference time with PyTorch eager

```
echo "handler:" >> model-config.yaml && \
echo "  profile: true" >> model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.



#### Measure inference time with using Eager mode

```
echo "minWorkers: 1
maxWorkers: 2
handler:
  profile: true" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` needs a few inferences to warmup. Once warmed up, we can see the following:

```
2024-04-26T11:47:13,647 [INFO ] W-9000-vgg16_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:4.891157150268555|#ModelName:vgg16,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714132033,e74472af-f966-4416-9f8f-f971da717c20, pattern=[METRICS]
2024-04-26T11:47:13,665 [INFO ] W-9000-vgg16_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:17.756938934326172|#ModelName:vgg16,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714132033,e74472af-f966-4416-9f8f-f971da717c20, pattern=[METRICS]
2024-04-26T11:47:13,665 [INFO ] W-9000-vgg16_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.06771087646484375|#ModelName:vgg16,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714132033,e74472af-f966-4416-9f8f-f971da717c20, pattern=[METRICS]
```

#### Measure inference time with using `torch.compile` with backend Inductor

```
echo "minWorkers: 1
maxWorkers: 2
pt2: {backend: inductor, mode: reduce-overhead}
handler:
  profile: true" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` needs a few inferences to warmup. Once warmed up, we can see the following:

```
2024-04-26T11:45:01,170 [INFO ] W-9000-vgg16_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:5.16819953918457|#ModelName:vgg16,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714131901,28514996-1e1f-43c6-87a7-dec9ebf07151, pattern=[METRICS]
2024-04-26T11:45:01,182 [INFO ] W-9000-vgg16_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:12.207269668579102|#ModelName:vgg16,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714131901,28514996-1e1f-43c6-87a7-dec9ebf07151, pattern=[METRICS]
2024-04-26T11:45:01,182 [INFO ] W-9000-vgg16_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.06318092346191406|#ModelName:vgg16,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714131901,28514996-1e1f-43c6-87a7-dec9ebf07151, pattern=[METRICS]
```

#### Measure inference time with using `torch.compile` with backend OpenVINO

```
echo "minWorkers: 1
maxWorkers: 2
pt2: {backend: openvino}
handler:
  profile: true" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` needs a few inferences to warmup. Once warmed up, we can see the following:

```
2024-04-26T11:42:35,243 [INFO ] W-9000-vgg16_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:5.307197570800781|#ModelName:vgg16,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714131755,e278874e-2061-4cc0-bd01-a59d48eaaf66, pattern=[METRICS]
2024-04-26T11:42:35,250 [INFO ] W-9000-vgg16_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:6.224632263183594|#ModelName:vgg16,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714131755,e278874e-2061-4cc0-bd01-a59d48eaaf66, pattern=[METRICS]
2024-04-26T11:42:35,250 [INFO ] W-9000-vgg16_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.0782012939453125|#ModelName:vgg16,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714131755,e278874e-2061-4cc0-bd01-a59d48eaaf66, pattern=[METRICS]
```

### Conclusion

`torch.compile` with openvino backend reduces the inference time from 17ms and 12ms to about 6ms when compared to eager and inductor backend respectively.
