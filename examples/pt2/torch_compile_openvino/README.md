
# TorchServe inference with torch.compile with OpenVINO backend of Resnet50 model

This example shows how to take eager model of `Resnet50`, configure TorchServe to use `torch.compile` and run inference using `torch.compile` with [OpenVINO backend](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html) and compare the performance

This example has been tested on Intel Xeon Platinum 8469 CPU.

### Pre-requisites
- `PyTorch >= 2.1.0`

Install the latest OpenVINO package from Pypi
```
pip install openvino
```

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
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
mkdir model_store
torch-model-archiver --model-name resnet-50 --version 1.0 --model-file model.py --serialized-file resnet50-11ad3fa6.pth --export-path model_store --extra-files ../../image_classifier/index_to_name.json --handler image_classifier --config-file model-config.yaml
```

#### Start TorchServe
```
torchserve --start --ncs --model-store model_store --models resnet-50.mar
```

#### Run Inference

**NOTE**: `torch.compile` needs a few inferences to warmup. Once warmed up, you can observe the speedup.
(Number of warmup runs >= num workers)


```
# Open a new terminal
cd  examples/pt2/torch_compile_openvino
curl http://127.0.0.1:8080/predictions/resnet-50 -T ../../image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.27249985933303833,
  "tiger_cat": 0.13740447163581848,
  "Egyptian_cat": 0.04627467691898346,
  "lynx": 0.0032067003194242716,
  "lens_cap": 0.002257897751405835
}
```

#### Stop the server

```
torchserve --stop

```


### Performance improvement from using `torch.compile` with OpenVINO backend

To measure the handler `preprocess`, `inference`, `postprocess` times, run the following

```
echo "handler:" >> model-config.yaml && \
echo "  profile: true" >> model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.



#### Measure inference time with Pytorch Eager mode

```
echo "minWorkers: 1
maxWorkers: 2
handler:
  profile: true" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` needs a few inferences to warmup. Once warmed up, we can see the following:

```
2024-05-01T10:29:29,586 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:5.254030227661133|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559369,fd3743e0-9c89-41b2-9972-c1f403872113, pattern=[METRICS]
2024-05-01T10:29:29,609 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:22.122859954833984|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559369,fd3743e0-9c89-41b2-9972-c1f403872113, pattern=[METRICS]
2024-05-01T10:29:29,609 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.057220458984375|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559369,fd3743e0-9c89-41b2-9972-c1f403872113, pattern=[METRICS]
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
2024-05-01T10:32:05,808 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:5.209445953369141|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559525,9f84ea11-7b77-40e3-bf2c-926746db9c6f, pattern=[METRICS]
2024-05-01T10:32:05,821 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:12.910842895507812|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559525,9f84ea11-7b77-40e3-bf2c-926746db9c6f, pattern=[METRICS]
2024-05-01T10:32:05,822 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.06079673767089844|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559525,9f84ea11-7b77-40e3-bf2c-926746db9c6f, pattern=[METRICS]
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
2024-05-01T10:40:45,031 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:5.637407302856445|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714560045,7fffdb96-7022-495d-95bb-8dd0b17bf30a, pattern=[METRICS]
2024-05-01T10:40:45,036 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:5.518198013305664|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714560045,7fffdb96-7022-495d-95bb-8dd0b17bf30a, pattern=[METRICS]
2024-05-01T10:40:45,037 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.06508827209472656|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714560045,7fffdb96-7022-495d-95bb-8dd0b17bf30a, pattern=[METRICS]

```

### Conclusion

`torch.compile` with openvino backend reduces the inference time from 22and 12ms to about 5ms when compared to eager and inductor backend respectively.
