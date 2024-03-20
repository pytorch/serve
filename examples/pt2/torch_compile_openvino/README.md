
# TorchServe inference with torch.compile with OpenVINO backend of densenet161 model

This example shows how to take eager model of `densenet161`, configure TorchServe to use `torch.compile` and run inference using `torch.compile` with [OpenVINO backend](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html) 

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
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
mkdir model_store
torch-model-archiver --model-name densenet161 --version 1.0 --model-file model.py --serialized-file densenet161-8d451a50.pth --export-path model_store --extra-files ../../image_classifier/index_to_name.json --handler image_classifier --config-file model-config.yaml -f
```

#### Start TorchServe
```
torchserve --start --ncs --model-store model_store --models densenet161.mar
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
echo "handler:" > model-config.yaml && \
echo "  profile: true" >> model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.



#### Measure inference time with `torch.compile` with OpenVINO backend

```
echo "minWorkers: 1
maxWorkers: 2
pt2: {backend: openvino}" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` needs a few inferences to warmup. Once warmed up, we see the following


### Conclusion

`torch.compile` with openvino backend reduces the inference time.
