
# TorchServe inference with torchao optimizations

This example shows how to take eager model of `ResNet50`, configure TorchServe to use `torch.compile`, using PyTorch native quantization using [`torchao`](https://github.com/pytorch/ao) and run inference


## Pre-requisites

- `PyTorch >= 2.4`
- `CUDA` enabled device
- `torchao` [installed](https://github.com/pytorch/ao?tab=readme-ov-file#installation)

Change directory to the examples directory
Ex:  `cd  examples/pt2/torchao/quantization`


## torchao autoquant API

`torchao.autoquant` first identifies the shapes of the activations that the different linear layers see, it then benchmarks these shapes across different types of quantized and non-quantized layers in order to pick the fastest one, attempting to take into account fusions where possible. Finally once the best class is found for each layer, it swaps the linear. You can find additional details [here](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#autoquantization)

In this example , we use the following config

```
minWorkers: 1
maxWorkers: 1
responseTimeout: 600
handler:
  profile: True
pt2:
  compile:
    enable: True
    backend: inductor
    mode: max-autotune
  ao:
    enable: True
    autoquant:
      enable: True > model-config.yaml
```

By default torch.autoquant uses `int8` quantization. You can specify `int4` quantization using the following config

```
pt2:
  ao:
    enable: True
    autoquant:
      enable: True
      qtensor_class_list: DEFAULT_INT4_AUTOQUANT_CLASS_LIST
```

### Create model archive

```
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
mkdir model_store
torch-model-archiver --model-name resnet-50 --version 1.0 --model-file model.py   --serialized-file resnet50-11ad3fa6.pth --export-path model_store   --extra-files ../../../image_classifier/index_to_name.json --handler image_classifier   --config-file model-config.yaml -f
```

#### Start TorchServe
```
torchserve --start --ncs --model-store model_store --models resnet-50.mar --disable-token-auth --enable-model-api
```

#### Run Inference

```
curl http://127.0.0.1:8080/predictions/resnet-50 -T ../../../image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.2691785991191864,
  "tiger_cat": 0.13622364401817322,
  "Egyptian_cat": 0.04588942974805832,
  "lynx": 0.0032150563783943653,
  "lens_cap": 0.0023105053696781397
}
```

## torchao Affine Quantization

Affine Quantization refers to the type of quantization that maps from high precision floating point numbers to quantized numbers (low precision integer or floating point dtype) with an Affine transformation. You can find additional details [here](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#affine-quantization) In the example below, we use `int8_weight_only` quantization

In this example , we use the following config

```
minWorkers: 1
maxWorkers: 1
responseTimeout: 600
handler:
  profile: True
pt2:
  compile:
    enable: True
    backend: inductor
    mode: max-autotune
  ao:
    enable: True
    quantize:
      enable: True
      quant_api: int8_weight_only > model-config.yaml
```

The rest of the steps to create a model archive and run inference are the same as shown above.
