# TorchServe inference with torch._export.aot_compile

This example shows how to run TorchServe with Torch exported model with AOTInductor

To understand when to use `torch._export.aot_compile`, please refer to this [section](https://github.com/pytorch/serve/tree/master/examples/pt2#torch_exportaot_compile)


### Pre-requisites

- `PyTorch >= 2.2.0`
- `CUDA 12.1`

Change directory to the examples directory
Ex:  `cd  examples/pt2/torch_export_aot_compile`

Install PyTorch 2.2 nightlies by running
```
chmod +x install_pytorch_nightlies.sh
source install_pytorch_nightlies.sh
```

You can also achieve this by installing TorchServe dependencies with the `nightly_torch` flag
```
python ts_scripts/install_dependencies.py --cuda=cu121 --nightly_torch
```


### Create a Torch exported model with AOTInductor

The model is saved with `.so` extension
Here we are torch exporting with AOT Inductor with `max_autotune` mode.
This is also making use of `dynamic_shapes` to support batch size from 1 to 32.
In the code, the min batch_size is mentioned as 2 instead of 1. Its by design. The code works for batch size 1. You can find an explanation for this [here](https://pytorch.org/docs/main/export.html#expressing-dynamism)

```
python resnet18_torch_export.py
```

### Create model archive

```
torch-model-archiver --model-name res18-pt2 --handler image_classifier --version 1.0 --serialized-file resnet18_pt2.so --config-file model-config.yaml --extra-files ../../image_classifier/index_to_name.json
mkdir model_store
mv res18-pt2.mar model_store/.
```

#### Start TorchServe
```
torchserve --start --model-store model_store --models res18-pt2=res18-pt2.mar --ncs
```

#### Run Inference

```
curl http://127.0.0.1:8080/predictions/res18-pt2 -T ../../image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.4087875485420227,
  "tiger_cat": 0.34661102294921875,
  "Egyptian_cat": 0.13007202744483948,
  "lynx": 0.024034621194005013,
  "bucket": 0.011633828282356262
}
```
