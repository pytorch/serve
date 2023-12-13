# TorchServe inference with torch._export aot_compile

This example shows how to run TorchServe with Torch exported model with AOTInductor

Using `torch.compile` to wrap your existing eager PyTorch model can result in out of the box speedups. However, `torch.compile` is a JIT compiler. TorchServe has been supporting `torch.compile` since PyTorch 2.0 release. In a production setting, when you have multiple instances of TorchServe, each of of your instances would `torch.compile` the model on inference. TorchServe's model archiever is not able to truly guarantee reproducibility because its a JIT compiler.

In addition, the first inference request with `torch.compile` will be slow as the model needs to compile.

To solve this problem, `torch.export` has an experimental API `torch._export.aot_compile` which is able to `torch.export` a torch compilable model with no graphbreaks along with AOTInductor.

You can find more details [here](https://pytorch.org/docs/main/torch.compiler_aot_inductor.html)



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

### Create a Torch exported model with AOTInductor

The model is saved with `.so` extension
Here we are torch exporting with AOT Inductor with `max_auotune` mode.
This is also making use of `dynamic_shapes` to support batch size from 1 to 32.
In the code, the min batch_size is mentioned as 2 instead of 1. You can find an explanation for this [here](https://pytorch.org/docs/main/export.html#expressing-dynamism)

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
