# TorchServe inference with torch tensorrt (using TorchScript) model

This example shows how to run TorchServe inference with [Torch-TensorRT](https://github.com/pytorch/TensorRT) model using TorchScript. This is the legacy way of using TensorRT with PyTorch. We recommend using torch.compile for new deployments (see [../README.md](../README.md)). TorchScript is in maintenance mode.

### Pre-requisites

- Install CUDA and cuDNN. Verified with CUDA 11.7 and cuDNN 8.9.3.28
- Verified to be working with `tensorrt==8.5.3.1` and `torch-tensorrt==1.4.0`

Change directory to the root of `serve`
Ex: if `serve` is under `/home/ubuntu`, change directory to `/home/ubuntu/serve`


### Create a Torch Tensor RT model

We use `float16` precision
TorchServe's base handler supports loading Torch TensorRT model with `.pt` extension. Hence, the model is saved with `.pt` extension.

```
python examples/torch_tensorrt/resnet_tensorrt.py
```

### Create model archive

```
torch-model-archiver --model-name res50-trt-fp16 --handler image_classifier --version 1.0 --serialized-file res50_trt_fp16.pt --extra-files ./examples/image_classifier/index_to_name.json
mkdir model_store
mv res50-trt-fp16.mar model_store/.
```

#### Start TorchServe
```
torchserve --start --model-store model_store --models res50-trt-fp16=res50-trt-fp16.mar --ncs --disable-token-auth  --enable-model-api
```

#### Run Inference

```
curl http://127.0.0.1:8080/predictions/res50-trt-fp16 -T ./examples/image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.2723647356033325,
  "tiger_cat": 0.13748960196971893,
  "Egyptian_cat": 0.04659610986709595,
  "lynx": 0.00318642589263618,
  "lens_cap": 0.00224193069152534
}
```
