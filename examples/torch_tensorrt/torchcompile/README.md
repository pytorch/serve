# TorchServe inference with torch.compile with tensorrt backend

This example shows how to run TorchServe inference with [Torch-TensorRT](https://github.com/pytorch/TensorRT) model

### Pre-requisites

- Verified to be working with `torch-tensorrt==2.3.0`

Change directory to examples directory `cd examples/torch_tensorrt/torchcompile`

### torch.compile config

To use `tensorrt` backend with `torch.compile`, we specify the following config in `model-config.yaml`

```
pt2:
  compile:
    enable: True
    backend: tensorrt
```

### Download the weights

```
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
```
### Create model archive

```
mkdir model_store

torch-model-archiver --model-name res50-trt --handler image_classifier --version 1.0 --model-file model.py --serialized-file resnet50-11ad3fa6.pth --config-file model-config.yaml --extra-files ../../image_classifier/index_to_name.json --export-path model_store -f

```

#### Start TorchServe
```
torchserve --start --model-store model_store --models res50-trt=res50-trt.mar --disable-token --ncs
```

#### Run Inference

```
curl http://127.0.0.1:8080/predictions/res50-trt -T ../../image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.27221813797950745,
  "tiger_cat": 0.13754481077194214,
  "Egyptian_cat": 0.04620043560862541,
  "lynx": 0.003195191267877817,
  "lens_cap": 0.00225762533955276
}
```

## Measuring speedup

```
2024-06-22T18:40:52,651 [INFO ] W-9000-res50-trt_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:6.462495803833008|#ModelName:res50-trt,Level:Model|#type:GAUGE|#hostname:ip-172-31-4-205,1719081652,edac5623-7904-47a9-b6f6-bdcc5f8590ed, pattern=[METRICS]
2024-06-22T18:40:52,653 [INFO ] W-9000-res50-trt_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:1.600767970085144|#ModelName:res50-trt,Level:Model|#type:GAUGE|#hostname:ip-172-31-4-205,1719081652,edac5623-7904-47a9-b6f6-bdcc5f8590ed, pattern=[METRICS]
2024-06-22T18:40:52,653 [INFO ] W-9000-res50-trt_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.21452799439430237|#ModelName:res50-trt,Level:Model|#type:GAUGE|#hostname:ip-172-31-4-205,1719081652,edac5623-7904-47a9-b6f6-bdcc5f8590ed, pattern=[METRICS]
```

To switch to PyTorch eager, we remove the `pt2` config in `model-config.yaml` or use the following

```
pt2:
  compile:
    enable: false
```

If we disable `torch.compile` and use PyTorch eager, we see the following

```
2024-06-22T18:42:32,540 [INFO ] W-9000-res50-trt_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:6.869855880737305|#ModelName:res50-trt,Level:Model|#type:GAUGE|#hostname:ip-172-31-4-205,1719081752,1eb885cf-c857-4d9e-b2f8-27ec70311e32, pattern=[METRICS]
2024-06-22T18:42:32,545 [INFO ] W-9000-res50-trt_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:5.565248012542725|#ModelName:res50-trt,Level:Model|#type:GAUGE|#hostname:ip-172-31-4-205,1719081752,1eb885cf-c857-4d9e-b2f8-27ec70311e32, pattern=[METRICS]
2024-06-22T18:42:32,546 [INFO ] W-9000-res50-trt_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.16128000617027283|#ModelName:res50-trt,Level:Model|#type:GAUGE|#hostname:ip-172-31-4-205,1719081752,1eb885cf-c857-4d9e-b2f8-27ec70311e32, pattern=[METRICS]
```

We see that `torch.compile` with `tensorrt` backend reduces model inference from `5.56 ms` to `1.6 ms`.
Please note that `torch.compile` is a JIT compiler and it takes a few iterations (1-3) to warmup before you see the speedup

