# TorchServe inference with torch.compile with tensorrt backend

This example shows how to run TorchServe inference with T5 [Torch-TensorRT](https://github.com/pytorch/TensorRT) model



[T5](https://huggingface.co/docs/transformers/en/model_doc/t5#inference) is an encode-decoder model used for a variety of text tasks out of the box by prepending a different text corresponding to each task. In this example, we use T5 for translation from English to German.

### Pre-requisites

- Verified to be working with `torch-tensorrt==2.3.0`
Installation instructions can be found in [pytorch/TensorRT](https://github.com/pytorch/TensorRT)

Change directory to examples directory `cd examples/torch_tensorrt/T5/torchcompile`

### torch.compile config

To use `tensorrt` backend with `torch.compile`, we specify the following config in `model-config.yaml`

```
pt2:
  compile:
    enable: True
    backend: tensorrt
```

### Download the model

```
python ../../../large_models/Huggingface_accelerate/Download_model.py --model_name google-t5/t5-small
```

### Create the model archive
```
mkdir model_store
torch-model-archiver --model-name t5-translation --version 1.0 --handler T5_handler.py --config-file model-config.yaml -r requirements.txt --archive-format no-archive --export-path model_store -f
mv model model_store/t5-translation/.
```

### Start TorchServe

```
torchserve --start --ncs --ts-config config.properties --model-store model_store --models t5-translation --disable-token-auth
```

### Run Inference

```
curl -X POST http://127.0.0.1:8080/predictions/t5-translation -T sample_text.txt
```

results in

```
Das Haus ist wunderbar
```
