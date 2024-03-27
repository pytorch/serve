# TorchServe Quick Start Examples

## Pre-requisites

1) Docker for CPU runs. To make use of Nvidia GPU, please make sure you have nvidia-docker installed.

## Quick Start Example
To quickly get started with TorchServe, you can execute the following commands where `serve` is cloned.

```
./examples/getting_started/build_image.sh vit

docker run --rm -it --env TORCH_COMPILE=false --env MODEL_NAME=vit --platform linux/amd64 -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -v /home/ubuntu/serve/model_store_1:/home/model-server/model-store pytorch/torchserve:demo
```

You can point `/home/ubuntu/serve/model_store_1` to a volume where you want the model archives to be stored

In another terminal, run the following command for inference
```
curl http://127.0.0.1:8080/predictions/vit -T ./examples/image_classifier/kitten.jpg
```

### Supported models

The following models are supported in this example
```
resnet, densenet, vit, fasterrcnn, bertsc, berttc, bertqa, berttg
```

We use HuggingFace BERT models. So you need to set `HUGGINGFACE_TOKEN`

```
export HUGGINGFACE_TOKEN=< Your token>
```

### `torch.compile`

To enable `torch.compile` with these models, pass this optional argument `--torch.compile`

```
./examples/getting_started/build_image.sh resnet --torch.compile
```

## Register multiple models

TorchServe supports multi-model endpoints out of the box. Once, you have loaded a model, you can register it along with any other model using TorchServe's management API.
Depending on the amount of memory (or GPU memory) you have on your machine, you can load as many models.

```
curl -X POST "127.0.0.1:8081/models?model_name=resnet&url=/home/ubuntu/serve/model_store_1/resnet"
```
You can check all the loaded models using
```
curl -X GET "127.0.0.1:8081/models"
```

For other management APIs, please refer to the [document](https://github.com/pytorch/serve/blob/master/docs/management_api.md)
