# Large model inference on Inferentia2

This folder briefs on serving the [Llama 2](https://huggingface.co/meta-llama) model on [AWS Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/) for text completion with TorchServe's features:

* demo1: [micro batching](https://github.com/pytorch/serve/tree/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/examples/micro_batching) and [streaming response](https://github.com/pytorch/serve/blob/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/docs/inference_api.md#curl-example-1) support in folder [streamer](streamer).
* demo2: continuous batching support in folder [continuous_batching](continuous_batching)
