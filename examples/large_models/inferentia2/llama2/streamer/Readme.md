# Demo1: Llama-2 Using TorchServe micro-batching and Streamer on inf2

This document briefs on serving the [Llama 2](https://huggingface.co/meta-llama) model on [AWS Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/) for text completion with TorchServe [micro batching](https://github.com/pytorch/serve/tree/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/examples/micro_batching) and [streaming response](https://github.com/pytorch/serve/blob/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/docs/inference_api.md#curl-example-1) support.

**Note**: To run the model on an Inf2 instance, the model gets compiled as a preprocessing step. As part of the compilation process, to generate the model graph, a specific batch size is used. Following this, when running inference, we need to pass input which matches the batch size that was used during compilation. Model compilation and input padding to match compiled model batch size is taken care of by the [custom handler](inf2_handler.py) in this example.

The batch size and micro batch size configurations are present in [model-config.yaml](model-config.yaml). The batch size indicates the maximum number of requests torchserve will aggregate and send to the custom handler within the batch delay.
The batch size is chosen to be a relatively large value, say 16 since micro batching enables running the preprocess(tokenization) and inference steps in parallel on the micro batches. The micro batch size is the batch size used for the Inf2 model compilation.
Since compilation batch size can influence compile time and also constrained by the Inf2 instance type, this is chosen to be a relatively smaller value, say 4.
