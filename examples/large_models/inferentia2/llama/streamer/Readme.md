# ⚠️ Notice: Limited Maintenance

This project is no longer actively maintained. While existing releases remain available, there are no planned updates, bug fixes, new features, or security patches. Users should be aware that vulnerabilities may not be addressed.

# Demo1: Llama-2 Using TorchServe micro-batching and Streamer on inf2

This document briefs on serving the [Llama 2](https://huggingface.co/meta-llama) model on [AWS Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/) for text completion with [micro batching](https://github.com/pytorch/serve/tree/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/examples/micro_batching) and [streaming response](https://github.com/pytorch/serve/blob/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/docs/inference_api.md#curl-example-1) support.

Inferentia2 uses [Neuron SDK](https://aws.amazon.com/machine-learning/neuron/) which is built on top of PyTorch XLA stack. For large model inference [`transformers-neuronx`](https://github.com/aws-neuron/transformers-neuronx) package is used that takes care of model partitioning and running inference.

This example can also be extended to support Mistral without code changes. Customers only set the following items in model-config.yaml. For example:
* model_path: "model/models--meta-llama--Llama-2-70b-hf/snapshots/90052941a64de02075ca800b09fcea1bdaacb939"
* model_checkpoint_dir: "llama-2-70b-split"
* model_module_prefix: "transformers_neuronx"
* model_class_name: "llama.model.LlamaForSampling"
* tokenizer_class_name: "transformers.LlamaTokenizer"

| Model       | Model Class                        |
| :---        | :----:                             |
| llama       | lama.model.LlamaForSampling        |
| mistral     | mistral.model.MistralForSampling   |


The `batchSize` in [model-config.yaml](model-config.yaml) indicates the maximum number of requests torchserve will aggregate and send to the custom handler within the batch delay. `micro_batch_size` is the batch size used for the Inf2 model compilation.
Since compilation batch size can influence compile time and also constrained by the Inf2 instance type, this is chosen to be a relatively smaller value, say 4.

`inf2-llama-2-micro-batching.ipynb` is the notebook example.
