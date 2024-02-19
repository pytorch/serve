# Demo2: Llama-2 Using TorchServe continuous batching on inf2

This document briefs on serving the [Llama 2](https://huggingface.co/meta-llama) model on [AWS transformers-neuronx continuous batching](https://aws.amazon.com/ec2/instance-types/inf2/).

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


The batch size in [model-config.yaml](model-config.yaml) indicates the maximum number of requests torchserve will aggregate and send to the custom handler within the batch delay. It is the batch size used for the Inf2 model compilation.
Since compilation batch size can influence compile time and also constrained by the Inf2 instance type, this is chosen to be a relatively smaller value, say 4.

`inf2-llama-2-continuous-batching.ipynb` is the notebook example.
