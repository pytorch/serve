# Demo2: Llama-2 Using TorchServe continuous batching on inf2

This document briefs on serving the [Llama 2](https://huggingface.co/meta-llama) model on [AWS transformers-neuronx continuous batching](https://aws.amazon.com/ec2/instance-types/inf2/).

This example can also be extended to support the following models.


| Model       | Model Class                        |
| :---        | :----:                             |
| opt         | opt.model.OPTForSampling           |
| gpt2        | gpt2.model.GPT2ForSampling         |
| gptj        | gptj.model.GPTJForSampling         |
| gpt_neox    | gptneox.model.GPTNeoXForSampling   |
| llama       | lama.model.LlamaForSampling        |
| mistral     | mistral.model.MistralForSampling   |
| bloom       | bloom.model.BloomForSampling       |

The batch size [model-config.yaml](model-config.yaml). The batch size indicates the maximum number of requests torchserve will aggregate and send to the custom handler within the batch delay. It is the batch size used for the Inf2 model compilation.
Since compilation batch size can influence compile time and also constrained by the Inf2 instance type, this is chosen to be a relatively smaller value, say 4.

`inf2-llama-2-continuous-batching.ipynb` is the notebook example.
