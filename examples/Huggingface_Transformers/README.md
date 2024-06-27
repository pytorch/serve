## Serving Huggingface Transformers using TorchServe

In this example, we show how to serve a fine tuned or off the shelf Transformer model from [huggingface](https://huggingface.co/docs/transformers/index) using TorchServe.

We use a custom handler, [Transformer_handler.py](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py).

This handler enables us to use pre-trained transformer models from Huggingface, such as BERT, RoBERTA, XLM for token classification, sequence classification and question answering.

We borrowed ideas to write a custom handler for transformers from tutorial presented in [mnist from image classifiers examples](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist) and the post by [MFreidank](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18).

To get started [install Torchserve](https://github.com/pytorch/serve) and then

 `pip install -r requirements.txt`

### Objectives
1. How to package a transformer into a torch model archive (.mar) file (eager mode or Torchscript) with `torch-model-archiver`
2. How to load mar file in torch serve to run inferences and explanations using `torchserve`

### **Getting Started with the Demo**

If you're finetuning an existing model then you need to save your model and tokenizer with `save_pretrained()` which will create a `model.safetensors`, `vocab.txt` and `config.json` file. Make sure to create them then run

```
mkdir Transformer_model
mv model.safetensors vocab.txt config.json Transformer_model/
```

If you'd like to download a pretrained model without fine tuning we've provided a simple helper script which will do the above for you. All you need to do is change [model-config.yaml](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/model-config.yaml) to your liking and run

`python Download_Transformer_models.py`

In this example, we are using `torch.compile` by default

This is enabled by the following config in `model-config.yaml` file

```
pt2:
  compile:
    enable: True
    backend: inductor
    mode: reduce-overhead
```
When batch_size is 1, for BERT models, the operations are memory bound. Hence, we make use of `reduce-overhead` mode to make use of CUDAGraph and get better performance.

To use PyTorch Eager or TorchScript, you can remove the above config.

For Torchscript support, check out [torchscript.md](torchscript.md)

#### Setting the handler config in model-config.yaml

In `model-config.yaml`  :

*model_name* : bert-base-uncased , roberta-base or other available pre-trained models.

*mode:* `sequence_classification`, `token_classification`, `question_answering` or `text_generation`

*do_lower_case* : `true` or `false` which configures the tokenizer

*num_labels* : number of outputs for `sequence_classification`: 2, `token_classification`: 9, `question_answering`: 0 or `text_generation`: 0

*save_mode* : "torchscript" or "pretrained", this setting will be used by `Download_transformer_models.py` script as well as the handler, to download/save and load the model in Torchscript or save_pretrained mode respectively.

*max_length* : maximum length for the  input sequences to the models, this will be used in preprocessing of the handler. Also, if you choose to use Torchscript as the serialized model  for packaging your model this length should be equal to the length that has been used during the tracing of the model using torch.jit.trace.

*captum_explanation* : `true` for eager mode models but should be set to `false` for torchscripted models or if you don't need explanations

*embedding_name* : The name of embedding layer in the chosen model, this could be `bert` for `bert-base-uncased`, `roberta` for `roberta-base` or `roberta` for `xlm-roberta-large`, or `gpt2` for `gpt2` model

*hardware* : The target platform to trace the model for. Specify as `neuron` for [Inferentia1](https://aws.amazon.com/ec2/instance-types/inf1/) and `neuronx` for [Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/).

*batch_size* : Input batch size when tracing the model for `neuron` or `neuronx` as target hardware.

Once, `model-config.yaml` has been set properly, the next step is to run

`python Download_Transformer_models.py`

This produces all the required files for packaging using a huggingface transformer model off-the-shelf without fine-tuning process. Using this option will create and saved the required files into Transformer_model directory.


#### Setting the extra_files

There are few files that are used for model packaging and at the inference time.
* `index_to_name.json`: maps predictions to labels
* `sample_text.txt`: input text for inference
* `vocab.txt`: by default will use the tokenizer from the pretrained model

For custom vocabs, it is required to pass all other tokenizer related files such `tokenizer_config.json`, `special_tokens_map.json`, `config.json` and if available `merges.txt`.

For examples of how to configure a model for a use case and what the input format should look like
* Model configuration: `Transformer_model` directory after running `python Download_Transformer_models.py`
* Examples: `QA_artifacts`, `Seq_classification_artifacts`, `Token_classification_artifacts` or `Text_gen_artifacts`


## Sequence Classification

### Create model archive for eager mode or torch.compile

```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/model.safetensors --handler ./Transformer_handler_generalized.py --config-file model-config.yaml --extra-files "Transformer_model/config.json,./Seq_classification_artifacts/index_to_name.json"

```

### Create model archive Torchscript mode

```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --config-file model-config.yaml --extra-files "./Seq_classification_artifacts/index_to_name.json"

```

### Register the model

To register the model on TorchServe using the above model archive file, we run the following commands:

```
mkdir model_store
mv BERTSeqClassification.mar model_store/
torchserve --start --model-store model_store --models my_tc=BERTSeqClassification.mar --disable-token --ncs

```

### Run an inference

To run an inference: `curl -X POST http://127.0.0.1:8080/predictions/my_tc -T Seq_classification_artifacts/sample_text_captum_input.txt`

The response should be a "Not Accepted" classification.

To get an explanation: `curl -X POST http://127.0.0.1:8080/explanations/my_tc -T Seq_classification_artifacts/sample_text_captum_input.txt`

## Token Classification

Change the `handler` section in `model-config.yaml` to

```
handler:
  model_name: bert-base-uncased
  mode: token_classification
  do_lower_case: true
  num_labels: 9
  save_mode: pretrained
  max_length: 150
  captum_explanation: true
  embedding_name: bert
  BetterTransformer: false
  model_parallel: false
```

```
rm -r Transformer_model
python Download_Transformer_models.py
```

### Create model archive for eager mode or torch.compile
```
torch-model-archiver --model-name BERTTokenClassification --version 1.0 --serialized-file Transformer_model/model.safetensors --handler ./Transformer_handler_generalized.py --config-file model-config.yaml --extra-files "Transformer_model/config.json,./Token_classification_artifacts/index_to_name.json"
```

### Create model archive Torchscript mode
```
torch-model-archiver --model-name BERTTokenClassification --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --config-file model-config.yaml --extra-files "./Token_classification_artifacts/index_to_name.json"
```

### Register the model

```
mkdir model_store
mv BERTTokenClassification.mar model_store
torchserve --start --model-store model_store --models my_tc=BERTTokenClassification.mar --disable-token --ncs
```

### Run an inference
To run an inference: `curl -X POST http://127.0.0.1:8080/predictions/my_tc -T Token_classification_artifacts/sample_text_captum_input.txt`
To get an explanation: `curl -X POST http://127.0.0.1:8080/explanations/my_tc -T Token_classification_artifacts/sample_text_captum_input.txt`

## Question Answering

Change the `handler` section in `model-config.yaml` to
```
handler:
  model_name: distilbert-base-cased-distilled-squad
  mode: question_answering
  do_lower_case: true
  num_labels: 0
  save_mode: pretrained
  max_length: 150
  captum_explanation: true
  embedding_name: distilbert
  BetterTransformer: false
  model_parallel: false
```

```
rm -r Transformer_model
python Download_Transformer_models.py
```

### Create model archive for eager mode or torch.compile
```
torch-model-archiver --model-name BERTQA --version 1.0 --serialized-file Transformer_model/model.safetensors --handler ./Transformer_handler_generalized.py --config-file model-config.yaml --extra-files "Transformer_model/config.json"
```

### Create model archive Torchscript mode
```
torch-model-archiver --model-name BERTQA --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --config-file model-config.yaml
```

### Register the model

```
mkdir model_store
mv BERTQA.mar model_store
torchserve --start --model-store model_store --models my_tc=BERTQA.mar --disable-token --ncs
```
### Run an inference
To run an inference: `curl -X POST http://127.0.0.1:8080/predictions/my_tc -T QA_artifacts/sample_text_captum_input.txt`
To get an explanation: `curl -X POST http://127.0.0.1:8080/explanations/my_tc -T QA_artifacts/sample_text_captum_input.txt`

## Text Generation

Change the `handler` section in `model-config.yaml` to
```
handler:
  model_name: gpt2
  mode: text_generation
  do_lower_case: true
  num_labels: 0
  save_mode: pretrained
  max_length: 150
  captum_explanation: true
  embedding_name: gpt2
  BetterTransformer: false
  model_parallel: false
```

```
rm -r Transformer_model
python Download_Transformer_models.py
```

### Create model archive eager mode

```
torch-model-archiver --model-name Textgeneration --version 1.0 --serialized-file Transformer_model/model.safetensors --handler ./Transformer_handler_generalized.py --config-file model-config.yaml --extra-files "Transformer_model/config.json"
```

### Create model archive Torchscript mode

```
torch-model-archiver --model-name Textgeneration --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --config-file model-config.yaml
```

### Register the model

To register the model on TorchServe using the above model archive file, we run the following commands:

```
mkdir model_store
mv Textgeneration.mar model_store/
torchserve --start --model-store model_store --models my_tc=Textgeneration.mar --disable-token --ncs
```

### Run an inference

To run an inference: `curl -X POST http://127.0.0.1:8080/predictions/my_tc -T Text_gen_artifacts/sample_text.txt`
To get an explanation: `curl -X POST http://127.0.0.1:8080/explanations/my_tc -T Text_gen_artifacts/sample_text.tx`


## Batch Inference

For batch inference the main difference is that you need set the batch size while registering the model. This can be done either through the management API or if using Torchserve 0.4.1 and above, it can be set through config.properties as well.  Here is an example of setting batch size for sequence classification with management API and through config.properties. You can read more on batch inference in Torchserve [here](https://github.com/pytorch/serve/tree/master/docs/batch_inference_with_ts.md).

* Management API
    ```
    mkdir model_store
    mv BERTSeqClassification.mar model_store/
    torchserve --start --model-store model_store --disable-token --ncs

    curl -X POST "localhost:8081/models?model_name=BERTSeqClassification&url=BERTSeqClassification.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
    ```

* Config.properties
    ```text

    models={\
      "BERTSeqClassification": {\
        "2.0": {\
            "defaultVersion": true,\
            "marName": "BERTSeqClassification.mar",\
            "minWorkers": 1,\
            "maxWorkers": 1,\
            "batchSize": 4,\
            "maxBatchDelay": 5000,\
            "responseTimeout": 120\
        }\
      }\
    }
    ```
     ```
    mkdir model_store
    mv BERTSeqClassification.mar model_store/
    torchserve --start --model-store model_store --ts-config config.properties --models BERTSeqClassification= BERTSeqClassification.mar

    ```
Now to run the batch inference following command can be used:

```
curl -X POST http://127.0.0.1:8080/predictions/BERTSeqClassification  -T ./Seq_classification_artifacts/sample_text1.txt
& curl -X POST http://127.0.0.1:8080/predictions/BERTSeqClassification  -T ./Seq_classification_artifacts/sample_text2.txt
& curl -X POST http://127.0.0.1:8080/predictions/BERTSeqClassification -T ./Seq_classification_artifacts/sample_text3.txt &
```

## More information

### Captum Explanations for Visual Insights

The [Captum Explanations for Visual Insights Notebook](https://github.com/pytorch/serve/tree/master/examples/captum/Captum_visualization_for_bert.ipynb) provides a visual example for how model interpretations can help

Known issues:
* Captum does't work well for batched inputs and may result in timeouts
* No support for torchscripted models

### Captum JSON support
In order to run Captum Explanations with the request input in a json file, follow the below steps:

In the config.properties, specify `service_envelope=body` and make the curl request as below:
```bash
curl -H "Content-Type: application/json" --data @examples/Huggingface_Transformers/bert_ts.json http://127.0.0.1:8080/explanations/bert_explain
```

When a json file is passed as a request format to the curl, Torchserve unwraps the json file from the request body. This is the reason for specifying service_envelope=body in the config.properties file

## Speed up inference with Better Transformer (Flash Attentions/ Xformer Memory Efficient kernels)

In the `model-config.yaml`, specify `"BetterTransformer":true,`.


[Better Transformer(Accelerated Transformer)](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/) from PyTorch is integrated into [Huggingface Optimum](https://huggingface.co/docs/optimum/bettertransformer/overview) that bring major speedups for many of encoder models on different modalities (text, image, audio). It is a one liner API that we have also added in the `Transformer_handler_generalized.py` in this example as well. That as shown above you just need to set `"BetterTransformer":true,` in the `model-config.yaml`.

Main speed ups in the Better Transformer comes from kernel fusion in the [TransformerEncoder] (https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html) and making use of sparsity with [nested tensors](https://pytorch.org/tutorials/prototype/nestedtensor.html) when input sequences are padded to avoid unnecessary computation on padded tensors. We have seen up to 4.5x speed up with distill_bert when used higher batch sizes with padding. Please read more about it in this [blog post](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2). You get some speedups even with Batch size = 1 and no padding however, major speed ups will show up when running inference with higher batch sizes (8.16,32) with padding.

The Accelerated Transformer integration with HuggingFace also added the support for decoder models, please read more about it [here](https://pytorch.org/blog/out-of-the-box-acceleration/). This adds the native support for Flash Attentions and Xformer Memory Efficient kernels in PyTorch and make it available on HuggingFace decoder models. This will brings significant speed up and memory savings with just one line of the code as before.


## Model Parallelism

[Parallelize] (https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Model.parallelize) is a an experimental feature that HuggingFace recently added to support large model inference for some very large models, GPT2 and T5. GPT2 model choices based on their size are gpt2-medium, gpt2-large, gpt2-xl. This feature only supports LMHeadModel that could be used for text generation, other application such as sequence, token classification and question answering are not supported. We have added parallelize support for GPT2 model in the custom handler in this example that will enable you to perform model parallel inference for GPT2 models used for text generation. The same logic in the handler can be extended to T5 and the applications it supports. Make sure that you register your model with one worker using this feature. To run this example, a machine with #gpus > 1 is required. The number of required gpus depends on the size of the model. This feature only supports single node, one machine with multi-gpus.

Change the `handler` section in `model-config.yaml` to
```
handler:
  model_name: gpt2
  mode: text_generation
  do_lower_case: true
  num_labels: 0
  save_mode: pretrained
  max_length: 150
  captum_explanation: true
  embedding_name: gpt2
  BetterTransformer: false
  model_parallel: true
```

```
rm -r Transformer_model
python Download_Transformer_models.py
```

### Create model archive eager mode

```
torch-model-archiver --model-name Textgeneration --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json"
```

### Register the model

To register the model on TorchServe using the above model archive file, we run the following commands:

```
mkdir model_store
mv Textgeneration.mar model_store/
torchserve --start --model-store model_store --disable-token
curl -X POST "localhost:8081/models?model_name=Textgeneration&url=Textgeneration.mar&batch_size=1&max_batch_delay=5000&initial_workers=1&synchronous=true"
```

### Run an inference

To run an inference: `curl -X POST http://127.0.0.1:8080/predictions/Textgeneration -T Text_gen_artifacts/sample_text.txt`
To get an explanation: `curl -X POST http://127.0.0.1:8080/explanations/Textgeneration -T Text_gen_artifacts/sample_text.tx`

### Running KServe

[BERT Readme for KServe](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/custom/torchserve/bert-sample/hugging-face-bert-sample.md).
[End to End KServe document](https://github.com/pytorch/serve/blob/master/kubernetes/kserve/README.md).
