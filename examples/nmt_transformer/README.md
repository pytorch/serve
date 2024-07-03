# Transformer (NMT) models for English-French and English-German translation.

The Transformer, introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), is a powerful sequence-to-sequence modeling architecture capable of producing state-of-the-art neural machine translation (NMT) systems.

Recently, the [fairseq](https://github.com/pytorch/fairseq#join-the-fairseq-community) team has explored large-scale semi-supervised training of Transformers using back-translated data, further improving translation quality over the original model. More details can be found in [this blog post](https://engineering.fb.com/ai-research/scaling-neural-machine-translation-to-bigger-data-sets-with-faster-training-and-inference/).

In this example, we have shown how to serve a [English-to-French/English-German Translation](https://pytorch.org/hub/pytorch_fairseq_translation/#english-to-french-translation) model using TorchServe. We have used a generalized [custom handler](model_handler_generalized.py) which enables us to translate English-to-French and English-to-German simultaneously. The generalized custom handler uses pre-trained [Transformer_WMT14_En-Fr / Transformer_WMT19_En-De](https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md ) models from [fairseq](https://github.com/pytorch/fairseq).

_NOTE: This example currently works with Py36 only due to fairseq dependency on dataclasses [issue](https://github.com/huggingface/transformers/issues/8638#issuecomment-790772391). This example currently doesn't work on Windows_

## Objectives
1. Demonstrate how to package a pre-trained Transformer (NMT) models for English-French and English-German translation with generalized custom handler into torch model archive (.mar) file
2. Demonstrate how to load model archive (.mar) file into TorchServe and run inference.

## Serve the Transformer (NMT) models for English-French/English-German on TorchServe

* To generate the model archive (.mar) file for English-to-French translation model using following command

    ```bash
    ./create_mar.sh en2fr_model
    ```
    The above command will create a "model_store" directory in the current working directory and generate TransformerEn2Fr.mar file.

* To generate the model archive (.mar) file for English-to-German translation model using following command

    ```bash
    ./create_mar.sh en2de_model
    ```
    The above command will create a "model_store" directory in the current working directory and generate TransformerEn2De.mar file.


* Start the TorchServe using the model archive (.mar) file created in above step

    ```bash
    torchserve --start --model-store model_store --ts-config config.properties --disable-token-auth  --enable-model-api
    ```

* Use [Management API](https://github.com/pytorch/serve/blob/master/docs/management_api.md#management-api) to register the model with one initial worker
	For English-to-French model
    ```bash
    curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=TransformerEn2Fr.mar"
    {
        "status": "Model \"TransformerEn2Fr\" Version: 1.0 registered with 1 initial workers"
    }
    ```
	For English-to-German model
	```bash
	curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=TransformerEn2De.mar"
    {
        "status": "Model \"TransformerEn2De\" Version: 1.0 registered with 1 initial workers"
    }
	```

* To get the inference use the following curl command
	For English-to-French model
    ```bash
    curl http://127.0.0.1:8080/predictions/TransformerEn2Fr -T model_input/sample.txt | json_pp
    {
        "input" : "Hi James, when are you coming back home? I am waiting for you.\nPlease come as soon as possible.",
        "french_output" : "Bonjour James, quand rentrerez-vous chez vous, je vous attends et je vous prie de venir le plus tôt possible."
    }
    ```
	For English-to-German model
	```bash
	curl http://127.0.0.1:8080/predictions/TransformerEn2De -T model_input/sample.txt | json_pp
    {
        "input" : "Hi James, when are you coming back home? I am waiting for you.\nPlease come as soon as possible.",
        "german_output" : "Hallo James, wann kommst du nach Hause? Ich warte auf dich. Bitte komm so bald wie möglich."
    }
	```
    Here sample.txt contains simple english sentences which are given as input to [Inference API](https://github.com/pytorch/serve/blob/master/docs/inference_api.md#predictions-api). The output of above curl command will be the french translation of sentences present in the sample.txt file.

## Batch Inference with TorchServe using Translation (NMT) model

### TorchServe Model Configuration

To configure TorchServe to use the batching feature, provide the batch configuration information through ["POST /models" API](https://github.com/pytorch/serve/blob/master/docs/batch_inference_with_ts.md#batch-inference-with-torchserve).

The configuration that we are interested in is the following:

1. `batch_size`: This is the maximum batch size that a model is expected to handle.
2. `max_batch_delay`: This is the maximum batch delay time TorchServe waits to receive `batch_size` number of requests. If TorchServe doesn't receive `batch_size` number of
requests before this timer time's out, it sends what ever requests that were received to the model `handler`.

### Steps to configure English-to-French translation model with batch-support

* Start the model server. In this example, we are starting the model server with config.properties file

    ```bash
    torchserve --start --model-store model_store --ts-config config.properties --disable-token-auth  --enable-model-api
    ```

* Now let's launch English_to_French translation model, which we have built to handle batch inference.
In this example, we are going to launch 1 worker which handles a `batch size` of 4 with a `max_batch_delay` of 10s.

    ```bash
    curl -X POST "http://localhost:8081/models?url=TransformerEn2Fr.mar&initial_workers=1&synchronous=true&batch_size=4&max_batch_delay=10000"
    ```

* Run batch inference command to test the model.

    ```bash
    curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2Fr -T ./model_input/sample1.txt&
    curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2Fr -T ./model_input/sample2.txt&
    curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2Fr -T ./model_input/sample3.txt&
    curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2Fr -T ./model_input/sample4.txt&
    {
        "input" : "Hello World !!!\n",
        "french_output" : "Bonjour le monde ! ! !"
    }
    {
        "input" : "Hi James, when are you coming back home? I am waiting for you.\nPlease come as soon as possible.\n",
        "french_output" : "Bonjour James, quand rentrerez-vous chez vous, je vous attends et je vous prie de venir le plus tôt possible."
    }
    {
        "input" : "I’m sorry, I don’t remember your name. You are you?\n",
        "french_output" : "Je vous prie de m'excuser, je ne me souviens pas de votre nom."
    }
    {
        "input" : "I’m well. How are you?\nIt’s going well, thank you. How are you doing?\nFine, thanks. And yourself?\n",
        "french_output" : "Je me sens bien. Comment allez-vous ? Ça va bien, merci. Comment allez-vous ?"
    }
    ```

### Steps to configure English-to-German translation model with batch-support

* Start the model server. In this example, we are starting the model server with config.properties file

    ```bash
    torchserve --start --model-store model_store --ts-config config.properties --disable-token-auth  --enable-model-api
    ```

* Now let's launch English_to_French translation model, which we have built to handle batch inference.
In this example, we are going to launch 1 worker which handles a `batch size` of 4 with a `max_batch_delay` of 10s.

    ```bash
    curl -X POST "http://localhost:8081/models?url=TransformerEn2De.mar&initial_workers=1&synchronous=true&batch_size=4&max_batch_delay=10000"
    ```

* Run batch inference command to test the model.

    ```bash
	curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2De -T ./model_input/sample1.txt&
	curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2De -T ./model_input/sample2.txt&
	curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2De -T ./model_input/sample3.txt&
	curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2De -T ./model_input/sample4.txt&
    {
        "input" : "Hello World !!!\n",
        "german_output" : "Hallo Welt!!!"
    }
    {
        "input" : "Hi James, when are you coming back home? I am waiting for you.\nPlease come as soon as possible.\n",
        "german_output" : "Hallo James, wann kommst du nach Hause? Ich warte auf dich. Bitte komm so bald wie möglich."
    }
    {
        "input" : "I’m sorry, I don’t remember your name. You are you?\n",
        "german_output" : "Es tut mir leid, ich erinnere mich nicht an Ihren Namen. Sie sind es?"
    }
    {
        "input" : "I’m well. How are you?\nIt’s going well, thank you. How are you doing?\nFine, thanks. And yourself?\n",
        "german_output" : "Mir geht es gut. Wie geht es Ihnen? Es läuft gut, danke. Wie geht es Ihnen? Gut, danke. Und sich selbst?"
    }
    ```
