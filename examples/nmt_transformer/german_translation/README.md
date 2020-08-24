# English-to-German translation using Fairseq Transformer model
The Transformer, introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), is a powerful sequence-to-sequence modeling architecture capable of producing state-of-the-art neural machine translation (NMT) systems.

Recently, the [fairseq](https://github.com/pytorch/fairseq#join-the-fairseq-community) team has explored large-scale semi-supervised training of Transformers using back-translated data, further improving translation quality over the original model. More details can be found in [this blog post](https://engineering.fb.com/ai-research/scaling-neural-machine-translation-to-bigger-data-sets-with-faster-training-and-inference/).

In this example, we have show how to serve a [English-to-German Translation](https://pytorch.org/hub/pytorch_fairseq_translation/#english-to-german-translation) model using TorchServe. We have used a custom handler, model_handler.py which enables us to use pre-trained [Transformer_WMT14_En-De](https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md) models from [fairseq](https://github.com/pytorch/fairseq). 

## Objectives
1. Demonstrate how to package a pre-trained English-to-German translation model with custom handler into torch model archive (.mar) file
2. Demonstrate how to load model archive (.mar) file into TorchServe and run inference.

## Serve the English-to-German Translation model on TorchServe

* Generate the model archive (.mar) file for English-to-German translation model using following command

    ```bash
    ./create_mar.sh
    ```
    By execuing the above script file, the the model archive (.mar) file will be auto generated in "model_store" folder in the same working directory.


* Start the TorchServe using the model archive (.mar) file created in above step

    ```bash
    torchserve --start --model-store model_store --ts-config config.properties
    ```
    Note:- Our model requires "fairseq" transformer modules to load the model from the model archive file. So we have added faiseq repo build as an additional dependency in requirements.txt file while creating the model archieve (.mar) file. And to make these additional dependency avaialble to each model we need to set "install_py_dep_per_model" property as "true" in config.properties file, by default this property is set to "false".


* Use [Management API](https://github.com/pytorch/serve/blob/master/docs/management_api.md#management-api) to register the model with one initial worker

    ```bash
    curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=TransformerEn2De.mar"
    ```
    Note:- Our model works only for one worker.

* To get the inference use the following curl command

    ```bash
    curl http://127.0.0.1:8080/predictions/TransformerEn2De -T sample.txt
    ```
    Here sample.txt contains simple english sentences which are given as input to [Inference API](https://github.com/pytorch/serve/blob/master/docs/inference_api.md#predictions-api). The output of above curl command will be the german translation of sentences present in the sample.txt file.

## Batch Inference with TorchServe using English_to_German Translation model

To support batch inference, TorchServe needs the following:

1. TorchServe model configuration: Configure `batch_size` and `max_batch_delay` by using the  "POST /models" [Management API](https://github.com/pytorch/serve/blob/master/docs/management_api.md#management-api).
   TorchServe needs to know the maximum batch size that the model can handle and the maximum time that TorchServe should wait to fill each batch request.
2. Model handler code: TorchServe requires the Model handler to handle batch inference requests.

### TorchServe Model Configuration

To configure TorchServe to use the batching feature, provide the batch configuration information through "POST /models" API.

The configuration that we are interested in is the following:

1. `batch_size`: This is the maximum batch size that a model is expected to handle.
2. `max_batch_delay`: This is the maximum batch delay time TorchServe waits to receive `batch_size` number of requests. If TorchServe doesn't receive `batch_size` number of
requests before this timer time's out, it sends what ever requests that were received to the model `handler`.

### Demo to configure TorchServe with batch-supported model

* Start the model server. In this example, we are starting the model server with config.properties file

    ```bash
    torchserve --start --model-store model_store --ts-config config.properties
    ```

* Now let's launch English_to_German translation model, which we have built to handle batch inference. 
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
    ```
