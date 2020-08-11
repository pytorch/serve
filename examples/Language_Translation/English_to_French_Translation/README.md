# English-to-French translation using Fairseq Transformer model
The Transformer, introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762), is a powerful sequence-to-sequence modeling architecture capable of producing state-of-the-art neural machine translation (NMT) systems.

Recently, the fairseq team has explored large-scale semi-supervised training of Transformers using back-translated data, further improving translation quality over the original model. More details can be found in [this blog post](https://engineering.fb.com/ai-research/scaling-neural-machine-translation-to-bigger-data-sets-with-faster-training-and-inference/).

In this example, we have show how to serve a [English-to-French Translation](https://pytorch.org/hub/pytorch_fairseq_translation/) model using TorchServe. We have used a custom handler, model_handler.py which enables us to use pre-trained [Transformer_WMT14_En-Fr](https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md) models from [fairseq](https://github.com/pytorch/fairseq). 

## Objectives
1. Demonstrate how to package a pre-trained English-to-French translation model with custom handler into torch model archive (.mar) file
2. Demonstrate how to load model archive (.mar) file into TorchServe and run inference.

## Install pip dependencies using following commands
We require a few additional Python dependencies for preprocessing:
```bash
pip install fastBPE regex requests sacremoses subword_nmt
```
## Serve the English-to-French Translation model on TorchServe

* Generate the model archive (.mar) file for English-to-French translation model using following command

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
    curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=TransformerEn2Fr.mar"
    ```
    Note:- Our model works only for one worker.

* To get the inference use the following curl command

    ```bash
    curl http://127.0.0.1:8080/predictions/TransformerEn2Fr -T sample.txt
    ```
    Here sample.txt contains simple english sentences which are given as input to [Inference API](https://github.com/pytorch/serve/blob/master/docs/inference_api.md#predictions-api). The output of above curl command will be the french translation of sentences present in the sample.txt file.
