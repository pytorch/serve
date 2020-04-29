# Text classification using Transformers

In this example, we show how to use a Fine_tuned or off-the-shelf Transformer model to perform real time text classification with TorchServe. We borrowed ideas to write a custom handler for transformers from tutorial presented in [mnist from image classifiers examples](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist) and the post by [MFreidank](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18).

The inference service would return the label/ class inferred by the model from the text.

We need to make sure that have installed the Transformers, it can be installed using the following  command.

 `pip install transformers`

# Objective
1. Demonstrate how to package a transformer model with custom handler into torch model archive (.mar) file
2. Demonstrate how to load model archive (.mar) file into TorchServe and run inference.

# Serve a custom model on TorchServe

 * Step - 1: In this step, we provide the requirements for packaging a model. We use the following [Colab notebook](https://drive.google.com/open?id=1p3v-JjNi8xfE8vGd-Jhzisi1ztNLdbTb) to fine-tune BERT from Transformers on [The Corpus of Linguistic Acceptability (COLA)](https://nyu-mll.github.io/CoLA/). After running the Colab notebook, pytorch_model.bin file along with vocab.txt and config.json will be saved on your google drive that can be downloaded later into current directory. After downloading, we create Transformer_model directory and move the files to this directory.

 `mkdir Transformer_model`

 `mv pytorch_model.bin vocab.txt config.json Transformer_model`

 There is another option just for demonstration purposes, that you can run the following:

`python Download_Transformer_models.py`

 This produces all the required files for packaging using off-the-shelf BERT model without fine-tuning process. Using this option will create and saved the required files into Transformer_model directory.

 * Step - 2: Create a torch model archive using the torch-model-archiver utility to archive the above files along with [custom handler](Transformers_handler.py) for transfromers.

    ```bash
    torch-model-archiver --model-name BertSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformers_handler.py --extra-files "./index_to_name.json,Transformer_model/vocab.txt,Transformer_model/config.json"
    ```

 * Step - 3: Register the model on TorchServe using the above model archive file and run text classification.

    ```bash
    mkdir model_store
    mv BertSeqClassification.mar model_store/
    torchserve --start --model-store model_store --models my_tc=BertSeqClassification.mar

    ```
* Step - 4: open a new terminal and run:
`curl -X POST http://127.0.0.1:8080/predictions/my_tc -T ./sample_text.txt`
