## Serving Huggingface Transformers using TorchServe

In this example, we show how to serve a Fine_tuned or off-the-shelf Transformer model from [huggingface](https://huggingface.co/transformers/index.html) using TorchServe. We use a custom handler, Transformer_handler.py. This handler enables us to use pre-trained transformer models from Hugginface, such as BERT, RoBERTA, XLM, etc. for use-cases defined in the AutoModel class such as AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForTokenClassification, and AutoModelWithLMHead will be added later. 

We borrowed ideas to write a custom handler for transformers from tutorial presented in [mnist from image classifiers examples](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist) and the post by [MFreidank](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18).

First, we need to make sure that have installed the Transformers, it can be installed using the following  command.

 `pip install transformers`

### Objectives
1. Demonstrate how to package a transformer model with custom handler into torch model archive (.mar) file
2. Demonstrate how to load model archive (.mar) file into TorchServe and run inference.

### Serving a Model using TorchServe

To serve a model using TrochServe following steps are required.

-  Frist, preparing the requirements for packaging a model, including serialized model, and other required files. 
- Create a torch model archive using the "torch-model-archiver" utility to archive the above files along with  a handler ( in this example custom handler for transformers) . 
- Register the model on TorchServe using the above model archive file and run the inference.

### **Getting Started with the Demo**

There are two paths to obtain the required model files for this demo. 

- **Option A** : To yield desired results, one should fine-tuned each of the intended models to use before hand and saving the model and tokenizer using "save_pretrained() ". This will result in pytorch_model.bin file along with vocab.txt and config.json files. These files should be moved to a folder named "Transformer_model" in the current directory. 

- **Option B**: There is another option just for demonstration purposes, to simply run "Download_Transformer_models.py", . The  "Download_Transformer_models.py" script loads and saves the required files mentioned above in  "Transformer_model" directory, using a setup config file, "setup_config.json". Also, settings in  "setup_config.json", are used in the handler, "Transformer_handler_generalized.py", as well to operate on the selected mode and other related settings. 

#### Setting the setup_config.json

In the setup_config.json :

*model_name* : bert-base-uncased , roberta-base or other available pre-trained models.

*mode:* "sequence_classification "for sequence classification, "question_answering "for question answering and "token_classification" for token classification. 

*do_lower_case* : True or False for use of the Tokenizer.

*num_labels* : number of outputs for "sequence_classification", or "token_classification". 

Once, setup_config.json has been set properly, the next step is to run " Download_Transformer_models.py":

`python Download_Transformer_models.py`

This produces all the required files for packaging using a huggingface transformer model off-the-shelf without fine-tuning process. Using this option will create and saved the required files into Transformer_model directory. In case, the "vocab.txt" was not saved into this directory, we can load the tokenizer from pre-trained model vocab, this case has been addressed in the handler. 



#### Setting the extra_files

There are few files that are used for model packaging and at the inference time. "index_to_name.json"  is passed as extra file to the model archiver and used for mapping predictions to labels. "sample_text.txt", is used at the inference time to pass the text that we want to get the inference on. 

index_to_name.json for question answering is not required. 

If intended to use Transformer handler for Token classification, the index_to_name.json should be formatted as follows for example:

`{"label_list":"[O, B-MISC, I-MISC, B-PER,I-PER,B-ORG,I-ORG,B-LOC,I-LOC]"}`

To use Transformer handler for question answering, the sample_text.txt should be formatted as follows:

`{"question" :"Who was Jim Henson?", "context": "Jim Henson was a nice puppet"}`

"question" represents the question to be asked from the source text named as "context" here. 

### Creating a torch Model Archive

Once, setup_config.json,  sample_text.txt and  index_to_name.json are set properly, we can go ahead and package the model and start serving it. The current setting in "setup_config.json" is based on "roberta_base " model for question answering. To fine-tuned RoBERTa can be obtained from running [squad example](https://huggingface.co/transformers/examples.html#squad) from huggingface. Alternatively a fine_tuned BERT model can be used by setting "model_name" to "bert-large-uncased-whole-word-masking-finetuned-squad" in the "setup_config.json".

```
torch-model-archiver --model-name RobertaQA --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json"

```

### Registering the Model on TorchServe and Running Inference

To register the model on TorchServe using the above model archive file, we run the following commands:

```
mkdir model_store
mv RobertaQA.mar model_store/
torchserve --start --model-store model_store --models my_tc=RobertaQA.mar

```

- To run the inference using our registered model, open a new terminal and run: `curl -X POST http://127.0.0.1:8080/predictions/my_tc -T ./sample_text.txt`