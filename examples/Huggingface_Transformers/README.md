## Serving Huggingface Transformers using TorchServe

In this example, we show how to serve a fine tuned or off the shelf Transformer model from [huggingface](https://huggingface.co/transformers/index.html) using TorchServe. 

We use a custom handler, [Transformer_handler.py](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py).

This handler enables us to use pre-trained transformer models from Huggingface, such as BERT, RoBERTA, XLM for token classification, sequence classification and question answering.

We borrowed ideas to write a custom handler for transformers from tutorial presented in [mnist from image classifiers examples](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist) and the post by [MFreidank](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18).

To get started [install Torchserve](https://github.com/pytorch/serve) and then

 `pip install transformers==4.6.0`

### Objectives
1. How to package a transformer into a torch model archive (.mar) file (eager mode or Torchscript) with `torch-model-archiver`
2. How to load mar file in torch serve to run inferences and explanations using `torchserve`

### **Getting Started with the Demo**

If you're finetuning an existing model then you need to save your model and tokenizer with `save_pretrained()` which will create a `pytorch_model.bin`, `vocab.txt` and `config.json` file. Make sure to create them then run

```
mkdir Transformer_model
mv pytorch_model.bin vocab.txt config.json Transformer_model/
```

If you'd like to download a pretrained model without fine tuning we've provided a simple helper script which will do the above for you. All you need to do is change [setup.config.json](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/setup_config.json) to your liking and run

`python Download_Transformer_models.py`

For Torchscript support, check out [torchscript.md](torchscript.md)

#### Setting the setup_config.json

In the setup_config.json :

*model_name* : bert-base-uncased , roberta-base or other available pre-trained models.

*mode:* `sequence_classification`, `token_classification` or `question_answering`

*do_lower_case* : `true` or `false` which configures the tokenizer

*num_labels* : number of outputs for `sequence_classification`: 2, `token_classification`: 9  or `question_answering`: 0

*save_mode* : "torchscript" or "pretrained", this setting will be used by `Download_transformer_models.py` script as well as the handler, to download/save and load the model in Torchscript or save_pretrained mode respectively.

*max_length* : maximum length for the  input sequences to the models, this will be used in preprocessing of the handler. Also, if you choose to use Torchscript as the serialized model  for packaging your model this length should be equal to the length that has been used during the tracing of the model using torch.jit.trace.

*captum_explanation* : `true` for eager mode models but should be set to `false` for torchscripted models or if you don't need explanations

*embedding_name* : The name of embedding layer in the chosen model, this could be `bert` for `bert-base-uncased`, `roberta` for `roberta-base` or `roberta` for `xlm-roberta-large`.

Once, `setup_config.json` has been set properly, the next step is to run

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
* Examples: `QA_artifacts`, `Seq_classification_artifacts` and `Token_classification_artifacts`


## Sequence Classification

### Create model archive eager mode

```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json,./Seq_classification_artifacts/index_to_name.json"

```

### Create model archive Torchscript mode

```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --extra-files "./setup_config.json,./Seq_classification_artifacts/index_to_name.json"

```

### Register the model

To register the model on TorchServe using the above model archive file, we run the following commands:

```
mkdir model_store
mv BERTSeqClassification.mar model_store/
torchserve --start --model-store model_store --models my_tc=BERTSeqClassification.mar --ncs

```

### Run an inference

To run an inference: `curl -X POST http://127.0.0.1:8080/predictions/my_tc -T Seq_classification_artifacts/sample_text_captum_input.txt`
To get an explanation: `curl -X POST http://127.0.0.1:8080/explanations/my_tc -T Seq_classification_artifacts/sample_text_captum_input.txt`

## Token Classification

Change `setup_config.json` to

```
{
 "model_name":"bert-base-uncased",
 "mode":"token_classification",
 "do_lower_case":true,
 "num_labels":"9",
 "save_mode":"pretrained",
 "max_length":"150",
 "captum_explanation":true,
 "embedding_name": "bert"
}
```

```
rm -r Transformer_model
python Download_Transformer_models.py
```

### Create model archive eager mode
```
torch-model-archiver --model-name BERTTokenClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json,./Token_classification_artifacts/index_to_name.json"
```

### Create model archive Torchscript mode
```
torch-model-archiver --model-name BERTTokenClassification --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --extra-files "./setup_config.json,./Token_classification_artifacts/index_to_name.json"
```

### Register the model

```
mkdir model_store
mv BERTTokenClassification.mar model_store
torchserve --start --model-store model_store --models my_tc=BERTTokenClassification.mar --ncs
```

### Run an inference
To run an inference: `curl -X POST http://127.0.0.1:8080/predictions/my_tc -T Token_classification_artifacts/sample_text_captum_input.txt`
To get an explanation: `curl -X POST http://127.0.0.1:8080/explanations/my_tc -T Token_classification_artifacts/sample_text_captum_input.txt`

## Question Answering

Change `setup_config.json` to
```
{
 "model_name":"distilbert-base-cased-distilled-squad",
 "mode":"question_answering",
 "do_lower_case":true,
 "num_labels":"0",
 "save_mode":"pretrained",
 "max_length":"128",
 "captum_explanation":true,
 "embedding_name": "distilbert"
}
```

```
rm -r Transformer_model
python Download_Transformer_models.py
```

### Create model archive eager mode
```
torch-model-archiver --model-name BERTQA --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json"
```

### Create model archive Torchscript mode
```
torch-model-archiver --model-name BERTQA --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --extra-files "./setup_config.json"
```

### Register the model

```
mkdir model_store
mv BERTQA.mar model_store
torchserve --start --model-store model_store --models my_tc=BERTQA.mar --ncs
```
### Run an inference
To run an inference: `curl -X POST http://127.0.0.1:8080/predictions/my_tc -T QA_artifacts/sample_text_captum_input.txt`
To get an explanation: `curl -X POST http://127.0.0.1:8080/explanations/my_tc -T QA_artifacts/sample_text_captum_input.txt`

## Batch Inference

For batch inference the main difference is that you need set the batch size while registering the model. This can be done either through the management API or if using Torchserve 0.4.1 and above, it can be set through config.properties as well.  Here is an example of setting batch size for sequence classification with management API and through config.properties. You can read more on batch inference in Torchserve [here](https://github.com/pytorch/serve/tree/master/docs/batch_inference_with_ts.md).

* Management API
    ```
    mkdir model_store
    mv BERTSeqClassification.mar model_store/
    torchserve --start --model-store model_store --ncs

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

The [Captum Explanations for Visual Insights Notebook](../../captum/Captum_visualization_for_bert.ipynb) provides a visual example for how model interpretations can help

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

### Running KServe

[BERT Readme for KServe](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/torchserve/bert/README.md).
[End to End KServe document](https://github.com/pytorch/serve/blob/master/kubernetes/kserve/README.md).
