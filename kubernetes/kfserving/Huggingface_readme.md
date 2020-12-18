# Serve a BERT Model for Inference and Explanations on the KFServing side :

In this document, the .mar file creation, request & response on the KFServing side and the KFServing changes to the handler files for BERT Sequence Classification model using a custom handler.


## .mar file creation

Download_Transformer_models.py":

`python serve/examples/Huggingface_Transformers/Download_Transformer_models.py`

This produces all the required files for packaging using a huggingface transformer model off-the-shelf without fine-tuning process. Using this option will create and saved the required files into Transformer_model directory. 

The .mar file creation command is as below:

```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file serve/examples/Huggingface_Transformers/Transformer_model/pytorch_model.bin --handler serve/examples/Huggingface_Transformers/Transformer_model/Transformer_handler_generalized.py --source-vocab serve/examples/Huggingface_Transformers/Transformer_model/vocab.txt --extra-files "Transformer_model/config.json,serve/examples/Huggingface_Transformers/Transformer_model/setup_config.json,serve/examples/Huggingface_Transformers/Transformer_model/index_to_name.json"
```

## Starting Torchserve
To serve an Inference Request for Torchserve using the KFServing Spec, follow the below:

* create a config.properties file and specify the details as shown:
```
service_envelope=kfserving
```
The Service Envelope field is mandatory for Torchserve to process the KFServing Input Request Format.

* start Torchserve by invoking the below command:
```
torchserve --start --model-store model_store --ncs --models bert=BERTSeqClassification.mar

```

## Model Register for KFServing:

Hit the below curl request to register the model

```
curl -X POST "localhost:8081/models?model_name=bert&url=BERTSeqClassification.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
```
Please note that the batch size, the initial worker and synchronous values can be changed at your discretion and they are optional.

## Request and Response

### The curl request for inference is as below:
```
curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/bert.json http://127.0.0.1:8085/v1/models/bert:predict
```

The Prediction response is as below :

```
{
  "predictions": [
    "Accepted"
  ]
}
```
### The curl request for explanations is as below:

```bash
curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/bert.json http://127.0.0.1:8085/v1/models/bert:explain
```

The Explanation response is as below :

```bash
{
  "explanations": [
    {
      "importances": [
        0.0,
        -0.14403946955295752,
        0.27738993789305466,
        -0.07400246982841178,
        0.08229612410414909,
        0.48818893239148936,
        -0.4699206035873072,
        0.48373614214053895,
        -0.2930872942428316,
        -0.32673914053860964,
        -0.06515631495421433,
        0.0
      ],
      "words": [
        "[CLS]",
        "the",
        "recent",
        "climate",
        "change",
        "across",
        "world",
        "is",
        "impact",
        "##ing",
        "negatively",
        "[SEP]"
      ],
      "delta": -0.0010374430790551711
    }
  ]
}
```
### The curl request for the Server Health check 

Server Health check API returns the model's state for inference

The API is as below:

```bash
curl -X GET "http://127.0.0.1:8081/v1/models/bert"
```

The response is as below:

```bash
{
  "name": "bert",
  "ready": true
}
```

## KFServing changes to the handler files



* When you write a handler, always expect a plain Python list containing data ready to go into `preprocess`.

    The bert request difference between the regular torchserve and kfserving is as below

    ### Regular torchserve request:
    ```
    [
      {
        "data": "The recent climate change across world is impacting negatively"
      }
    ]
    ```

    ### KFServing Request:
    ```
    {
      "instances": [
        {
          "data": "The recent climate change across world is impacting negatively"
        }
      ]
    }
    
    ```

    The KFServing request is unwrapped by the kfserving envelope in torchserve  and sent like a torchserve request. So effectively the values of  `instances`  key is sent to the handlers.

        

* The Request data for kfserving  is a batches of dicts as opposed to batches of bytes array(text file) in the regular torchserve.

    So in the preprocess method of [Transformer_handler_generalized.py](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py), KFServing doesn't require the data to be utf-8 decoded for text inputs, hence the code was modified to ensure that Torchserve Input Requests which are sent as text file are only utf-8 decoded and not for the KFServing Input Requests.