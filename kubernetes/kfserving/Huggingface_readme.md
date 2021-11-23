# Serve a BERT Model for Inference and Explanations on the KFServing side :

In this document, the .mar file creation, request & response on the KFServing side and the KFServing changes to the handler files for BERT Sequence Classification model using a custom handler.


## .mar file creation

Download_Transformer_models.py:

`python serve/examples/Huggingface_Transformers/Download_Transformer_models.py`

This produces all the required files for packaging using a huggingface transformer model off-the-shelf without fine-tuning process. Using this option will create and saved the required files into Transformer_model directory. 

The .mar file creation command is as below:

```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file serve/examples/Huggingface_Transformers/Transformer_model/pytorch_model.bin --handler serve/examples/Huggingface_Transformers/Transformer_model/Transformer_handler_generalized.py --extra-files "serve/examples/Huggingface_Transformers/Transformer_model/vocab.txt,serve/examples/Huggingface_Transformers/Transformer_model/config.json,serve/examples/Huggingface_Transformers/Transformer_model/setup_config.json,serve/examples/Huggingface_Transformers/Transformer_model/index_to_name.json"
```

## Starting Torchserve for KFServing Predictor
To serve an Inference Request for Torchserve using the KFServing Spec, follow the below:

* create a config.properties file and specify the details as shown:

```bash
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8085
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_format=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
model_store=/mnt/models/model-store
```

* Set service envelope environment variable

The 
```export TS_SERVICE_ENVELOPE=kfserving``` or ```export TS_SERVICE_ENVELOPE=kfservingv2``` envvar is for choosing between KFServing v1 and v2 protocols. This is set by the controller in KFServing cluster.

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

When the curl request is made, ensure that the request is made inisde of the serve folder.
```
curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/bert.json http://127.0.0.1:8085/v1/models/bert:predict
```

The Prediction response is as below :

```json
{
  "predictions": [
    "Accepted"
  ]
}
```
### The curl request for explanations is as below:

Torchserve supports KFServing Captum Explanations for Eager Models only.

```bash
curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/bert.json http://127.0.0.1:8085/v1/models/bert:explain
```

The Explanation response is as below :

```json
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

KFServing supports Static batching by adding new examples in the instances key of the request json
But the batch size should still be set at 1, when we register the model. Explain doesn't support batching.

```json
{
  "instances": [
    {
      "data": "Bloomberg has reported on the economy"
    },
    {
      "data": "Bloomberg has reported on the economy"
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

```json
{
  "name": "bert",
  "ready": true
}
```

## KFServing changes to the handler files



* When you write a handler, always expect a plain Python list containing data ready to go into `preprocess`.

    The bert request difference between the regular torchserve and kfserving is as below

    ### Regular torchserve request:
    ```json
    [
      {
        "data": "The recent climate change across world is impacting negatively"
      }
    ]
    ```

    ### KFServing Request:
    ```json
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
    