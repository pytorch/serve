# Serve a Text Classification Model in KServe for Inference and Explanations:

In this document, the .mar file creation, request & response on the KServe side and the KServe changes to the handler files for Text Classification model using Torchserve's default text handler.

## .mar file creation

Before the mar file creation process, the user needs to create the source_vocab.pt file and the model.pt file
Navigate to serve/examples/text_classification and run the below command inside that directory path

```bash
./run_script.sh
```

This creates the model.pt and source_vocab.pt file

The .mar file creation command is as below:

```bash
torch-model-archiver --model-name my_text_classifier --version 1.0 --model-file serve/examples/text_classification/model.py --serialized-file serve/examples/text_classification/model.pt --handler text_classifier --extra-files "serve/examples/text_classification/index_to_name.json,serve/examples/text_classification/source_vocab.pt"
```

## Starting Torchserve for KServe Predictor

To serve an Inference Request for Torchserve using the KServe Spec, follow the below:

- create a config.properties file and specify the details as shown:

```
inference_address=http://127.0.0.0:8085
management_address=http://127.0.0.0:8081
number_of_netty_threads=4
enable_envvars_config=true
job_queue_size=10
model_store=model-store
```

- Set service envelope environment variable

The
`export TS_SERVICE_ENVELOPE=kserve` or `export TS_SERVICE_ENVELOPE=kservev2` envvar is for choosing between
KServe v1 and v2 protocols

- start Torchserve by invoking the below command:

```
torchserve --start --model-store model_store --ncs --models my_tc=my_text_classifier.mar

```

## Model Register for KServe:

Hit the below curl request to register the model

```
curl -X POST "localhost:8081/models?model_name=my_tc&url=my_text_classifier.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
```

Please note that the batch size, the initial worker and synchronous values can be changed at your discretion and they are optional.

## Request and Response

### The curl request is as below for predict:

When the curl request is made, ensure that the request is made inisde of the serve folder.

```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kserve/kf_request_json/text_classifier.json http://127.0.0.1:8085/v1/models/my_tc:predict
```

The Prediction response is as below :

```json
{
	"predictions" : [
	   {
		"World" : 0.0036
		"Sports" : 0.0065
		"Business" :0.9160
		"Sci/Tec" :0.079
	   }
	]
}
```

### The curl request is as below for explain:

Torchserve supports KServe Captum Explanations for Eager Models only.

```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kserve/kf_request_json/text_classifier.json http://127.0.0.1:8085/v1/models/my_tc:explain
```

The Explanation response is as below :

```json
{
  "explanations": [
    {
      "importances": [
        [
          0.00017786371265739233,
          0.9824386919377469,
          4.193646962600815e-6,
          0.00014836451651210265,
          6.149280398342056e-5,
          ,
          ,
        ]
      ],
      "words": ["bloomberg", "has", "reported", "on", "the", "economy"]
    }
  ]
}
```

KServe supports Static batching by adding new examples in the instances key of the request json.
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

```bash
curl -X GET "http://127.0.0.1:8081/v1/models/my_tc"
```

The response is as below:

```json
{
  "name": "my_tc",
  "ready": true
}
```

## KServe changes to the handler files

- When you write a handler, always expect a plain Python list containing data ready to go into `preprocess`.

       The text classifier request difference between the regular torchserve and kserve is as below

### Regular torchserve request:

    ```json
    [
    	{
    		"data" : "The recent climate change across world is impacting negatively"
    	}
    ]
    ```

    ###	KServe Request:
    ```json
    {
    "instances": [
    	{
    		"data": "The recent climate change across world is impacting negatively"
    	}
    ]
    }
    ```

    The KServe request is unwrapped by the kserve envelope in torchserve  and sent like a torchserve request. So effectively the values of  `instances`  key is sent to the handlers.

- The Request data for kserve is a batches of dicts as opposed to batches of bytes array(text file) in the regular torchserve.

  So in the preprocess method of [text_classifier.py](https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_classifier.py) KServe doesnt require the data to be utf-8 decoded for text inputs, hence the code was modified to ensure that Torchserve Input Requests which are sent as text file are only utf-8 decoded and not for the KServe Input Requests.

NOTE :
The current default model for text classification uses EmbeddingBag which Computes sums or means of ‘bags’ of embeddings, without instantiating the intermediate embedding, so it returns the captum explanations on a sentence embedding level and not on a word embedding level.
