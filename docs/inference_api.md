# [Inference API](#inference-api)

Inference API is listening on port 8080 and only accessible from localhost by default. To change the default setting, see [TorchServe Configuration](configuration.md).

The TorchServe server supports the following APIs:

* [API Description](#api-description) - Gets a list of available APIs and options
* [Health check API](#health-check-api) - Gets the health status of the running server
* [Predictions API](#predictions-api) - Gets predictions from the served model
* [Explanations API](#explanations-api) - Gets the explanations from the served model
* [KServe Inference API](#kserve-inference-api) - Gets predictions of the served model from KServe
* [KServe Explanations API](#kserve-explanations-api) - Gets explanations of the served model from KServe

## API Description

To view a full list of inference APIs, you can use following command:

```bash
curl -X OPTIONS http://localhost:8080
```

The output is in the OpenAPI 3.0.1 json format. You can use it to generate client code, see [swagger codegen](https://swagger.io/swagger-codegen/) for more details.

* [Inference API description output](https://github.com/pytorch/serve/blob/master/frontend/server/src/test/resources/inference_open_api.json)

## Health check API

This API follows the [InferenceAPIsService.Ping](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/inference.proto) gRPC API. It returns the status of a model in the ModelServer.

TorchServe supports a `ping` API that you can call to check the health status of a running TorchServe server:

```bash
curl http://localhost:8080/ping
```

If the server is running, the response is:

```json
{
  "status": "Healthy"
}
```

"maxRetryTimeoutInSec" (default: 5MIN) can be defined in a model's config yaml file(e.g model-config.yaml). It is the maximum time window of recovering a dead backend worker. A healthy worker can be in the state: WORKER_STARTED, WORKER_MODEL_LOADED, or WORKER_STOPPED within maxRetryTimeoutInSec window. "Ping" endpoint"
* return 200 + json message "healthy": for any model, the number of active workers is equal or larger than the configured minWorkers.
* return 500 + json message "unhealthy": for any model, the number of active workers is less than the configured minWorkers.


## Predictions API

This API follows the [InferenceAPIsService.Predictions](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/inference.proto) gRPC API. It returns the status of a model in the ModelServer.

To get predictions from the default version of each loaded model, make a REST call to `/predictions/{model_name}`:

* POST /predictions/{model_name}

### curl Example

```bash
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg

curl http://localhost:8080/predictions/resnet-18 -T kitten_small.jpg

or:

curl http://localhost:8080/predictions/resnet-18 -F "data=@kitten_small.jpg"
```

To get predictions from the loaded model which expects multiple inputs
```bash
curl http://localhost:8080/predictions/squeezenet1_1 -F 'data=@docs/images/dogs-before.jpg' -F 'data=@docs/images/kitten_small.jpg'

or:

import requests

res = requests.post("http://localhost:8080/predictions/squeezenet1_1", files={'data': open('docs/images/dogs-before.jpg', 'rb'), 'data': open('docs/images/kitten_small.jpg', 'rb')})
```
To get predictions from a specific version of each loaded model, make a REST call to `/predictions/{model_name}/{version}`:

* POST /predictions/{model_name}/{version}

### curl Example

```bash
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg

curl http://localhost:8080/predictions/resnet-18/2.0 -T kitten_small.jpg

or:

curl http://localhost:8080/predictions/resnet-18/2.0 -F "data=@kitten_small.jpg"
```

The result is JSON that tells you that the image is most likely a tabby cat. The highest prediction was:

```json
{
    "class": "n02123045 tabby, tabby cat",
    "probability": 0.42514491081237793
}
```
* Streaming response via HTTP 1.1 chunked encoding
TorchServe the inference API support streaming response to allow a sequence of inference responses to be sent over HTTP 1.1 chunked encoding. This new feature is only recommended for use case when the inference latency of the full response is high and the inference intermediate results are sent to client. An example could be LLMs for generative applications, where generating "n" number of tokens can have high latency, in this case user can receive each generated token once ready until the full response completes. To achieve streaming response, backend handler calls "send_intermediate_predict_response" to send one intermediate result to frontend, and return the last result as the existing style. For example,
```
from ts.protocol.otf_message_handler import send_intermediate_predict_response
def handle(data, context):
    if type(data) is list:
        for i in range (3):
            send_intermediate_predict_response(["intermediate_response"], context.request_ids, "Intermediate Prediction success", 200, context)
        return ["hello world "]
```
Client side receives the chunked data.
```
def test_echo_stream_inference():
    test_utils.start_torchserve(no_config_snapshots=True, gen_mar=False)
    test_utils.register_model('echo_stream',
                              'https://torchserve.pytorch.org/mar_files/echo_stream.mar')

    response = requests.post(TF_INFERENCE_API + '/predictions/echo_stream', data="foo", stream=True)
    assert response.headers['Transfer-Encoding'] == 'chunked'

    prediction = []
    for chunk in (response.iter_content(chunk_size=None)):
        if chunk:
            prediction.append(chunk.decode("utf-8"))

    assert str(" ".join(prediction)) == "hello hello hello hello world "
    test_utils.unregister_model('echo_stream')
```
## Explanations API

Torchserve makes use of Captum's functionality to return the explanations of the models that is served.

To get explanations from the default version of each loaded model, make a REST call to `/explanations/{model_name}`:

* POST /explanations/{model_name}

### curl example
```bash
curl http://127.0.0.1:8080/explanations/mnist -T examples/image_classifier/mnist/test_data/0.png
```

The result is a json that gives you the explanations for the input image
```json
  [
    [
      [
        [
          0.004570948731989492,
          0.006216969640322402,
          0.008197565423679522,
          0.009563574612830427,
          0.008999274832810742,
          0.009673474804303854,
          0.007599905146155397,
          ,
	        ,

        ]
      ]
    ]
  ]
```

## KServe Inference API

Torchserve makes use of KServe Inference API to return the predictions of the models that is served.

To get predictions from the loaded model, make a REST call to `/v1/models/{model_name}:predict`:

* POST /v1/models/{model_name}:predict

### curl example
```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kserve/kf_request_json/v1/mnist.json http://127.0.0.1:8080/v1/models/mnist:predict
```

The result is a json that gives you the predictions for the input json
```json
{
  "predictions": [
    2
  ]
}
```

## KServe Explanations API

Torchserve makes use of KServe API spec to return the explanations for the the models that it served.

To get explanations from the loaded model, make a REST call to `/v1/models/{model_name}:explain`:

* /v1/models/{model_name}:explain

### curl example
```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kserve/kf_request_json/v1/mnist.json http://127.0.0.1:8080/v1/models/mnist:explain
```

The result is a json that gives you the explanations for the input json
```json
{
  "explanations": [
    [
      [
        [
          0.004570948731989492,
          0.006216969640322402,
          0.008197565423679522,
          0.009563574612830427,
          0.008999274832810742,
          0.009673474804303854,
          0.007599905146155397,
          ,
          ,
	        ,
        ]
      ]
    ]
  ]
}
