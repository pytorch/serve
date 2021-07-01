# Inference API

Inference API is listening on port 8080 and only accessible from localhost by default. To change the default setting, see [TorchServe Configuration](configuration.md).

The TorchServe server supports the following APIs:

* [API Description](#api-description) - Gets a list of available APIs and options
* [Health check API](#health-check-api) - Gets the health status of the running server
* [Predictions API](#predictions-api) - Gets predictions from the served model
* [Explanations API](#explanations-api) - Gets the explanations from the served model
* [KFServing Inference API](#kfserving-inference-api) - Gets predictions of the served model from KFServing side
* [KFServing Explanations API](#kfserving-explanations-api) - Gets explanations of the served model from KFServing

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
  "health": "healthy!"
}
```

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

To get predictions from a specific version of each loaded model, make a REST call to `/predictions/{model_name}/{version}`:

* POST /predictions/{model_name}/{version}

## curl Example

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

## KFServing Inference API

Torchserve makes use of KFServing Inference API to return the predictions of the models that is served.

To get predictions from the loaded model, make a REST call to `/v1/models/{model_name}:predict`:

* POST /v1/models/{model_name}:predict

### curl example
```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/mnist.json http://127.0.0.1:8080/v1/models/mnist:predict
```

The result is a json that gives you the predictions for the input json
```json
{
  "predictions": [
    2
  ]
}
```

## KFServing Explanations API

Torchserve makes use of KFServing API spec to return the explanations for the the models that it served.

To get explanations from the loaded model, make a REST call to `/v1/models/{model_name}:explain`:

* /v1/models/{model_name}:explain

### curl example
```bash
 curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/mnist.json http://127.0.0.1:8080/v1/models/mnist:explain
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

