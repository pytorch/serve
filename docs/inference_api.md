# Inference API

Inference API is listening on port 8080 and only accessible from localhost by default. To change the default setting, see [TorchServe Configuration](configuration.md).

The TorchServe server supports the following APIs:

* [API Description](#api-description) - Gets a list of available APIs and options
* [Health check API](#health-check-api) - Gets the health status of the running server
* [Predictions API](#predictions-api) - Gets predictions from the served model

## API Description

To view a full list of inference APIs, you can use following command:

```bash
curl -X OPTIONS http://localhost:8080
```

The out is OpenAPI 3.0.1 json format. You can use it to generate client code, see [swagger codegen](https://swagger.io/swagger-codegen/) for detail.

* [Inference API description output](../frontend/server/src/test/resources/inference_open_api.json)

## Health check API

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

To get predictions from the default version of each loaded model, make a REST call to `/predictions/{model_name}`:

* POST /predictions/{model_name}

### curl Example

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg

curl http://localhost:8080/predictions/resnet-18 -T kitten.jpg

or:

curl http://localhost:8080/predictions/resnet-18 -F "data=@kitten.jpg"
```

To get predictions from a specific version of each loaded model, make a REST call to `/predictions/{model_name}/{version}`:

* POST /predictions/{model_name}/{version}

## curl Example

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg

curl http://localhost:8080/predictions/resnet-18/2.0 -T kitten.jpg

or:

curl http://localhost:8080/predictions/resnet-18/2.0 -F "data=@kitten.jpg"
```

The result is JSON that tells you that the image is most likely a tabby cat. The highest prediction was:

```json
{
    "class": "n02123045 tabby, tabby cat",
    "probability": 0.42514491081237793
}
```
