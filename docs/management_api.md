# Management API

TorchServe provides the following APIs that allows you to manage models at runtime:

1. [Register a model](#register-a-model)
2. [Increase/decrease number of workers for specific model](#scale-workers)
3. [Describe a model's status](#describe-model)
4. [Unregister a model](#unregister-a-model)
5. [List registered models](#list-models)
6. [Set default version of a model](#set-default-version)

The Management API listens on port 8081 and is only accessible from localhost by default. To change the default setting, see [TorchServe Configuration](configuration.md).

Similar to the [Inference API](inference_api.md), the Management API provides a [API description](#api-description) to describe management APIs with the OpenAPI 3.0 specification.

## Register a model

`POST /models`

* `url` - Model archive download url. Supports the following locations:
  * a local model archive (.mar); the file must be in the `model_store` folder (and not in a subfolder).
  * a URI using the HTTP(s) protocol. TorchServe can download .mar files from the Internet.
* `model_name` - the name of the model; this name will be used as {model_name} in other APIs as part of the path. If this parameter is not present, `modelName` in MANIFEST.json will be used.
* `handler` - the inference handler entry-point. This value will override `handler` in MANIFEST.json if present. **NOTE: Make sure that the given `handler` is in the `PYTHONPATH`. The format of handler is `module_name:method_name`.**
* runtime - the runtime for the model custom service code. This value will override runtime in MANIFEST.json if present. The default value is `PYTHON`.
* batch_size - the inference batch size. The default value is `1`.
* max_batch_delay - the maximum delay for batch aggregation. The default value is 100 milliseconds.
* initial_workers - the number of initial workers to create. The default value is `0`. TorchServe will not run inference until there is at least one work assigned.
* synchronous - whether or not the creation of worker is synchronous. The default value is false. TorchServe will create new workers without waiting for acknowledgement that the previous worker is online.
* response_timeout - If the model's backend worker doesn't respond with inference response within this timeout period, the worker will be deemed unresponsive and rebooted. The units is seconds. The default value is 120 seconds.

```bash
curl -X POST "http://localhost:8081/models?url=https://<s3_path>/squeezenet_v1.1.mar"

{
  "status": "Model \"squeezenet_v1.1\" registered"
}
```

You might want to create workers during registration. because creating initial workers might take some time,
you can choose between synchronous or asynchronous call to make sure initial workers are created properly.

The asynchronous call returns with HTTP code 202 before trying to create workers.

```bash
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=false&url=https://<s3_path>/squeezenet_v1.1.mar"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: 29cde8a4-898e-48df-afef-f1a827a3cbc2
< content-length: 33
< connection: keep-alive
<
{
  "status": "Worker updated"
}
```

The synchronous call returns with HTTP code 200 after all workers have been adjusted.

```bash
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=https://<s3_path>/squeezenet_v1.1.mar"

< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: c4b2804e-42b1-4d6f-9e8f-1e8901fc2c6c
< content-length: 32
< connection: keep-alive
<
{
  "status": "Worker scaled"
}
```

## Scale workers

`PUT /models/{model_name}`

* `min_worker` - (optional) the minimum number of worker processes. TorchServe will try to maintain this minimum for specified model. The default value is `1`.
* `max_worker` - (optional) the maximum number of worker processes. TorchServe will make no more that this number of workers for the specified model. The default is the same as the setting for `min_worker`.
* `number_gpu` - (optional) the number of GPU worker processes to create. The default value is `0`. If number_gpu exceeds the number of available GPUs, the rest of workers will run on CPU.
* `synchronous` - whether or not the call is synchronous. The default value is `false`.
* `timeout` - the specified wait time for a worker to complete all pending requests. If exceeded, the work process will be terminated. Use `0` to terminate the backend worker process immediately. Use `-1` to wait infinitely. The default value is `-1`. 

Use the Scale Worker API to dynamically adjust the number of workers for any version of a model to better serve different inference request loads.

There are two different flavors of this API, synchronous and asynchronous.

The asynchronous call will return immediately with HTTP code 202:

```bash
curl -v -X PUT "http://localhost:8081/models/noop?min_worker=3"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: 74b65aab-dea8-470c-bb7a-5a186c7ddee6
< content-length: 33
< connection: keep-alive
<
{
  "status": "Worker updated"
}
```

The synchronous call returns with HTTP code 200 after all workers have been adjusted.

```bash
curl -v -X PUT "http://localhost:8081/models/noop?min_worker=3&synchronous=true"

< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: c4b2804e-42b1-4d6f-9e8f-1e8901fc2c6c
< content-length: 32
< connection: keep-alive
< 
{
  "status": "Worker scaled"
}
```

To scale workers of a specific version of a model use URI : /models/{model_name}/{version}
`PUT /models/{model_name}/{version}`

The following synchronous call will return after all workers for version "2.0" for model "noop" has be adjusted with HTTP code 200.

```bash
curl -v -X PUT "http://localhost:8081/models/noop/2.0?min_worker=3&synchronous=true"

< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: c4b2804e-42b1-4d6f-9e8f-1e8901fc2c6c
< content-length: 32
< connection: keep-alive
< 
{
  "status": "Worker scaled"
}
```

## Describe model

`GET /models/{model_name}`

Use the Describe Model API to get detail runtime status of default version of a model:

```bash
curl http://localhost:8081/models/noop
[
    {
      "modelName": "noop",
      "modelVersion": "1.0",
      "modelUrl": "noop.mar",
      "engine": "Torch",
      "runtime": "python",
      "minWorkers": 1,
      "maxWorkers": 1,
      "batchSize": 1,
      "maxBatchDelay": 100,
      "workers": [
        {
          "id": "9000",
          "startTime": "2018-10-02T13:44:53.034Z",
          "status": "READY",
          "gpu": false,
          "memoryUsage": 89247744
        }
      ]
    }
]
```

`GET /models/{model_name}/{version}`

Use the Describe Model API to get detail runtime status of specific version of a model:

```bash
curl http://localhost:8081/models/noop/2.0
[
    {
      "modelName": "noop",
      "modelVersion": "2.0",
      "modelUrl": "noop_2.mar",
      "engine": "Torch",
      "runtime": "python",
      "minWorkers": 1,
      "maxWorkers": 1,
      "batchSize": 1,
      "maxBatchDelay": 100,
      "workers": [
        {
          "id": "9000",
          "startTime": "2018-10-02T13:44:53.034Z",
          "status": "READY",
          "gpu": false,
          "memoryUsage": 89247744
        }
      ]
    }
]
```

`GET /models/{model_name}/all`

Use the Describe Model API to get detail runtime status of all version of a model:

```bash
curl http://localhost:8081/models/noop/all
[
    {
      "modelName": "noop",
      "modelVersion": "1.0",
      "modelUrl": "noop.mar",
      "engine": "Torch",
      "runtime": "python",
      "minWorkers": 1,
      "maxWorkers": 1,
      "batchSize": 1,
      "maxBatchDelay": 100,
      "workers": [
        {
          "id": "9000",
          "startTime": "2018-10-02T13:44:53.034Z",
          "status": "READY",
          "gpu": false,
          "memoryUsage": 89247744
        }
      ]
    },
    {
      "modelName": "noop",
      "modelVersion": "2.0",
      "modelUrl": "noop_2.mar",
      "engine": "Torch",
      "runtime": "python",
      "minWorkers": 1,
      "maxWorkers": 1,
      "batchSize": 1,
      "maxBatchDelay": 100,
      "workers": [
        {
          "id": "9000",
          "startTime": "2018-10-02T13:44:53.034Z",
          "status": "READY",
          "gpu": false,
          "memoryUsage": 89247744
        }
      ]
    }
]
```

## Unregister a model

`DELETE /models/{model_name}/{version}`

Use the Unregister Model API to free up system resources by unregistering specific version of a model from TorchServe:

```bash
curl -X DELETE http://localhost:8081/models/noop/1.0

{
  "status": "Model \"noop\" unregistered"
}
```

## List models

`GET /models`

* `limit` - (optional) the maximum number of items to return. It is passed as a query parameter. The default value is `100`.
* `next_page_token` - (optional) queries for next page. It is passed as a query parameter. This value is return by a previous API call.

Use the Models API to query default versions of current registered models:

```bash
curl "http://localhost:8081/models"
```

This API supports pagination:

```bash
curl "http://localhost:8081/models?limit=2&next_page_token=2"

{
  "nextPageToken": "4",
  "models": [
    {
      "modelName": "noop",
      "modelUrl": "noop-v1.0"
    },
    {
      "modelName": "noop_v0.1",
      "modelUrl": "noop-v0.1"
    }
  ]
}
```


## API Description

`OPTIONS /`

To view a full list of inference and management APIs, you can use following command:

```bash
# To view all inference APIs:
curl -X OPTIONS http://localhost:8080

# To view all management APIs:
curl -X OPTIONS http://localhost:8081
```

The out is OpenAPI 3.0.1 json format. You use it to generate client code, see [swagger codegen](https://swagger.io/swagger-codegen/) for detail.

Example outputs of the Inference and Management APIs:

* [Inference API description output](../frontend/server/src/test/resources/inference_open_api.json)
* [Management API description output](../frontend/server/src/test/resources/management_open_api.json)

## Set Default Version

`PUT /models/{model_name}/{version}/set-default`

To set any registered version of a model as default version use:

```bash
curl -v -X PUT http://localhost:8081/models/noop/2.0/set-default
```

The out is OpenAPI 3.0.1 json format. You use it to generate client code, see [swagger codegen](https://swagger.io/swagger-codegen/) for detail.

Example outputs of the Inference and Management APIs:

* [Inference API description output](../frontend/server/src/test/resources/inference_open_api.json)
* [Management API description output](../frontend/server/src/test/resources/management_open_api.json)
