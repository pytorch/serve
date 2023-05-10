# [Management API](#management-api)

TorchServe provides the following APIs that allows you to manage models at runtime:

1. [Register a model](#register-a-model)
2. [Increase/decrease number of workers for specific model](#scale-workers)
3. [Describe a model's status](#describe-model)
4. [Unregister a model](#unregister-a-model)
5. [List registered models](#list-models)
6. [Set default version of a model](#set-default-version)

The Management API listens on port 8081 and is only accessible from localhost by default. To change the default setting, see [TorchServe Configuration](./configuration.md).

Similar to the [Inference API](inference_api.md), the Management API provides a [API description](#api-description) to describe management APIs with the OpenAPI 3.0 specification.

Alternatively, if you want to use KServe, TorchServe supports both v1 and v2 API. For more details please look into this [kserve documentation](https://github.com/pytorch/serve/tree/master/kubernetes/kserve)

## Register a model

This API follows the [ManagementAPIsService.RegisterModel](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/management.proto) gRPC API.

`POST /models`

* `url` - Model archive download url. Supports the following locations:
  * a local model archive (.mar); the file must be in the `model_store` folder (and not in a subfolder).
  * a URI using the HTTP(s) protocol. TorchServe can download .mar files from the Internet.
* `model_name` - the name of the model; this name will be used as {model_name} in other APIs as part of the path. If this parameter is not present, `modelName` in MANIFEST.json will be used.
* `handler` - the inference handler entry-point. This value will override `handler` in MANIFEST.json if present. **NOTE: Make sure that the given `handler` is in the `PYTHONPATH`. The format of handler is `module_name:method_name`.**
* `runtime` - the runtime for the model custom service code. This value will override runtime in MANIFEST.json if present. The default value is `PYTHON`.
* `batch_size` - the inference batch size. The default value is `1`.
* `max_batch_delay` - the maximum delay for batch aggregation. The default value is 100 milliseconds.
* `initial_workers` - the number of initial workers to create. The default value is `0`. TorchServe will not run inference until there is at least one work assigned.
* `synchronous` - whether or not the creation of worker is synchronous. The default value is false. TorchServe will create new workers without waiting for acknowledgement that the previous worker is online.
* `response_timeout` - If the model's backend worker doesn't respond with inference response within this timeout period, the worker will be deemed unresponsive and rebooted. The units is seconds. The default value is 120 seconds.

```bash
curl -X POST  "http://localhost:8081/models?url=https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar"

{
  "status": "Model \"squeezenet_v1.1\" Version: 1.0 registered with 0 initial workers. Use scale workers API to add workers for the model."
}
```

### Encrypted model serving
If you'd like to serve an encrypted model then you need to setup [S3 SSE-KMS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingKMSEncryption.html) with the following environment variables:
* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACCESS_KEY
* AWS_DEFAULT_REGION

And set "s3_sse_kms=true" in HTTP request.

For example: model squeezenet1_1 is [encrypted on S3 under your own private account](https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingKMSEncryption.html). The model http url on S3 is `https://torchserve.pytorch.org/sse-test/squeezenet1_1.mar`.
- if torchserve will run on EC2 instance (e.g. OS: ubuntu)
1. add an IAM Role (AWSS3ReadOnlyAccess) for the EC2 instance
2. run ts_scripts/get_aws_credential.sh to export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
3. export AWS_DEFAULT_REGION=your_s3_bucket_region
4. start torchserve
5. Register encrypted model squeezenet1_1 by setting s3_sse_kms=true in curl command.

```bash
curl -X POST  "http://localhost:8081/models?url=https://torchserve.pytorch.org/sse-test/squeezenet1_1.mar&s3_sse_kms=true"

{
  "status": "Model \"squeezenet_v1.1\" Version: 1.0 registered with 0 initial workers. Use scale workers API to add workers for the model."
}
```

- if torchserve will run on local (e.g. OS: macOS)
1. Find your AWS access key and secret key. You can [reset them](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys_retrieve.html) if you forgot the keys.
2. export AWS_ACCESS_KEY_ID=your_aws_access_key
3. export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
4. export AWS_DEFAULT_REGION=your_s3_bucket_region
5. start torchserve
6. Register encrypted model squeezenet1_1 by setting s3_sse_kms=true in curl command (same as EC2 example step 5).

You might want to create workers during registration. because creating initial workers might take some time,
you can choose between synchronous or asynchronous call to make sure initial workers are created properly.

The asynchronous call returns with HTTP code 202 before trying to create workers.

```bash
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=false&url=https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: 4dc54158-c6de-42aa-b5dd-ebcb5f721043
< content-length: 47
< connection: keep-alive
<
{
  "status": "Processing worker updates..."
}
```

The synchronous call returns with HTTP code 200 after all workers have been adjusted.

```bash
curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar"

< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: ecd2e502-382f-4c3b-b425-519fbf6d3b85
< content-length: 89
< connection: keep-alive
<
{
  "status": "Model \"squeezenet1_1\" Version: 1.0 registered with 1 initial workers"
}
```

## Scale workers

This API follows the [ManagementAPIsService.ScaleWorker](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/management.proto) gRPC API. It returns the status of a model in the ModelServer.


`PUT /models/{model_name}`

* `min_worker` - (optional) the minimum number of worker processes. TorchServe will try to maintain this minimum for specified model. The default value is `1`.
* `max_worker` - (optional) the maximum number of worker processes. TorchServe will make no more that this number of workers for the specified model. The default is the same as the setting for `min_worker`.
* `synchronous` - whether or not the call is synchronous. The default value is `false`.
* `timeout` - the specified wait time for a worker to complete all pending requests. If exceeded, the work process will be terminated. Use `0` to terminate the backend worker process immediately. Use `-1` to wait infinitely. The default value is `-1`.

Use the Scale Worker API to dynamically adjust the number of workers for any version of a model to better serve different inference request loads.

There are two different flavors of this API, synchronous and asynchronous.

The asynchronous call will return immediately with HTTP code 202:

```bash
curl -v -X PUT "http://localhost:8081/models/noop?min_worker=3"

< HTTP/1.1 202 Accepted
< content-type: application/json
< x-request-id: 42adc58e-6956-4198-ad07-db6c620c4c1e
< content-length: 47
< connection: keep-alive
<
{
  "status": "Processing worker updates..."
}
```

The synchronous call returns with HTTP code 200 after all workers have been adjusted.

```bash
curl -v -X PUT "http://localhost:8081/models/noop?min_worker=3&synchronous=true"

< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: b72b1ea0-81c6-4cce-92c4-530d3cfe5d4a
< content-length: 63
< connection: keep-alive
<
{
  "status": "Workers scaled to 3 for model: noop"
}
```

To scale workers of a specific version of a model use URI : /models/{model_name}/{version}
`PUT /models/{model_name}/{version}`

The following synchronous call will return after all workers for version "2.0" for model "noop" has be adjusted with HTTP code 200.

```bash
curl -v -X PUT "http://localhost:8081/models/noop/2.0?min_worker=3&synchronous=true"

< HTTP/1.1 200 OK
< content-type: application/json
< x-request-id: 3997ccd4-ae44-4570-b249-e361b08d3d47
< content-length: 77
< connection: keep-alive
<
{
  "status": "Workers scaled to 3 for model: noop, version: 2.0"
}
```

## Describe model

This API follows the [ManagementAPIsService.DescribeModel](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/management.proto) gRPC API. It returns the status of a model in the ModelServer.

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

`GET /models/{model_name}/{model_version}?customized=true`
or
`GET /models/{model_name}?customized=true`

Use the Describe Model API to get detail runtime status and customized metadata of a version of a model:
* Implement function describe_handle. E.g.

```python
    def describe_handle(self):
        """Customized describe handler
        Returns:
            dict : A dictionary response.
        """
        output_describe = None

        logger.info("Collect customized metadata")

        return output_describe
```

* Implement function _is_describe if handler is not inherited from BaseHandler. And then, call _is_describe and describe_handle in handle.

```python
    def _is_describe(self):
        if self.context and self.context.get_request_header(0, "describe"):
            if self.context.get_request_header(0, "describe") == "True":
                return True
        return False

    def handle(self, data, context):
        if self._is_describe():
            output = [self.describe_handle()]
        else:
            data_preprocess = self.preprocess(data)

            if not self._is_explain():
                output = self.inference(data_preprocess)
                output = self.postprocess(output)
            else:
                output = self.explain_handle(data_preprocess, data)

        return output
```

* Call function _is_describe and describe_handle in handle. E.g.

```python
def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.
        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artifacts parameters.
        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            output, _ = self._infer_with_profiler(data=data)
        else:
            if self._is_describe():
                output = [self.describe_handle()]
            else:
                data_preprocess = self.preprocess(data)

                if not self._is_explain():
                    output = self.inference(data_preprocess)
                    output = self.postprocess(output)
                else:
                    output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round(
            (stop_time - start_time) * 1000, 2), None, 'ms')
        return output
```

* Here is an example. "customizedMetadata" shows the metadata from user's model. These metadata can be decoded into a dictionary.

```bash
curl http://localhost:8081/models/noop-customized/1.0?customized=true
[
    {
        "modelName": "noop-customized",
        "modelVersion": "1.0",
        "modelUrl": "noop-customized.mar",
        "runtime": "python",
        "minWorkers": 1,
        "maxWorkers": 1,
        "batchSize": 1,
        "maxBatchDelay": 100,
        "loadedAtStartup": false,
        "workers": [
          {
            "id": "9010",
            "startTime": "2022-02-08T11:03:20.974Z",
            "status": "READY",
            "memoryUsage": 0,
            "pid": 98972,
            "gpu": false,
            "gpuUsage": "N/A"
          }
        ],
        "customizedMetadata": "{\n  \"data1\": \"1\",\n  \"data2\": \"2\"\n}"
     }
]
```

* Decode customizedMetadata on client side. For example:

```python
import requests
import json

response = requests.get('http://localhost:8081/models/noop-customized/?customized=true').json()
customizedMetadata = response[0]['customizedMetadata']
print(customizedMetadata)
```

## Unregister a model

This API follows the [ManagementAPIsService.UnregisterModel](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/management.proto) gRPC API. It returns the status of a model in the ModelServer.

`DELETE /models/{model_name}/{version}`

Use the Unregister Model API to free up system resources by unregistering specific version of a model from TorchServe:

```bash
curl -X DELETE http://localhost:8081/models/noop/1.0

{
  "status": "Model \"noop\" unregistered"
}
```

## List models
This API follows the [ManagementAPIsService.ListModels](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/management.proto) gRPC API. It returns the status of a model in the ModelServer.

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

* [Inference API description output](https://github.com/pytorch/serve/blob/master/frontend/server/src/test/resources/inference_open_api.json)
* [Management API description output](https://github.com/pytorch/serve/blob/master/frontend/server/src/test/resources/management_open_api.json)

## Set Default Version

This API follows the [ManagementAPIsService.SetDefault](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/management.proto) gRPC API. It returns the status of a model in the ModelServer.

`PUT /models/{model_name}/{version}/set-default`

To set any registered version of a model as default version use:

```bash
curl -v -X PUT http://localhost:8081/models/noop/2.0/set-default
```

The out is OpenAPI 3.0.1 json format. You use it to generate client code, see [swagger codegen](https://swagger.io/swagger-codegen/) for detail.
