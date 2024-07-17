# TorchServe gRPC API

__Note__: Current TorchServe gRPC does not support workflow.

TorchServe also supports [gRPC APIs](https://github.com/pytorch/serve/tree/master/frontend/server/src/main/resources/proto) for both inference and management calls.

TorchServe provides following gRPCs apis

* [Inference API](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/inference.proto)
  - **Ping** : Gets the health status of the running server
  - **Predictions** : Gets predictions from the served model
  - **StreamPredictions** : Gets server side streaming predictions from the saved model

For all Inference API requests, TorchServe requires the correct Inference token to be included or token authorization must be disable. For more details see [token authorization documentation](./token_authorization_api.md)

* [Management API](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/management.proto)
  - **RegisterModel** : Serve a model/model-version on TorchServe
  - **UnregisterModel** : Free up system resources by unregistering specific version of a model from TorchServe
  - **ScaleWorker** : Dynamically adjust the number of workers for any version of a model to better serve different inference request loads.
  - **ListModels** : Query default versions of current registered models
  - **DescribeModel** : Get detail runtime status of default version of a model
  - **SetDefault** : Set any registered version of a model as default version

For all Management API requests, TorchServe requires the correct Management token to be included or token authorization must be disabled. For more details see [token authorization documentation](./token_authorization_api.md)

By default, TorchServe listens on port 7070 for the gRPC Inference API and 7071 for the gRPC Management API on localhost.
To configure gRPC APIs on different addresses and ports refer [configuration documentation](configuration.md)

## Python client example for gRPC APIs

Run following commands to Register, run inference and unregister, densenet161 model from [TorchServe model zoo](model_zoo.md) using [gRPC python client](https://github.com/pytorch/serve/blob/master/ts_scripts/torchserve_grpc_client.py).

 - [Install TorchServe](../README.md)

 - Clone serve repo to run this example

```bash
git clone --recurse-submodules https://github.com/pytorch/serve
cd serve
```

 - Install gRPC python dependencies

```bash
pip install -U grpcio protobuf grpcio-tools googleapis-common-protos
```

 - Start torchServe

```bash
mkdir models
torchserve --start --disable-token-auth --enable-model-api --model-store models/
```

 - Generate python gRPC client stub using the proto files

```bash
python -m grpc_tools.protoc -I third_party/google/rpc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto
```

 - Register densenet161 model

__Note__: To use this API after TorchServe starts, model API control has to be enabled. Add `--enable-model-api` to command line when starting TorchServe to enable the use of this API. For more details see [model API control](./model_api_control.md)

If token authorization is disabled, use:
```bash
python ts_scripts/torchserve_grpc_client.py register densenet161
```

If token authorization is enabled, use:
```bash
python ts_scripts/torchserve_grpc_client.py register densenet161 --auth-token <management-token>
```

 - Run inference using

If token authorization is disabled, use:
```bash
python ts_scripts/torchserve_grpc_client.py infer densenet161 examples/image_classifier/kitten.jpg
```

If token authorization is enabled, use:
```bash
python ts_scripts/torchserve_grpc_client.py infer densenet161 examples/image_classifier/kitten.jpg --auth-token <inference-token>
```

 - Unregister densenet161 model

__Note__: To use this API after TorchServe starts, model API control has to be enabled. Add `--enable-model-api` to command line when starting TorchServe to enable the use of this API. For more details see [model API control](./model_api_control.md)

If token authorization is disabled, use:
```bash
python ts_scripts/torchserve_grpc_client.py unregister densenet161
```

If token authorization is enabled, use:
```bash
python ts_scripts/torchserve_grpc_client.py unregister densenet161 --auth-token <management-token>
```

## GRPC Server Side Streaming
TorchServe GRPC APIs adds a server side streaming of the inference API "StreamPredictions" to allow a sequence of inference responses to be sent over the same GRPC stream. This new API is only recommended for use case when the inference latency of the full response is high and the inference intermediate results are sent to client. An example could be LLMs for generative applications, where generating "n" number of tokens can have high latency, in this case user can receive each generated token once ready until the full response completes. This new API automatically forces the batchSize to be one.

```
service InferenceAPIsService {
    // Check health status of the TorchServe server.
    rpc Ping(google.protobuf.Empty) returns (TorchServeHealthResponse) {}

    // Predictions entry point to get inference using default model version.
    rpc Predictions(PredictionsRequest) returns (PredictionResponse) {}

    // Streaming response for an inference request.
    rpc StreamPredictions(PredictionsRequest) returns (stream PredictionResponse) {}
}
```
Backend handler calls "send_intermediate_predict_response" to send one intermediate result to frontend, and return the last result as the existing style. For example
```python
from ts.handler_utils.utils import send_intermediate_predict_response
''' Note: TorchServe v1.0.0 will deprecate
"from ts.protocol.otf_message_handler import send_intermediate_predict_response".
Please replace it with "from ts.handler_utils.utils import send_intermediate_predict_response".
'''

def handle(data, context):
    if type(data) is list:
        for i in range (3):
            send_intermediate_predict_response(["intermediate_response"], context.request_ids, "Intermediate Prediction success", 200, context)
        return ["hello world "]
```
