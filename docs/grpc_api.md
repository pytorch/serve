# TorchServe gRPC API

__Note__: Current TorchServe gRPC does not support workflow.

TorchServe also supports [gRPC APIs](https://github.com/pytorch/serve/tree/master/frontend/server/src/main/resources/proto) for both inference and management calls.

TorchServe provides following gRPCs apis

* [Inference API](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/inference.proto)
  - **Ping** : Gets the health status of the running server
  - **Predictions** : Gets predictions from the served model
  - **StreamPredictions** : Gets server side streaming predictions from the saved model

* [Management API](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/management.proto)
  - **RegisterModel** : Serve a model/model-version on TorchServe
  - **UnregisterModel** : Free up system resources by unregistering specific version of a model from TorchServe
  - **ScaleWorker** : Dynamically adjust the number of workers for any version of a model to better serve different inference request loads.
  - **ListModels** : Query default versions of current registered models
  - **DescribeModel** : Get detail runtime status of default version of a model
  - **SetDefault** : Set any registered version of a model as default version

By default, TorchServe listens on port 7070 for the gRPC Inference API and 7071 for the gRPC Management API.
To configure gRPC APIs on different ports refer [configuration documentation](configuration.md)

## Python client example for gRPC APIs

Run following commands to Register, run inference and unregister, densenet161 model from [TorchServe model zoo](model_zoo.md) using [gRPC python client](https://github.com/pytorch/serve/blob/master/ts_scripts/torchserve_grpc_client.py).

 - [Install TorchServe](../README.md)

 - Clone serve repo to run this example

```bash
git clone https://github.com/pytorch/serve
cd serve
```

 - Install gRPC python dependencies

```bash
pip install -U grpcio protobuf grpcio-tools
```

 - Start torchServe

```bash
mkdir models
torchserve --start --model-store models/
```

 - Generate python gRPC client stub using the proto files

```bash
python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto
```

 - Register densenet161 model

```bash
python ts_scripts/torchserve_grpc_client.py register densenet161
```

 - Run inference using

```bash
python ts_scripts/torchserve_grpc_client.py infer densenet161 examples/image_classifier/kitten.jpg
```

 - Unregister densenet161 model

```bash
python ts_scripts/torchserve_grpc_client.py unregister densenet161
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
```
from ts.protocol.otf_message_handler import send_intermediate_predict_response

def handle(data, context):
    if type(data) is list:
        for i in range (3):
            send_intermediate_predict_response(["intermediate_response"], context.request_ids, "Intermediate Prediction success", 200, context)
        return ["hello world "]
```