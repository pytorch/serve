# TorchServe gRPC API

__Note__: Current TorchServe gRPC does not support workflow.

TorchServe also supports [gRPC APIs](https://github.com/pytorch/serve/tree/master/frontend/server/src/main/resources/proto) for both inference and management calls.

TorchServe provides following gRPCs apis

* [Inference API](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/proto/inference.proto)
  - **Ping** : Gets the health status of the running server
  - **Predictions** : Gets predictions from the served model

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
