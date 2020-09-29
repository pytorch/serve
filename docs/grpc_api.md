# TorchServe gRPC API

TorchServe also supports [gRPC APIs](../frontend/server/src/main/resources/proto) for both inference and management calls.

TorchServe provides following gRPCs apis

* [Inference API](../frontend/server/src/main/resources/proto/management.proto)
 - Ping : Gets the health status of the running server
 - Predictions : Gets predictions from the served model

* [Management API](../frontend/server/src/main/resources/proto/management.proto)
 - **RegisterModel** : Serve a model/model-version on TorchServe
 - **UnregisterModel** : Free up system resources by unregistering specific version of a model from TorchServe
 - **ScaleWorker** : Dynamically adjust the number of workers for any version of a model to better serve different inference request loads.
 - **ListModels** : Query default versions of current registered models
 - **DescribeModel** : Get detail runtime status of default version of a model
 - **SetDefault** : Set any registered version of a model as default version

By default, TorchServe listens on port 9090 for the gRPC Inference API and 9091 for the gRPC Management API.
To configure gRPC APIs on different ports refer [configuration documentation](configuration.md)