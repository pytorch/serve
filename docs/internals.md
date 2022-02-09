## TorchServe internals

TorchServe was designed a multi model inferencing framework. A production grade inferencing framework needs both APIs to request inferences but also APIs to manage models all the while keeping track of logs. TorchServe manages several worker processes that are dynamically assigned to different models with the behavior of those workers determined by a handler file and a model store where weights are loaded from. 

## TorchServe Architecture
![Architecture Diagram](https://user-images.githubusercontent.com/880376/83180095-c44cc600-a0d7-11ea-97c1-23abb4cdbe4d.jpg)

### Terminology:
* **Frontend**: The request/response handling component of TorchServe. This portion of the serving component handles both request/response coming from clients and manages the lifecycles of the models.
* **Model Workers**: These workers are responsible for running the actual inference on the models.
* **Model**: Models could be a `script_module` (JIT saved models) or `eager_mode_models`. These models can provide custom pre- and post-processing of data along with any other model artifacts such as state_dicts. Models can be loaded from cloud storage or from local hosts.
* **Plugins**: These are custom endpoints or authz/authn or batching algorithms that can be dropped into TorchServe at startup time.
* **Model Store**: This is a directory in which all the loadable models exist.