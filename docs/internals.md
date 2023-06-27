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

## pytorch/serve directories and what they do
* `.github`: CI for docs and nightly builds
* `benchmarks`: tools to benchmark torchserve on reference models
* `binaries`: utilities to create binaries for pypi, conda and docker
* `docker`: user and dev dockerfiles to use torchserve
* `docs`: documentation for pytorch.org/serve
* `examples`: reference examples
* `experimental`: projects with no support or backwards compatibility guarantees
* `frontend`: Core Java engine for TorchServe
* `kubernetes`: how to deploy TorchServe in a K8 cluster
* `model-archiver`: model package CLI
* `plugins`: extend core TorchServe
* `requirements`: requirements.txt
* `serving_sdk`: SDK to support TorchServe in sagemaker
* `test`: tests
* `ts_scripts`: useful utility files that don't fit in any other folder
* `workflow-archiver`: workflow package CLI

## Important files for the core TorchServe engine

Frontend means the Java part of the code (potentially C++)

And backend is the Python code (most Pytorch specific stuff)

### Backend (Python)

https://github.com/pytorch/serve/blob/master/ts/arg_parser.py

* Arg parser controls config/not workflow and can also setup a model service worker with a custom socket


https://github.com/pytorch/serve/blob/master/ts/context.py

* Context object of incoming request - keeps model relevant worker information 

https://github.com/pytorch/serve/blob/master/ts/model_server.py

* model server open up pid and start torchserve by using the arg parser
* If stopping it they use psutil.Process(pid).terminate()
* loads config.properties

https://github.com/pytorch/serve/blob/master/ts/model_loader.py

* Model loader
* Uses manifest file to find handler and envelope and starts the service 
* Loads either default handler or custom handler
* Request envelopes which make it easier to interact with other systems like Seldon, KFserving, Google cloud AI platform

### Frontend (Java)


`../gradlew startServer`


https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/wlm/WorkerStateListener.java

* Takes care of closing workers 

https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/wlm/WorkerState.java

* Just an enum of worker states


https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/wlm/WorkLoadManager.java

* Get number of running workers
* Number of workers which is just a concurrent hashmap, backendgroup, ports etc are all here
* Add worker threads by submitting them to a threadpool Executor Service (create a pool of threads and assign tasks or worker threads to it)


https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/wlm/BatchAggregator.java

* Batch aggregator
* Puts requests and responses in a list


https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/wlm/Model.java

* Keeps track of workers, batch size, timeout, version and mar name
* Encoding the model state in a JSON and then pulling properties from it


https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/job/Job.java

* Keeps track of jobs which are either inference or management requests

https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/ts/metrics/system_metrics.py

* Many metrics are just added using psutil package in Python


https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/wlm/ModelManager.java

* Model registration calling https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/util/ApiUtils.java
* Install model dependencies
* create model archive

https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/util/ConfigManager.java

* All configs managed here

https://github.com/pytorch/serve/blob/8903ca1fb059eab3c1e8eccdee1376d4ff52fb67/frontend/server/src/main/java/org/pytorch/serve/wlm/WorkerThread.java

* Get GPU usage
* Worker thread has model, aggregator, listener, eventloop, port etc and then a run function which connects it to a request
