# Execute unit testing and generate a code coverage report

## Prerequisites

* Install all `TorchServe` prerequisites. For detailed steps refer [TorchServe installation documentation](../README.md#install-torchserve).

* Install additional Python modules to run the unit tests and linting using following commands

```bash
cd <path_to_serve>
pip install -r ci/requirements.txt
```


## Run sanity script

```bash
cd <path_to_serve>
./torchserve_sanity.sh
```

**The above command executes the following**

* TorchServe frontend build suite which consists of :

  * checkstyle
  * findbugs
  * PMD
  * UT

* TorchServe backend pytest suite

* torch-model-archive pytest suite

* Python linting on `ts` package

* Installs `torchserve` and `torch-model-archiver` from source

* Run basic sanity on `torchserve`

* Run basic sanity for snapshot feature

The reports can be accessed at the following paths:

* TorchServe frontend: `serve/frontend/server/build/reports`
* TorchServe backend: `serve/htmlcov`
* torch-model-archiver: `serve/model-archiver/htmlcov`
