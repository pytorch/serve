# Execute unit testing and generate a code coverage report

## Prerequisites

You need some additional Python modules to run the unit tests and linting.

```bash
pip install mock pytest pylint pytest-mock pytest-cov
cd serve
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

The reports can be accessed at the following paths:

* TorchServe frontend: `serve/frontend/server/build/reports`
* TorchServe backend: `serve/htmlcov`
* torch-model-archiver: `serve/model-archiver/htmlcov`
