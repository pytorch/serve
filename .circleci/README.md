# Pytorch Serve CircleCI build

TorchServe uses CircleCI for builds. This folder contains the config and scripts that are needed for CircleCI. To make it easy for developers to debug build issues locally, there is also support for running a CircleCI job locally on your machine.

#### Dependencies
1. CircleCI CLI ([installation](https://circleci.com/docs/2.0/local-cli/#quick-installation))
2. PyYAML  - ```pip install PyYaml```

#### Command
Use the following command to execute CircleCI job:  

```
./run_circleci_tests.py <workflow_name> -j <job_name> -e <executor_name>

workflow_name is a madatory positional arg.
-j, --job job_name : Executes a specific job.
-e, --executor executor_name
```

If specified, job is executed only on the specified executor(docker image), else its executed on all the supported executors.  

```bash
$ cd serve
$ ./run_circleci_tests.py sanity
$ ./run_circleci_tests.py sanity -j modelarchiver-tests
$ ./run_circleci_tests.py regression -e ubuntu18-conda38-cpu-docker
$ ./run_circleci_tests.py regression -j api-tests -e ubuntu18-venv36-cpu-docker
```

###### Checklist
> 1. Configure AWS CLI Credentials.
> 2. Make sure docker is running before you start local execution. / Docker containers to have **at least 4GB RAM, 2 CPU**.  


## Workflows and Jobs
Currently, following _workflows_ are available -
1. sanity
2. regression
3. performance

Following _jobs_ are executed under these workflows -
1. **build** : Builds torchserve and torch-model-archiver
2. **modelarchiver-tests** : Executes pylint, unit and integration pytests on model archiver module
3. **frontend-tests** : Executes frontend gradle tests
4. **torchserve-tests** : Executes pylint, unit and integration pytests on torchserve module
5. **sanity-tests** : Executes the sanity scripts
6. **api-tests** : Executes newman test suite for API testing
7. **regression-tests** : Executes the regression script
8. **benchmark** : Executes latency benchmark using resnet-18 model
9. **performance-regression** : Executes performance regression suite

Following _executors_ are available for job execution -
1. ubuntu18-pythn36-cpu-docker
2. ubuntu18-conda38-cpu-docker
3. ubuntu18-pyenv37-cpu-docker
4. ubuntu18-venv36-cpu-docker

> Please check the _workflows_, _jobs_ and _executors_ section in _config.yml_ for an up to date list


## Build Images
TorchServe uses customized docker images for its CircleCI build. Following file in the `images` folder is used to create the docker images
* [Dockerfile](images/Dockerfile) (This is a parameterised Dockerfile)

#### To create an image run following cmd
```
./build_cci_image.sh
```
[build_cci_image.sh](images/build_cci_image.sh) is a utility script to create docker images

#### To create a image with different env options available use `--env_type` or `-e` flag
Available environment types are
* `pythn36`: plain python with python 3.6.9
* `conda38`: python 3.8 with `conda` env
* `pyenv37`: python 3.7 with `pyenv` env
* `venv36`: python 3.6.9 with `venv` env
```
./build_cci_image.sh --env_type <env-type>
```
#### To create an image with custom tag use --tag or -t
```
./build_cci_image.sh --tag <tagname>
```
