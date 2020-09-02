# Pytorch Serve CircleCI build
TorchServe uses CircleCI for builds. This folder contains the config and scripts that are needed for CircleCI.

## Local CircleCI cli
To make it easy for developers to debug build issues locally, TorchServe supports CircleCI cli for running a job in a container on your machine.

#### Dependencies
1. CircleCI cli ([Quick Install](https://circleci.com/docs/2.0/local-cli/#quick-installation))
2. PyYAML (pip install PyYaml)
3. docker (installed and running)

#### Command
Developers can use the following command to execute CircleCI job:  
**_./run_circleci_tests.py <workflow_name> -j <job_name> -e <executor_name>**

- _workflow_name_  
This is a madatory parameter

- _-j, --job job_name_  
If specified, executes only the specified job (along with the required parent job).  
If not specified, all jobs in the workflow are executed sequentially.  

- _-e, --executor executor_name_  
If specified, job is executed only on the specified executor(docker image).  
If not specified, job is executed on all the supported executors.  

```bash
$ cd serve
$ ./run_circleci_tests.py sanity
$ ./run_circleci_tests.py sanity -j modelarchiver-tests
$ ./run_circleci_tests.py regression -e ubuntu18-conda38-cpu-docker
$ ./run_circleci_tests.py regression -j api-tests -e ubuntu18-venv36-cpu-docker
```

###### Checklist
> 1. Make sure you have configured "aws_access_key_id" and "aws_secret_access_key" using aws cli
>    - 'aws configure get aws_access_key_id' - should return the access key id
>    - 'aws configure get aws_secret_access_key' - should return the secret access key
> 2. Make sure docker is running before you start local execution.  
> 3. Docker containers to have **at least 4GB RAM, 2 CPU**.  
> 4. If you are on a network with low bandwidth, we advise you to explicitly pull the docker images -  
> docker pull 285923338299.dkr.ecr.us-east-1.amazonaws.com/torchserve-build:ubuntu18-pythn36-cpu    

`To avoid Pull Request build failures on github, developers should always make sure that their local builds pass.`

## config.yml
_config.yml_ contains TorchServe's build logic which is used by CircleCI.

## Workflows and Jobs
Currently, following _workflows_ are available -
1. sanity
2. regression
3. performance

Following _jobs_ are executed under these workflows -
1. **build** : Builds torchserve and torch-model-archiver
2. **modelarchiver-tests** : Executes pylint, unit and integration pytests on model archiver module
3. **frontend-tests** : Executes frontend gradle tests
4. **python-tests** : Executes pylint, unit and integration pytests on torchserve module
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

## scripts
Instead of using inline commands inside _config.yml_, job steps are configured as shell scripts.  
This is easier for maintenance and reduces chances of error in config.yml

## images
TorchServe uses customized docker images for its CircleCI build.    
We have published these docker images on AWS ECR
* 285923338299.dkr.ecr.us-east-1.amazonaws.com/torchserve-build:ubuntu18-pythn36-cpu
* 285923338299.dkr.ecr.us-east-1.amazonaws.com/torchserve-build:ubuntu18-conda38-cpu
* 285923338299.dkr.ecr.us-east-1.amazonaws.com/torchserve-build:ubuntu18-pyenv37-cpu
* 285923338299.dkr.ecr.us-east-1.amazonaws.com/torchserve-build:ubuntu18-venv36-cpu

Following file in the _images_ folder is used to create the docker images
* [Dockerfile](images/Dockerfile) (This is a parameterised Dockerfile)

#### To create a image run following cmd
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
