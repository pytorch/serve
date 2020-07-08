# Pytorch Serve CircleCI build
TorchServe uses CircleCI for builds. This folder contains the config and scripts that are needed for CircleCI.

## config.yml
_config.yml_ contains TorchServe's build logic which will be used by CircleCI.

## Workflows and Jobs
Currently, following _workflows_ are available -
1. smoke
2. nightly
3. weekly

Following _jobs_ are executed under each workflow -
1. **build** : Builds _frontend/model-server.jar_ and executes tests from gradle
2. **modelarchiver** : Builds and tests modelarchiver module
3. **python-tests** : Executes pytests from _serve/tests/unit_tests/_
4. **benchmark** : Executes latency benchmark using resnet-18 model
5. (NEW!) **api-tests** : Executes newman test suite for API testing

Following _executors_ are available for job execution -
1. py36

> Please check the _workflows_, _jobs_ and _executors_ section in _config.yml_ for an up to date list

## scripts
Instead of using inline commands inside _config.yml_, job steps are configured as shell scripts.  
This is easier for maintenance and reduces chances of error in config.yml

## images
TorchServe uses customized docker image for its CircleCI build.    
We have published the docker image on docker hub for code build
* prashantsail/pytorch-serve-build

Following file in the _images_ folder is used to create the docker image
* Dockerfile - Dockerfile for prashantsail/pytorch-serve-build

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
If specified, executes only the specified job name (along with the required parents).  
If not specified, all jobs in the workflow are executed sequentially.  

- _-e, --executor executor_name_  
If specified, job is executed only on the specified executor(docker image).  
If not specified, job is executed on all the available executors.  

```bash
$ cd serve
$ ./run_circleci_tests.py smoke
$ ./run_circleci_tests.py smoke -j modelarchiver
$ ./run_circleci_tests.py smoke -e py36
$ ./run_circleci_tests.py smoke -j modelarchiver -e py36
```

###### Checklist
> 1. Make sure docker is running before you start local execution.  
> 2. Docker containers to have **at least 4GB RAM, 2 CPU**.  
> 3. If you are on a network with low bandwidth, we advise you to explicitly pull the docker images -  
> docker pull prashantsail/pytorch-serve-build    

`To avoid Pull Request build failures on github, developers should always make sure that their local builds pass.`
