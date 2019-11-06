# Model Server CI build

Model Server us AWS codebuild for i ts CI build. This folder contains scripts that needed for AWS codebuild.

## buildspec.yml
buildspec.yml contains MMS build logic which will be used by AWS codebuild.

## Docker images
MMS use customized docker image for its AWS codebuild. To make sure MMS is compatible with
 both Python2 and Python3, we use two build projects. We published two codebuild docker
 images on docker hub:
* awsdeeplearningteam/mms-build:python2.7
* awsdeeplearningteam/mms-build:python3.6

Following files in this folder is used to create the docker images
* Dockerfile.python2.7 - Dockerfile for awsdeeplearningteam/mms-build:python2.7
* Dockerfile.python3.6 - Dockerfile for awsdeeplearningteam/mms-build:python3.6
* dockerd-entrypoint.sh - AWS codebuild entrypoint script, required by AWS codebuild
* m2-settings.xml - Limit with repository can be used by maven/gradle in docker container, provided by AWS codebuild.

## AWS codebuild local
To make it easy for developer debug build issue locally, MMS support AWS codebuild local.
Developer can use following command to build MMS locally:
```bash
$ cd serve
$ ./run_ci_tests.sh
```

To avoid Pull Request build failure on github, developer should always make sure local build can pass.
