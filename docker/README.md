[//]: # "All the references in this file should be actual links because this file would be used by docker hub. DO NOT use relative links or section tagging."

# Using Containers with MXNet Model Server

MXNet Model Server (MMS) can be used with any container service. In this guide, you will learn how to run MMS with Docker.

## Contents of this Document
* [Quickstart](https://github.com/awslabs/mxnet-model-server/blob/master/docker/README.md#quickstart)
* [Available pre-built containers](https://github.com/awslabs/mxnet-model-server/blob/master/docker/README.md#available-pre-built-containers)
* [Configuring MMS with Docker](https://github.com/awslabs/mxnet-model-server/blob/master/docker/README.md#configuring-mms-with-docker)


## Other Relevant Documents
* [Advanced Settings](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced_settings.md)
    * [GPU Inference](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced_settings.md#gpu-inference)
    * [Reference Commands](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced_settings.md#reference-commands)
    * [Docker Details](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced_settings.md#docker-details)
    * [Description of Config File Settings](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced_settings.md#description-of-config-file-settings)
    * [Configuring SSL](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced_settings.md#configuring-ssl)
* [Launch MMS as a managed inference service on AWS Fargate](https://github.com/awslabs/mxnet-model-server/blob/master/docs/mms_on_fargate.md)
    * [Introduction to published containers](https://github.com/awslabs/mxnet-model-server/blob/master/docs/mms_on_fargate.md#familiarize-yourself-with-our-containers)
    * [Creating a AWS Fargate task to server SqueezeNet V1.1](https://github.com/awslabs/mxnet-model-server/blob/master/docs/mms_on_fargate.md#create-a-aws-faragte-task-to-serve-squeezenet-model)
    * [Creating an Load Balancer](https://github.com/awslabs/mxnet-model-server/blob/master/docs/mms_on_fargate.md#create-a-load-balancer)
    * [Creating an AWS ECS Service](https://github.com/awslabs/mxnet-model-server/blob/master/docs/mms_on_fargate.md#creating-an-ecs-service-to-launch-our-aws-fargate-task)
    * [Testing your service](https://github.com/awslabs/mxnet-model-server/blob/master/docs/mms_on_fargate.md#test-your-service)
    * [Build custom MMS containers images to serve your Deep learning models](https://github.com/awslabs/mxnet-model-server/blob/master/docs/mms_on_fargate.md#customize-the-containers-to-server-your-custom-deep-learning-models)

## Quickstart
Running MXNet Model Server with Docker in two steps:

**Step 1: Run the Docker image.**

This will download the MMS Docker image and run its default configuration, serving a SqueezeNet model.

```bash
docker run -itd --name mms -p 80:8080 -p 8081:8081 awsdeeplearningteam/mxnet-model-server mxnet-model-server --start --models squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
```

With the `-p` flag, we're setting it up so you can run the Predict API on your host computer's port `80`. This maps to the Docker image's port `8080`.
It will run the Management API on your host computer's port `8081`. This maps to the Docker image's port `8081`.

**Step 2: Test inference.**

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1/predictions/squeezenet -T kitten.jpg
```

After fetching this image of a kitten and posting it to the `predict` endpoint, you should see a response similar to the following:

```
{
  "prediction": [
    [
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.9408261179924011
...
```

### Cleaning Up

Now that you have tested it out, you may stop the Docker container. The following command will stop the server and delete the container. It will retain the Docker image for trying out other models and configurations later.

```bash
docker rm -f mms
```

## Available pre-built containers
We have following container tags available on [Docker Hub](https://hub.docker.com/r/awsdeeplearningteam/mxnet-model-server/).
1. *latest*: This is the latest officially released MMS CPU container. This is based on the latest [Dockerfile.cpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/Dockerfile.cpu).
2. *latest-gpu*: This is the latest officially released MMS GPU container. This is based on the latest [Dockerfile.gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/Dockerfile.gpu).
3. *(MMS Release Tag)-mxnet-cpu*: Each released version since MMS 1.0.0 has an individual release tagged CPU MXNet container. These containers are based on [Dockerfile.cpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/Dockerfile.cpu), in that MMS release.
4. *(MMS Release Tag)-mxnet-cpu*: Each released version since MMS 1.0.0 has an individual release tagged GPU MXNet container. These containers are based on [Dockerfile.gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/Dockerfile.gpu), in that MMS release.
5. *nightly-mxnet-cpu*: This is the official CPU container which is built based on the nightly release of MMS pip package. This is built from [Dockerfile.nightly-cpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/Dockerfile.nightly-cpu).
6. *nightly-mxnet-gpu*: This is the official GPU container which is built based on the nightly release of MMS pip package. This is built from [Dockerfile.nightly-gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/Dockerfile.nightly-gpu).
7. *base-cpu-py2.7*: This is the official Base Python 2.7 CPU container which contains only MMS and python. This is built from [Dockerfile.nightly-gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced-dockerfiles/Dockerfile.base.ubuntu_16_04.py2_7). Please note, this container doesn't have any DL/ML engine installed by definition, it is meant to be used for cases where you would like to bring your own engine/framework into container. **WARNING: Python 2.x will be deprecated from Jan 1 2020.**
8. *base-cpu-py3.6*: This is the official Base Python 3.6 CPU container which contains only MMS and python. This is built from [Dockerfile.nightly-gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced-dockerfiles/Dockerfile.base.ubuntu_16_04.py3_6). Please note, this container doesn't have any DL/ML engine installed by definition, it is meant to be used for cases where you would like to bring your own engine/framework into container.
9. *base-gpu-py2.7*: This is the official Base Python 2.7 GPU container which contains only MMS and python. This is built from [Dockerfile.nightly-gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced-dockerfiles/Dockerfile.base.nvidia_cu92_ubuntu_16_04.py2_7). Please note, this container doesn't have any DL/ML engine installed by definition, it is meant to be used for cases where you would like to bring your own engine/framework into container. **WARNING: Python 2.x will be deprecated from Jan 1 2020.**
10. *base-gpu-py3.6*: This is the official Base Python 3.6 GPU container which contains only MMS and python. This is built from [Dockerfile.nightly-gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced-dockerfiles/Dockerfile.base.nvidia_cu92_ubuntu_16_04.py3_6). Please note, this container doesn't have any DL/ML engine installed by definition, it is meant to be used for cases where you would like to bring your own engine/framework into container.
11. *nightly-base-cpu-py2.7*: This is the official Nightly Base Python 2.7 CPU container which contains only MMS and python. This is built from [Dockerfile.nightly-gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced-dockerfiles/Dockerfile.base.ubuntu_16_04.py2_7). Please note, this container doesn't have any DL/ML engine installed by definition, it is meant to be used for cases where you would like to bring your own engine/framework into container. **WARNING: Python 2.x will be deprecated from Jan 1 2020.**
12. *nightly-base-cpu-py3.6*: This is the official Nightly Base Python 3.6 CPU container which contains only MMS and python. This is built from [Dockerfile.nightly-gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced-dockerfiles/Dockerfile.base.ubuntu_16_04.py3_6). Please note, this container doesn't have any DL/ML engine installed by definition, it is meant to be used for cases where you would like to bring your own engine/framework into container.
13. *nightly-base-gpu-py2.7*: This is the official Nightly Base Python 2.7 GPU container which contains only MMS and python. This is built from [Dockerfile.nightly-gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced-dockerfiles/Dockerfile.base.nvidia_cu92_ubuntu_16_04.py2_7). Please note, this container doesn't have any DL/ML engine installed by definition, it is meant to be used for cases where you would like to bring your own engine/framework into container. **WARNING: Python 2.x will be deprecated from Jan 1 2020.**
14. *nightly-base-gpu-py3.6*: This is the official Nightly Base Python 3.6 GPU container which contains only MMS and python. This is built from [Dockerfile.nightly-gpu](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced-dockerfiles/Dockerfile.base.nvidia_cu92_ubuntu_16_04.py3_6). Please note, this container doesn't have any DL/ML engine installed by definition, it is meant to be used for cases where you would like to bring your own engine/framework into container.

To pull the a particular container, run the following command

#### Pulling the latest CPU container:
Docker pull by default pulls the latest tag. This tag is associated with latest released MMS CPU container. This tag isn't available until after an official release. 
```bash
docker pull awsdeeplearningteam/mxnet-model-server 
``` 

#### Pulling the latest GPU container:
To pull a official latest released MMS GPU container run the following command. This tag isn't available until after an official release. 
```bash
docker pull awsdeeplearningteam/mxnet-model-server:latest-gpu
``` 

#### Pulling the `nightly-mxnet-cpu` tag:
To pull a latest nigthtly MMS CPU container run the following command. This track the pre-release version of MMS.
We do not recommend running this container in production setup.
```bash
docker pull awsdeeplearningteam/mxnet-model-server:nightly-mxnet-cpu
``` 

#### Pulling the `nightly-mxnet-gpu` tag:
To pull a latest nigthtly MMS CPU container run the following command. This track the pre-release version of MMS.
We do not recommend running this container in production setup.
```bash
docker pull awsdeeplearningteam/mxnet-model-server:nightly-mxnet-gpu
``` 

## Configuring MMS with Docker

In the Quickstart section, you launched a Docker image with MMS serving the SqueezeNet model.
Now you will learn how to configure MMS with Docker to run other models.
You will also learn how to collect MMS logs, and optimize MMS with Docker images.

### Using MMS and Docker with a Shared Volume

You may sometimes want to load different models with a different configuration.
Setting up a shared volume with the Docker image is the recommended way to handle this.

**Step 1: Create a folder to share with the Docker container.**

Create a directory for `models`. This will also provide a place for log files to be written.

```bash
mkdir /tmp/models
```

**Step 2: Download the configuration template.**

Download the template `config.properties` and place it in the `models` folder you just created:
* [config.properties](https://github.com/awslabs/mxnet-model-server/blob/master/docker/config.properties)

**Step 3: Modify the configuration template.**

Edit the file you downloaded, `config.properties`.

```properties
vmargs=-Xmx128m -XX:-UseLargePages -XX:+UseG1GC -XX:MaxMetaspaceSize=32M -XX:MaxDirectMemorySize=10m -XX:+ExitOnOutOfMemoryError
model_store=/opt/ml/model
load_models=ALL
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
# management_address=unix:/tmp/management.sock
# number_of_netty_threads=0
# netty_client_threads=0
# default_response_timeout=120
# unregister_model_timeout=120
# default_workers_per_model=0
# job_queue_size=100
# async_logging=false
# number_of_gpu=1
# cors_allowed_origin
# cors_allowed_methods
# cors_allowed_headers
# keystore=src/test/resources/keystore.p12
# keystore_pass=changeit
# keystore_type=PKCS12
# private_key_file=src/test/resources/key.pem
# certificate_file=src/test/resources/certs.pem
# blacklist_env_vars=
```

Modify the configuration file to suite your configuration needs before running the model server.

Save the file.

**Step 4: Run MMS with Docker using a shared volume.**

When you run the following command, the `-v` argument and path values of `/tmp/models/:/models` will map the Docker image's `models` folder to your local `/tmp/models` folder.
MMS will then be able to use the local model file.

```bash
docker run -itd --name mms -p 80:8080 -p 8081:8081 -v /tmp/models/:/models awsdeeplearningteam/mxnet-model-server mxnet-model-server --start --models squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-18.mar
```

**NOTE**: If you modify the inference_address or the management_address in the configuration file,
you must modify the ports exposed by Docker as well.

**Step 5: Test inference.**

You will upload the same kitten image as before, but this time you will request the `predictions/resnet` API endpoint.

```bash
curl -X POST http://127.0.0.1/predictions/resnet -T @kitten.jpg
```

Given that this is a different model, the same image yields a different inference result which is something similar to the following:

```
{
  "prediction": [
    [
      {
        "class": "n02123159 tiger cat",
        "probability": 0.3630334138870239
      },
...
```

## Conclusion

You have tried the default Predictions API settings using a SqueezeNet model. 
You then configured your Predictions API endpoints to also serve a ResNet-18 model.
Now you are ready to try some other more **advanced settings** such as:

* GPU inference
* MMS settings

Next Step: [Advanced Settings](https://github.com/awslabs/mxnet-model-server/blob/master/docker/advanced_settings.md)
