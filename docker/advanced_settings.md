# Advanced Settings

## Contents of this Document
* [GPU Inference](advanced_settings.md#gpu-inference)
* [Reference Commands](advanced_settings.md#reference-commands)
* [Docker Details](advanced_settings.md#docker-details)
* [Description of Config File Settings](advanced_settings.md#description-of-config-file-settings)
* [Configuring SSL](advanced_settings.md#configuring-ssl)


## Other Relevant Documents
* [Quickstart](README.md#quickstart)
* [Configuring MMS with Docker](README.md#configuring-mms-with-docker)



## GPU Inference

**Step 1: Install nvidia-docker.**

`nvidia-docker` is NVIDIA's customized version of Docker that makes accessing your host's GPU resources from Docker a seamless experience. All of your regular Docker commands work the same way.

Follow the [instructions for installing nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart). Return here and follow the next step when the installation completes.

**Step 2: Download the GPU configuration template.**

A GPU configuration template is provided for your use.
Download the template a GPU config and place it in the `/tmp/models` folder you just created:
* [config.properties](config.properties)

**Step 3: Modify the configuration template.**

Edit the file you downloaded, `config.properties` to configure the model-server.

Save the file.

**Step 4: Run MMS with Docker using a shared volume.**

When you run the following command, the `-v` argument and path values of `/tmp/models/:/models` will map the `models` folder you created (assuming it was in ) with a folder inside the Docker container. MMS will then be able to use the local model file.

```bash
nvidia-docker run -itd --name mms -p 80:8080  -p 8081:8081 -v /tmp/models/:/models awsdeeplearningteam/mxnet-model-server:latest-gpu mxnet-model-server --start --mms-config /models/config.properties --models squeezenet=https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model
```

**Step 5: Test inference.**

This configuration file is using the default Squeezenet model, so you will request the `predictions/squeezenet` API endpoint.

```bash
curl -X POST http://127.0.0.1/predictions/squeezenet -F "data=@kitten.jpg"
```

Given that this is a different model, the same image yields a different inference result which will be something similar to the following:

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

## Reference Commands

Manually pull the MMS Docker CPU image:
```bash
docker pull awsdeeplearningteam/mxnet-model-server
```

Manually pull the MMS Docker GPU image:
```bash
docker pull awsdeeplearningteam/mxnet-model-server:latest-gpu
```

List your Docker images:
```bash
docker images
```

Verify the Docker container is running:
```bash
docker ps -a
```

Stop the Docker container from running:
```bash
docker rm -f mms
```

Delete the MMS Docker GPU image:
```bash
docker rmi awsdeeplearningteam/mxnet-model-server:latest-gpu
```

Delete the MMS Docker GPU image:
```bash
docker rmi awsdeeplearningteam/mxnet-model-server:latest
```

Output the recent logs to console.
```bash
docker logs mms
```

Interact with the container. This will open a shell prompt inside the container. Use `$ Ctrl-p-Ctrl-q` to detach again.
```bash
docker attach mms
```

Run the MMS Docker image without starting the Model Server:
```bash
docker run -itd --name mms -p 80:8080 -p 8081:8081 awsdeeplearningteam/mxnet-model-server /bin/bash
```

Start MMS in the Docker container (CPU config):
```bash
docker exec mms mxnet-model-server --start --mms-config /home/model-server/config.properties
```

Start MMS in the Docker container using nvidia-docker command as follows. :
```bash
nvidia-docker exec mxnet-model-server --start --mms-config /home/model-server/config.properties
```

**Note**: To use GPU configuration, modify the config.properties to reflect that the model-server should use GPUs.

```properties
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
...
number_of_gpu=8
...
```

Stop MMS.
```bash
docker exec mms mxnet-model-server --stop
```

Get MMS help.
```bash
docker exec mms mxnet-model-server --help
```

Refer [Docker CLI](https://docs.docker.com/engine/reference/commandline/run/) to understand each parameter.


## Docker Details

### Docker Hub

Docker images are available on [Docker Hub](https://hub.docker.com/r/awsdeeplearningteam):
* [CPU](https://hub.docker.com/r/awsdeeplearningteam/mxnet-model-server/tags)
* [GPU](https://hub.docker.com/r/awsdeeplearningteam/mxnet-model-server/tags)
### Building a MMS Docker Image from Scratch
The following are the steps to build a container image from scratch.

#### Prerequisites
In order to build the Docker image yourself you need the following:

* Install Docker
* Clone the MMS repo

#### Docker Installation

For macOS, you have the option of [Docker's Mac installer](https://docs.docker.com/docker-for-mac/install/) or you can simply use `brew`:

```bash
brew install docker
```

For Windows, you should use [their Windows installer](https://docs.docker.com/docker-for-windows/install/).

For Linux, check your favorite package manager if brew is available, otherwise use their installation instructions for [Ubuntu](https://docs.docker.com/engine/installation/linux/ubuntu/) or [CentOS](https://docs.docker.com/engine/installation/linux/centos/).

#### Verify Docker

When you've competed the installation, verify that Docker is running by running `docker images` in your terminal. If this works, you are ready to continue.

#### Clone the MMS Repo

If you haven't already, clone the MMS repo and go into the `docker` folder.

```bash
git clone https://github.com/awslabs/mxnet-model-server.git && cd mxnet-model-server/docker
```

### Building the Container Image

#### Configuring the Docker Build for Use on EC2

Now you can examine how to build a Docker image with MMS and establish a public accessible endpoint on EC2 instance. You should be able to adapt this information for any cloud provider. This Docker image can be used in other production environments as well. Skip this section if you're building for local use.

The first step is to create an [EC2 instance](https://aws.amazon.com/ec2/).

### Build Step for CPU container image

There are separate `Dockerfile` configuration files for CPU and GPU. They are named `Dockerfile.cpu` and `Dockerfile.gpu` respectively.

The container image consists of MXNet, Java, MMS and all related python libraries.

We can build the MXNet Model Server image based on the Dockerfile as follows:  

```bash
# Building the MMS CPU image
docker build -f Dockerfile.cpu -t mms_image .
```

Once this completes, run `docker images` from your terminal. You should see the Docker image listed with the tag, `mms_image:latest`.

### Build Step for GPU

If your host machine has at least one GPU installed, you can use a GPU Docker image to benefit from improved inference performance.

You need to install [nvidia-docker plugin](https://github.com/NVIDIA/nvidia-docker) before you can use a NVIDIA GPU with Docker.

Once you install `nvidia-docker`, run following commands (for info modifying the tag, see the CPU section above):

```bash
# Building the MMS GPU image
docker build -f Dockerfile.gpu -t mms_image_gpu .
```

#### Running the MMS GPU Docker

```bash
nvidia-docker run -itd -p 80:8080 8081:8081 --name mms -v /home/user/models/:/models mms_image_gpu /bin/bash
```

This command starts the Docker instance in a detached mode and mounts `/home/user/models` of the host system into `/models` directory inside the Docker instance.
Considering that you modified and copied `config.properties` file into the models directory, before you ran the 
above `nvidia-docker` command, you would have this configuration file ready to use in the Docker instance.

```bash
nvidia-docker exec mms mxnet-model-server --start --mms-config /models/config.properties
```

### Testing the MMS Docker

Now you can send a request to your server's [api-description endpoint](http://localhost/api-description) to see the 
list of MMS endpoints or [ping endpoint](http://localhost/ping) to check the health status of the MMS API. 
Remember to add the port if you used a custom one or the IP or DNS of your server if you configured it for that instead 
of localhost. Here are some handy test links for common configurations:

* [http://localhost/api-description](http://localhost/api-description)
* [http://localhost/ping](http://localhost/ping)

If `config.properties` file is used as is, the following commands can be run to verify that the MXNet Model Server is running.

```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1/squeezenet/predict -F "data=@kitten.jpg"
```

The predict endpoint will return a prediction response in JSON. It will look something like the following result:

```json
{
  "prediction": [
    [
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.9408261179924011
      },
      {
        "class": "n02127052 lynx, catamount",
        "probability": 0.055966004729270935
      },
      {
        "class": "n02123045 tabby, tabby cat",
        "probability": 0.0025502564385533333
      },
      {
        "class": "n02123159 tiger cat",
        "probability": 0.00034320182749070227
      },
      {
        "class": "n02123394 Persian cat",
        "probability": 0.00026897044153884053
      }
    ]
  ]
}
```

## Description of Config File Settings

**For config.properties:**

The system settings are stored in [config.properties](config.properties). You can modify these settings to use different models, or to apply other customized settings. 

Notes on a couple of the parameters:

* **model_store** - The directory on the local host where models must reside for serving.
* **management_address** - The address:port value on which the model server would serve control plane APIs such as "GET", "PUT", "DELETE" of "models"
* **inference_address** - The address:port value on which the model server would serve data plane APIs such as predictions, ping and api-description
* **load_models** - List of all the models in the `model_store` which should be loaded on startup
* **number_of_netty_threads** - Number of threads present to handle the incoming requests. 
* **max_workers** - 
* **job_queue_size** - Number of requests that can be queued. This queue is shared across models.
* **number_of_gpu** - Number of GPUs available for model server when serving inferences on GPU hosts
* **keystore** - SSL Key Store
* **keystore_pass** - SSL password
* **keystore_type** - Store of cryptographic keys and certificates
* **private_key_file** - Location of the private key file
* **certificate_file** - Location of the certificate file
* **max_response_size** - The maximum buffer size the frontend allocates for a worker response, in bytes.
* **max_request_size** - The maximum allowable request size that the MMS accepts.

in the range of 0 .. (num-gpu-1) in a round-robin fashion. **By default MMS uses all the available GPUs but this parameter can be configured if user want to use only few of them**.

```properties
# vmargs=-Xmx1g -XX:MaxDirectMemorySize=512m -Dlog4j.configuration=file:///opt/ml/conf/log4j.properties
# model_store=/opt/ml/model
# load_models=ALL
# inference_address=http://0.0.0.0:8080
# management_address=http://0.0.0.0:8081
# number_of_netty_threads=0
# max_workers=0
# job_queue_size=1000
# number_of_gpu=1
# keystore=src/test/resources/keystore.p12
# keystore_pass=changeit
# keystore_type=PKCS12
# private_key_file=src/test/resources/key.pem
# certificate_file=src/test/resources/certs.pem
# max_response_size=6553500
# max_request_size=6553500
```
