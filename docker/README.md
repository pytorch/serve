# Create TorchServe docker image

```bash
cd serve/docker
git clone https://github.com/pytorch/serve.git
```

For creating CPU based image :
```bash
docker build --file Dockerfile.cpu -t torchserve:1.0 .
```

For creating GPU based image :
```bash
docker build --file Dockerfile.gpu -t torchserve:1.0 .
```

## Start a container with a TorchServe image

The following examples will start the container with 8080/81 port exposed to outer-world/localhost.

### Start CPU container

For specific versions you can pass in the specific tag to use (ex: 0.1-cpu):
```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 pytorch/torchserve:0.1-cpu
```

For the latest version, you can use the `latest` tag:
docker run --rm -it -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest

#### Start GPU container

For specific versions you can pass in the specific tag to use (ex: 0.1-cpu):
```bash
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 pytorch/torchserve:0.1-cuda10.1-cudnn7-runtime
```

For the latest version, you can use the `gpu-latest` tag:
```bash
docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest-gpu
```

#### Accessing TorchServe APIs inside container

The TorchServe's inference and management APIs can be accessed on localhost over 8080 and 8081 ports respectively. Example :

```bash
curl http://localhost:8080/ping
```

#### Check running containers

```bash
docker ps
```

#### Stop TorchServe containers

```bash
docker container stop <containerid>
```

Container ID can be found using `docker ps` command.

#### Check port mapping associated with your container

```bash
docker port <containerid>
```

#### Important Note

If you are hosting web-server inside your container then explicitly specify the ip/host as 0.0.0.0 for your web-server
For details, refer : https://docs.docker.com/v17.09/engine/userguide/networking/default_network/binding/#related-information
