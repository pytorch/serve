#### Create TorchServe docker image

```bash
cd serve/docker
git clone https://github.com/pytorch/serve.git
cp <path_to_resnet-18_MAR_file> .
docker build -t torchserve:1.0 .
```

Note:
1) To create resnet-18.mar file refer : [image classification with resnet18 example documentation](../examples/image_classifier/resnet_18)
2) The resnet-18.mar file must have model name as resnet-18.

#### Start container with TorchServe image

```bash
docker run --rm -it -p 8080:8080 -p 8081:8081 torchserve:1.0
```

The above command will start the container with 8080/81 port exposed to outer-world/localhost

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
