## Create docker image for codebuild

To create docker image for code build execute following commands from serve home :

 - To create a docker image for CPU:
```
cd docker
./build_image.sh --codebuild
```

 - To create a docker image for GPU:
```
cd docker
./build_image.sh --codebuild --gpu
```

- To create a docker image for CPU with a custom tag:
```
cd docker
./build_image.sh --codebuild --tag torchserve:codebuild
```

 - To create a docker image for GPU with a custom tag:
```
cd docker
./build_image.sh --codebuild --gpu --tag torchserve:codebuild
```