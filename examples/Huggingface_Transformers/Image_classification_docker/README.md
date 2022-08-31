

# Image Classification with Huggingface MobileViT using Torchserve Docker CPU container

## TLDR

* Decide a model name (variable `$modelName` below) for the task (variable `$task` below) supported by a 🤗 model that supports the task (its repo link as variable `$repoUrl` below), then run the below commands to download the models and create the `.mar` file:
```
#Setup
pip install torch-model-archiver
ROOT=$(pwd)
git clone https://github.com/pytorch/serve.git

#Creating Torchserve Docker image
cd $ROOT/serve/docker && ./build_image.sh -bt production -t torchserve-cpu-prod && cd $ROOT

#Edit the $WORKDIR/scripts/config.properties file to adjust Torchserve launch parameters

#Run the Docker container with shared volume between machine and container in a working directory
WORKDIR="$ROOT/serve/examples/Huggingface_Transformers/Image_classification_docker"
cd $WORKDIR && mkdir -p HF-models && mkdir -p model-store
docker run -d --rm -it --shm-size=50g -p 8080:8080 -p 8081:8081 --name torchserve-cpu-prod --mount type=bind,source=$WORKDIR/scripts/config.properties,target=/home/model-server/config.properties --mount type=bind,source=$WORKDIR/model-store,target=/home/model-server/model-store --mount type=bind,source=$WORKDIR/HF-models,target=/home/model-server/HF-models torchserve-cpu-prod torchserve --ncs --model-store=/home/model-server/model-store --ts-config config.properties


modelName="vitxxsmall"
task="image-classification"  

#should be either "pt" (for PyTorch) or "tf" (for Tensorflow) and the .pt or .tf file should be present in `HF-models/$modelName` folder
framework="pt"       
repoUrl="https://huggingface.co/apple/mobilevit-xx-small"
reqTxt="$ROOT/serve/requirements/developer.txt"

#Path to the to-be-created handler file specific to the task
handlerPath="$WORKDIR/scripts/mobilevit_handler.py"

#Create the specific handler .py file for the task at specified path
cd $ROOT/serve/examples/Huggingface_Transformers && chmod +x create_hf_handler.sh && ./create_hf_handler.sh -t $task -f $framework -o $handlerPath

#Download the model from HF hub and create model archive
cd $ROOT/serve/examples/Huggingface_Transformers && chmod +x prepare_mar_from_hf.sh && ./prepare_mar_from_hf.sh -p $handlerPath -n $modelName -r $reqTxt -d $WORKDIR -u $repoUrl

cd $WORKDIR
```

* After waiting for a few seconds for the server to boot up, register the downloaded model on the Torchserve server:
```
curl -X POST "localhost:8081/models?url=$modelName.mar&batch_size=8&max_batch_delay=10&initial_workers=1"
```

* Refer to the `Managing registered models` and `Inferencing and Benchmarking` sections below for auxillary details.

## Setup
Install the model archiver related dependencies:
```
pip install torch-model-archiver
```

Follow the [Docker installation guide](https://docs.docker.com/engine/install/) in case it is not installed.

## Running the Torchserve server
Navigate to the current folder after cloning this repo: 

```
ROOT=$(pwd)
git clone https://github.com/pytorch/serve.git
```

Build a Torchserve CPU Docker container (or GPU/IPEX containers following the [original guide](https://github.com/pytorch/serve/tree/master/docker#create-torchserve-docker-image)):  
```
cd $ROOT/serve/docker && ./build_image.sh -bt production -t torchserve-cpu-prod && cd $ROOT
```

Check whether the Torchserve image is present in the list of Docker images:
```
docker images
```

Run the Torchserve server container with Docker and archived model (refer to [this](https://github.com/pytorch/serve/tree/master/docker#create-torch-model-archiver-from-container) and [this](https://github.com/pytorch/serve/blob/fd4e3e8b72bed67c1e83141265157eed975fec95/docs/use_cases.md#secure-model-serving) for more details):

```
WORKDIR="$ROOT/serve/examples/Huggingface_Transformers/Image_classification_docker"
cd $WORKDIR && mkdir -p HF-models && mkdir -p model-store
docker run -d --rm -it --shm-size=50g -p 8080:8080 -p 8081:8081 --name torchserve-cpu-prod --mount type=bind,source=$WORKDIR/scripts/config.properties,target=/home/model-server/config.properties --mount type=bind,source=$WORKDIR/model-store,target=/home/model-server/model-store --mount type=bind,source=$WORKDIR/HF-models,target=/home/model-server/HF-models torchserve-cpu-prod torchserve --ncs --model-store=/home/model-server/model-store --ts-config /home/model-server/config.properties
```

Check whether the server was started properly (keep trying repeatedly for a few seconds while server boots up):
```
curl http://127.0.0.1:8080/ping
#OR
curl http://127.0.0.1:8081/models/
```

Install git-lfs to be able to download 🤗 models from the hub:
```
pip install git-lfs
```

## Registering a 🤗 model from the [hub](https://huggingface.co/models)

Download the 🤗 model repo with git-lfs ([example](https://huggingface.co/apple/mobilevit-xx-small)) along with all the model dependencies like checkpoints, vocabulary, config etc:
```
git clone https://huggingface.co/apple/mobilevit-xx-small $WORKDIR/HF-models/vitxxsmall/
git lfs install && cd HF-models/vitxxsmall/ && git lfs install && git lfs pull && cd $WORKDIR
```

⚠️ **IMPORTANT NOTE** ⚠️: The folder in which the 🤗 model repo is cloned & the name of the `.mar` file should be EXACTLY the same, this constraint was necessary to register new models with `curl POST` requests flexibly (i.e, during server intialization as well as afterwards).


Create a Torchserve model archive file by creating and using the model handler file (`scripts/mobilevit_handler.py` in our example) along with relevant dependencies in `developer.txt` (including 🤗 transformers).  

**NOTE:** Here, we are not giving a pretrained checkpoint as a `.pth` file, hence the `--serialized-file` option is redundant as we do not use the context in our handler. 
```
#Create the specific handler .py file for the task at specified path
cd $ROOT/serve/examples/Huggingface_Transformers && chmod +x create_hf_handler.sh && ./create_hf_handler.sh -t "image-classification" -f "pt" -o $WORKDIR/scripts/mobilevit_handler.py


touch dummy_file.pth
torch-model-archiver --model-name vitxxsmall --serialized-file dummy_file.pth --version 1.0 --handler $WORKDIR/scripts/mobilevit_handler.py --export-path $WORKDIR/model-store -r $ROOT/serve/requirements/developer.txt
rm -f dummy_file.pth
```

Since the `load_models` attribute of `config.properties` (that was passed to the `docker run` command while starting the Torchserve server) is set to "standalone", the Torchserve server is initialized without any models initially (even though `model-store` might contain `.mar` files). 

Registering the MobileViT XX Small model on the Torchserve server (more details [here](https://github.com/pytorch/serve/blob/master/docs/management_api.md#register-a-model)) with `max_batch_delay` in milliseconds, which is the time the Torchserve server waits to bundle concurrent inference requests into a batch with maximum size of `batch_size` (i.e, `preprocess` of the handler always receives list of requests with length <= `batch_size`):
```
curl -X POST "localhost:8081/models?url=vitxxsmall.mar&batch_size=8&max_batch_delay=10&initial_workers=1"
```

In case of bugs, recently created container can be accessed to check the logs for debugging, metrics etc(check the [logging documentation](https://github.com/pytorch/serve/blob/master/docs/logging.md) for details).  
In case `torch-model-archiver` is not available locally, the Torchserve container can be accessed to create the model archive in the `model-store` directory, which is mounted as shared memory between the machine and the container.  

```
serve_cont_id=$(docker ps -l -q) 
docker exec -it $serve_cont_id /bin/bash
cat logs/model_log.log
```

## Managing registered models

Run the following commands to check the registered models :
```
curl http://127.0.0.1:8081/models/

curl http://127.0.0.1:8081/models/vitxxsmall/
```

The models running on the Torchserve server can be managed with a [gRPC API](https://github.com/pytorch/serve/blob/master/docs/management_api.md#scale-workers).

For instance, the following command allows the model `vitxxsmall` to increase its number of workers, i.e, number of identical instances of the model (to process more inference requests per unit time) to 4.
```
curl -v -X PUT "http://localhost:8081/models/vitxxsmall?max_worker=4"
```



## Inferencing and Benchmarking

Example of a command to send inference requests to the registered models:
```
curl https://pbs.twimg.com/media/FM9MjZaUcAE7Wv1.png -o dog.png
time curl -X POST http://localhost:8080/predictions/vitxxsmall -T dog.png
```


However, in real scenarios, the registered model would receive many concurrent requests and hence, we need a better benchmarking approach. We provide several utilities that can be used to perform real-time benchmarking even with custom metrics ([reference](https://github.com/pytorch/serve/tree/master/benchmarks#torchserve-model-server-benchmarking)).  