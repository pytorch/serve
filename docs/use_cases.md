# Torchserve Use Cases

Torchserve can be used for different use cases. In order to make it convenient for users, some of them have been documented here.
These use-cases assume you have pre-trained model(s) and `torchserve`, `torch-model-archiver` is installed on your target system. 
This should help you in moving your development environment model to production/serving environment.

NOTES
- If you have not installed latest torchserve and torch-model-archiver then follow [installation](../README.md#install-torchserve-and-torch-model-archiver) instructions and complete installation
- If planning to use docker make sure following prerequisites are in place - 
    - Make sure you have latest docker engine install on your target node. If not then use [this](https://docs.docker.com/engine/install/) link to install it.
    - Follow instructions [install using docker](../docker/README.md#running-torchserve-in-a-production-docker-environment) to share `model-store` directory and start torchserve
- The following use-case steps uses `curl` to execute torchserve REST api calls. However, you can also use chrome plugin `postman` for this.
- Please refer [default_handler](default_handlers.md) to understand default handlers.
- Please refer [custom handlers](custom_service.md) to understand custom handlers.

## Use Cases

[Serve pytorch eager mode model](#deploy-pytorch-eager-mode-model)

[Serve pytorch scripted mode model](#deploy-pytorch-scripted-mode-model)

[Serve ready made models on torchserve model zoo](#serve-readymade-models-on-torchserve-model-zoo)

[Secure model serving](#secure-model-serving)

[Serve models on GPUs](#serve-models-on-gpus)

[Serve custom models with no third party dependency](#serve-custom-models-with-no-third-party-dependency)

[Serve custom models with third party dependency](#serve-custom-models-with-third-party-dependency)

[Serve models for A/B testing](#serve-models-for-ab-testing)

### Deploy pytorch eager mode model

**Steps to deploy your model(s)**
- Create MAR file for [torch eager model](../examples/README.md#creating-mar-file-for-eager-mode-model)
    ```
    torch-model-archiver --model-name <your_model_name> --version 1.0 --model-file <your_model_file>.py --serialized-file <your_model_name>.pth --handler <default_handler> --extra-files ./index_to_name.json
    mkdir model_store
    mv <your_model_name>.mar model_store/
    ```
    - Docker - It is possible to build MAR file directly on docker, refer [this](../docker/README.md#create-torch-model-archiver-from-container) for details.
- Place MAR file in a new directory name it as `model-store` (this can be any name)
    - Docker -  Make sure that MAR file is being copied in volume/directory shared while starting torchserve docker image
- Start torchserve with following command - `torchserve --start --ncs --model-store <model_store or your_model_store_dir>`
    - Docker - This is not applicable.
- Register model i.e. MAR file created in step 1 above as `curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=<your model_name>.mar"` 
- Check if model has been successfully registered as `curl http://localhost:8081/models/<your_model_name>`
- Scale up workers based on kind of load you are expecting. We have kept min-worker as 1 in registration request above. `curl -v -X PUT "http://localhost:8081/models/<your model_name>?min_worker=1&synchronous=true”`
- Do inference using following `curl` api call - `curl http://localhost:8080/predictions/<your_model_name> -T <your_input_file>`. You can also use `Postman` GUI tool for HTTP request and response. 

**Expected outcome**
- Able to deploy any scripted model
- Able to do inference using deployed model

### Deploy pytorch scripted mode model
**Prerequisites**
- Assuming you have a torchscripted model if not then follow instructions in this [example](https://github.com/pytorch/serve/tree/master/examples/image_classifier/densenet_161) to save your eager mode model as scripted model.

**Steps to deploy your model(s)**
- Create MAR file for [torch scripted model](../examples#creating-mar-file-for-torchscript-mode-model)
    ```
    torch-model-archiver --model-name <your_model_name> --version 1.0  --serialized-file <your_model_name>.pt --extra-files ./index_to_name.json --handler <default_handler>
    mkdir model-store
    mv <your_model_name>.mar model-store/
    ```
    - Docker - It is possible to build MAR file directly on docker, refer [this](../docker#create-torch-model-archiver-from-container) for details.
- Place MAR file in a new directory name it as `model-store` (this can be any name)
    - Docker -  Make sure that MAR file is being copied in volume/directory shared while starting torchserve docker image
- Start torchserve with following command - `torchserve --start --ncs --model-store <model_store or your_model_store_dir>`
    - Docker - This is not applicable.
- Register model i.e. MAR file created in step 1 above as `curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=<your model_name>.mar"` 
- Check if model has been successfully registered as `curl http://localhost:8081/models/<your_model_name>`
- Scale up workers based on kind of load you are expecting. We have kept min-worker as 1 in registration request above. `curl -v -X PUT "http://localhost:8081/models/<your model_name>?min_worker=1&synchronous=true”`
- Do inference using following `curl` api call - `curl http://localhost:8080/predictions/<your_model_name> -T <your_input_file>`. You can also use `Postman` GUI tool for HTTP request and response. 

**Expected outcome**
- Able to deploy any scripted model
- Able to do inference using deployed model

**Examples**
- ../examples/image_classifier

### Serve readymade models on torchserve model zoo
This use case demostrates deployment of [torch hub](https://pytorch.org/hub/) based `vision` models (classifier, object detector, segmenter) available on [torchserve model zoo](https://github.com/pytorch/serve/blob/master/docs/model_zoo.md). 
Use these steps to deploy **publically hosted** models as well.

**Steps to deploy your model(s)**
- Start torchserve with following command - `torchserve --start --ncs --model-store <model_store or your_model_store_dir>`
    - Docker - This is not applicable.
- Register model i.e. MAR file created in step 1 above as `curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=https://<public_url>/<your model_name>.mar"` 
- Check if model has been successfully registered as `curl http://localhost:8081/models/<your_model_name>`
- Scale up workers based on kind of load you are expecting. We have kept min-worker as 1 in registration request above. `curl -v -X PUT "http://localhost:8081/models/<your model_name>?min_worker=1&synchronous=true”`
- Do inference using following `curl` api call - `curl http://localhost:8080/predictions/<your_model_name> -T <your_input_file>`. You can also use `Postman` GUI tool for HTTP request and response. 

**Expected outcome**
- Able to deploy any model available in model zoo
- Able to do inference using deployed model

**Examples**
- [image_classifier](../examples/image_classifier)
- [image_segmenter](../examples/image_segmenter)
- [object_detector](../examples/object_detector)

### Secure model serving
This use case demonstrates torchserve deployment for secure model serving. 
The example taken here uses eager mode model however you can also deploy scripted models.

**Steps to deploy your model(s)**
- Create MAR file for [torch eager model](../examples#creating-mar-file-for-eager-mode-model)
    ```
    torch-model-archiver --model-name <your_model_name> --version 1.0 --model-file <your_model_file>.py --serialized-file <your_model_name>.pth --handler <default_handler> --extra-files ./index_to_name.json
    mkdir model_store
    mv <your_model_name>.mar model_store/
    ```
    - Docker - It is possible to build MAR file directly on docker, refer [this](../docker#create-torch-model-archiver-from-container) for details.
- Place MAR file in a new directory name it as `model-store` (this can be any name)
    - Docker -  Make sure that MAR file is being copied in volume/directory shared while starting torchserve docker image
- Create `config.properties` file with parameters option 1 or 2 given in [enable SSL](https://github.com/pytorch/serve/blob/master/docs/configuration.md#examples)
- Start torchserve using properties file created above as - `torchserve --start --ncs --model-store <model_store or your_model_store_dir> --ts-config <your_path>/config.properties`
    - Docker - `docker run --rm -p8443:8433 -p8444:8444 -p8445:8445 -v <local_dir>/model-store:/home/model-server/model-store <your_docker_image> torchserve --model-store=/tmp/models --ts-config <your_path>/config.properties`
- Register model i.e. MAR file created in step 1 above as `curl -k -v -X POST "https://localhost:8081/models?initial_workers=1&synchronous=true&url=https://<s3_path>/<your model_name>.mar"` 
- Check if model has been successfully registered as `curl -k https://localhost:8081/models/<your_model_name>`
- Scale up workers based on kind of load you are expecting. We have kept min-worker as 1 in registration request above. `curl -v -X PUT "http://localhost:8081/models/<your model_name>?min_worker=1&synchronous=true”`
- Do inference using following `curl` api call - `curl -k https://localhost:8080/predictions/<your_model_name> -T <your_input_file>`. You can also use `Postman` GUI tool for HTTP request and response. 
  
  NOTICE the use of https and -k option in curl command. In place of -k, you can use other options such as -key etc if you have required key.
- 
**Expected outcome**
- Able to deploy torchserve and access APIs over HTTPs

**Examples/Reference**
- https://github.com/pytorch/serve/blob/master/docs/configuration.md#enable-ssl

### Serve models on GPUs
This use case demonstrates torchserve deployment on GPU. 
The example taken here uses scripted mode model however you can also deploy eager models.

**Prerequisites**
- Assuming you have a torchscripted model if not then follow instructions in this [example](../examples/image_classifier/densenet_161/README.md#torchscript-example-using-densenet161-image-classifier) to save your eager mode model as scripted model.

**Steps to deploy your model(s)**
- Create MAR file for [torch scripted model](../examples#creating-mar-file-for-torchscript-mode-model)
    ```
    torch-model-archiver --model-name <your_model_name> --version 1.0  --serialized-file <your_model_name>.pt --extra-files ./index_to_name.json --handler <default_handler>
    mkdir model-store
    mv <your_model_name>.mar model-store/
    ```
    - Docker - It is possible to build MAR file directly on docker, refer [this](../docker#create-torch-model-archiver-from-container) for details.
- Move MAR file in a new directory name it as `model-store`
    - Docker -  Make sure that MAR file is being copied in volume/directory shared while starting torchserve docker image
- torchserve start command in following instruction will automatically detect GPUs and use for loading/serving models. If you want to [limit the GPU usage](https://github.com/pytorch/serve/blob/master/docs/configuration.md#limit-gpu-usage)
then use `nvidia-smi` to determine the number of GPU and corresponding ids. Once you have gpu details, you can add `number_of_gpu` param in config.proerties and use second command as given next instruction.
e.g. number_of_gpu=2
- Start torchserve with all GPUs- `torchserve --start --ncs --model-store <model_store or your_model_store_dir>`. With restricted GPUs - `torchserve --start --ncs --model-store <model_store or your_model_store_dir> --ts-config <your_path>/config.properties`
    - Docker -  For all GPU `docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 torchserve:gpu-latest` For GPUs 1 and 2 `docker run --rm -it --gpus '"device=1,2"' -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest-gpu`
    - Docker - For details refer [start gpu container](../docker#start-gpu-container)
- Register model i.e. MAR file created in step 1 above as `curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=<your model_name>.mar"` 
- Check if model has been successfully registered as `curl http://localhost:8081/models/<your_model_name>` The response includes flag indicating model has been loaded on GPU.
- Scale up workers based on kind of load you are expecting. We have kept min-worker as 1 in registration request above. `curl -v -X PUT "http://localhost:8081/models/<your model_name>?min_worker=1&synchronous=true”`
- Do inference using following `curl` api call - `curl http://localhost:8080/predictions/<your_model_name> -T <your_input_file>`. You can also use `Postman` GUI tool for HTTP request and response. 

**Expected outcome**
- Able to deploy any model to GPU
- Able to do inference using deployed model

### Serve custom models with no third party dependency
This use case demonstrates torchserve deployment for custom models with no python dependency apart from pytorch and related libs.
The example taken here uses scripted mode model however you can also deploy eager models.

**Prerequisites**
- Assuming you have a torchscripted model if not then follow instructions in this [example](../examples/image_classifier/densenet_161/README.md#torchscript-example-using-densenet161-image-classifier) to save your eager mode model as scripted model.

**Steps to deploy your model(s)**
- Create [<your_custom_handler_py_file>](https://github.com/pytorch/serve/blob/master/docs/custom_service.md)
- Create MAR file for [torch scripted model](../examples#creating-mar-file-for-torchscript-mode-model)
    ```
    torch-model-archiver --model-name <your_model_name> --version 1.0  --serialized-file <your_model_name>.pt --extra-files ./index_to_name.json --handler <**path/to/your_custom_handler_py_file**>
    mkdir model-store
    mv <your_model_name>.mar model-store/
    ```
    - Docker - It is possible to build MAR file directly on docker, refer [this](../docker#create-torch-model-archiver-from-container) for details.
- Place MAR file in a new directory name it as `model-store` (this can be any name)
    - Docker -  Make sure that MAR file is being copied in volume/directory shared while starting torchserve docker image
- Start torchserve with following command - `torchserve --start --ncs --model-store <model_store or your_model_store_dir>`
    - Docker - This is not applicable.
- Register model i.e. MAR file created in step 1 above as `curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=<your model_name>.mar"` 
- Check if model has been successfully registered as `curl http://localhost:8081/models/<your_model_name>`
- Scale up workers based on kind of load you are expecting. We have kept min-worker as 1 in registration request above. `curl -v -X PUT "http://localhost:8081/models/<your model_name>?min_worker=1&synchronous=true”`
- Do inference using following `curl` api call - `curl http://localhost:8080/predictions/<your_model_name> -T <your_input_file>`. You can also use `Postman` GUI tool for HTTP request and response. 

**Expected outcome**
- Able to deploy any model with custom handler

**Examples**
- [MNIST example](../examples/image_classifier/mnist)


### Serve custom models with third party dependency
This use case demonstrates torchserve deployment for custom models with python dependency apart from pytorch and related libs.
The example taken here uses scripted mode model however you can also deploy eager models.

**Prerequisites**
- Assuming you have a torchscripted model if not then follow instructions in this [example](https://github.com/pytorch/serve/tree/master/examples/image_classifier/densenet_161) to save your eager mode model as scripted model.

**Steps to deploy your model(s)**
- Create [<your_custom_handler_py_file>](https://github.com/pytorch/serve/blob/master/docs/custom_service.md) which uses third party python package such as [fairseq](https://github.com/pytorch/fairseq) for pretrained NMT models
- Create a requirements.txt file with an entry for `fairseq` python package name in it
- Create MAR file for [torch scripted model](../examples#creating-mar-file-for-torchscript-mode-model) with requirements.txt
    ```
    torch-model-archiver --model-name <your_model_name> --version 1.0  --serialized-file <your_model_name>.pt --extra-files ./index_to_name.json --handler <**path/to/your_custom_handler_py_file**> --requirements-file <your_requirements_txt>
    mkdir model-store
    mv <your_model_name>.mar model-store/
    ```
    - Docker - It is possible to build MAR file directly on docker, refer [this](../docker#create-torch-model-archiver-from-container) for details.
- Place MAR file in a new directory name it as `model-store` (this can be any name)
    - Docker -  Make sure that MAR file is being copied in volume/directory shared while starting torchserve docker image
- Add following parameter to config.properties file - `install_py_dep_per_model=true` . For details refer [Allow model specific custom python packages](https://github.com/pytorch/serve/blob/master/docs/configuration.md#allow-model-specific-custom-python-packages) .
- Start torchserve with following command with config.properties file - `torchserve --start --ncs --model-store <model_store or your_model_store_dir> --ts-config <your_path>/config.properties`
    - Docker - `docker run --rm -p8080:8080 -p8081:8081 -v <local_dir>/model-store:/home/model-server/model-store <your_docker_image> torchserve --model-store=/tmp/models --ts-config <your_path>/config.properties
- Register model i.e. MAR file created in step 1 above as `curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=<your model_name>.mar"` 
- Check if model has been successfully registered as `curl http://localhost:8081/models/<your_model_name>`
- Scale up workers based on kind of load you are expecting. We have kept min-worker as 1 in registration request above. `curl -v -X PUT "http://localhost:8081/models/<your model_name>?min_worker=1&synchronous=true”`
- Do inference using following `curl` api call - `curl http://localhost:8080/predictions/<your_model_name> -T <your_input_file>`. You can also use `Postman` GUI tool for HTTP request and response. 

**Expected outcome**
- Able to deploy any model with custom handler having third party python dependency

**Examples and References**
- [Installing model specific python dependencies](https://github.com/pytorch/serve/blob/master/docs/custom_service.md#installing-model-specific-python-dependencies)


### Serve models for AB testing

This use case demonstrates serving two or more versions of same model using version API. It is an extension of any of the above use cases.

**Prerequisites**
- You have followed any of the above procedure and have a working torchserve setup along with torch-model-archiver installed.

**Steps to deploy your model(s)**
- Create a model [i.e. mar file] with version 1.0 or as per requirement. Follow the steps given above to create model file
  e.g. torch-model-archiver --model-name <your-model-name-X> **--version 1.0** --model-file model.py --serialized-file <your-model-name-X>.pth --extra-files index_to_name.json --handler <your-model-name-X-handler>.py
- Create another model [i.e. mar file] with version 2.0 or as per requirement
  e.g. torch-model-archiver --model-name <your-model-name-X> **--version 2.0** --model-file model.py --serialized-file <your-model-name-X>.pth --extra-files index_to_name.json --handler <your-model-name-X-handler>.py
- Register both these models with a initial worker. If you want, you can increase workers by using update api.
  `curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=<your-model-name-X>.mar"`
- Now you will be able to invoke these models as
  - Model version 1.0
  `curl http://localhost:8081/models/<your-model-name-X>/1.0`
  OR 
  `curl http://localhost:8080/predictions/<your-model-name-X>/1.0 -F "data=@kitten.jpg"`
  - Model version 2.0
  `curl http://localhost:8081/models/<your-model-name-X>/2.0`
    OR 
  `curl http://localhost:8080/predictions/<your-model-name-X>/2.0 -F "data=@kitten.jpg"`

**Expected outcome**
- Able to deploy multiple versions of same model

**Examples and References**
- [Model management APIs](https://github.com/pytorch/serve/blob/master/docs/management_api.md)
