Torchserve can be used for different use cases. In order to make it convenient for users, some of them have been documented here.
These use-cases assume you have pre-trained model(s) and torchserve is installed on your target system.

NOTES
- You have installed latest torchserve and torch-model-archive. If not then follow [installation](https://github.com/pytorch/serve#install-torchserve) instructions and complete installation
- If planning to use docker (henceforth referred as `Docker`), make sure following prerequisites are in place - 
    - Make sure you have latest docker engine install on your target node. If not then use [this](https://docs.docker.com/engine/install/) link to install it.
    - Follow instructions [install using docker](https://github.com/pytorch/serve/tree/master/docker#running-torchserve-in-a-production-docker-environment) to share `model-store` directory and start torchserve
- The following use-case steps uses `curl` to execute torchserve REST api calls. However, you can also use chrome plugin `postman` for this.

[Deploy pytorch eager mode model](#deploy_pytorch_eager_mode_model)

[Deploy pytorch scripted mode model](#deploy_pytorch_scripted_mode_model)

#### Deploy pytorch eager mode model
**Prerequisites**
- NA

**Steps to deploy your model(s)**
- Create MAR file for [torch eager model](https://github.com/pytorch/serve/tree/master/examples#creating-mar-file-for-eager-mode-model)
    ```
    torch-model-archiver --model-name <your_model_name> --version 1.0 --model-file <your_model_file>.py --serialized-file <your_model_name>.pth --handler <default handler or your_custom_handler_py_file> --extra-files ./index_to_name.json
    mkdir model_store
    mv <your_model_name>.mar model_store/
    ```
    - Docker - It is possible to build MAR file directly on docker, refer [this](https://github.com/pytorch/serve/tree/master/docker#create-torch-model-archiver-from-container) for details.
- Place MAR file in a new directory name it as `model-store` (this can be any name)
    - Docker -  Make sure that MAR file is being copied in volume/directory shared while starting torchserve docker image
- Start torchserve with following command - `torchserve --start --ncs --model-store <model_store or your_model_store_dir>`
    - Docker - This is not applicable.
- Register model i.e. MAR file created in step 1 above as `curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=https://<s3_path>/<your model_name>.mar"` 
- Check if model has been successfully registered as `curl http://localhost:8081/models/<your_model_name>`
- Scale up workers based on kind of load you are expecting. We have kept min-worker as 1 in registration request above. `curl -v -X PUT "http://localhost:8081/models/<your model_name>?min_worker=1&synchronous=true”`
- Do inference using following `curl` api call - `curl http://localhost:8080/predictions/<your_model_name> -T <your_input_file>`. You can also use `Postman` GUI tool for HTTP request and response. 

**Expected outcome**
- Able to deploy any scripted model
- Able to do inference using deployed model

**Examples**
- https://github.com/pytorch/serve/tree/master/examples/image_classifier

**Related blogs/demos**
- NA


#### Deploy pytorch scripted mode model
**Prerequisites**
- Assuming you have a torchscripted model if not then follow instructions in this [example](https://github.com/pytorch/serve/tree/master/examples/image_classifier#torchscript-example-using-densenet161-image-classifier) to save your eager mode model as scripted model.

**Steps to deploy your model(s)**
- Create MAR file for [torch scripted model](https://github.com/pytorch/serve/tree/master/examples#creating-mar-file-for-torchscript-mode-model)
    ```
    torch-model-archiver --model-name <your_model_name> --version 1.0  --serialized-file <your_model_name>.pt --extra-files ./index_to_name.json --handler <default handler or your_custom_handler_py_file>
    mkdir model-store
    mv <your_model_name>.mar model-store/
    ```
    - Docker - It is possible to build MAR file directly on docker, refer [this](https://github.com/pytorch/serve/tree/master/docker#create-torch-model-archiver-from-container) for details.
- Place MAR file in a new directory name it as `model-store` (this can be any name)
    - Docker -  Make sure that MAR file is being copied in volume/directory shared while starting torchserve docker image
- Start torchserve with following command - `torchserve --start --ncs --model-store <model_store or your_model_store_dir>`
    - Docker - This is not applicable.
- Register model i.e. MAR file created in step 1 above as `curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=https://<s3_path>/<your model_name>.mar"` 
- Check if model has been successfully registered as `curl http://localhost:8081/models/<your_model_name>`
- Scale up workers based on kind of load you are expecting. We have kept min-worker as 1 in registration request above. `curl -v -X PUT "http://localhost:8081/models/<your model_name>?min_worker=1&synchronous=true”`
- Do inference using following `curl` api call - `curl http://localhost:8080/predictions/<your_model_name> -T <your_input_file>`. You can also use `Postman` GUI tool for HTTP request and response. 

**Expected outcome**
- Able to deploy any scripted model
- Able to do inference using deployed model

**Examples**
- https://github.com/pytorch/serve/tree/master/examples/image_classifier

**Related blogs/demos**
- NA

TBA-
1.	Serve pre-trained general CNN models which are available on torchhub
2.	Securely deploy your models
4.	Deploy optimized model with ease [SCRIPTED]
5.	Deploy pytorch eager mode model
6.	Serve models on GPUs
7.	Serve local models with local packages
8.	Serve Huggingface models
9.	Serve Huggingface models with custom handler
10.	Serve audio synthesis models
11.	Serve your custom models with no third party dependency
12.	Move you development environment model to production/serving environment easily
    1.	Serve your custom models with third party dependency
    2.	Serve your model using docker
    3.	Serve your custom model using docker
13.	Deploy your models on Windows WSL
14.	Deploy your models on Windows native
    1.	Not supported
15.	Serve model using EKS
16.	Serve models using ECS
17.	Serve models via enterprise grade ELB backed by ASG
18.	Run A/B test 
19.	Backup and restore you models
20.	Serve models which return multiple output images?
21.	Serve models with authentication and authorization?
