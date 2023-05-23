# FAQ'S
Contents of this document.
* [General](#general)
* [Deployment and config](#deployment-and-config)
* [API](#api)
* [Handler](#handler)
* [Model-archiver](#model-archiver)

## General
Relevant documents.
- [Torchserve readme](https://github.com/pytorch/serve#torchserve)

### Does Torchserve API's follow some REST API standard?
Torchserve API's are compliant with the [OpenAPI specification 3.0](https://swagger.io/specification/).

### How to use Torchserve in production?
Depending on your use case, you will be able to deploy torchserve in production using following mechanisms.
> Standalone deployment. Refer [TorchServe docker documentation](https://github.com/pytorch/serve/tree/master/docker#readme) or [TorchServe documentation](README.md)
> Cloud based deployment. Refer [TorchServe kubernetes documentation](https://github.com/pytorch/serve/tree/master/kubernetes#readme) or [TorchServe cloudformation documentation](https://github.com/pytorch/serve/tree/master/examples/cloudformation/README.md#cloudformation)


### What's difference between Torchserve and a python web app using web frameworks like Flask, Django?
Torchserve's main purpose is to serve models via http REST APIs, Torchserve is not a Flask app and it uses netty engine for serving http requests.

Relevant issues: [[581](https://github.com/pytorch/serve/issues/581),[569](https://github.com/pytorch/serve/issues/569)]

### Are there any sample Models available?
Various models are provided in Torchserve out of the box. Checkout out Torchserve [Model Zoo](model_zoo.md) for list of all available models. You can also check out the [examples](https://github.com/pytorch/serve/tree/master/examples) folder.

### Does Torchserve support other models based on programming languages other than python?
No, As of now only python based models are supported.

### What benefits does Torchserve have over AWS Multi-Model-Server?
Torchserve is derived from Multi-Model-Server. However, Torchserve is specifically tuned for Pytorch models. It also has new features like Snapshot and model versioning.

### How to decode international language in inference response on client side?
By default, Torchserve uses utf-8 to encode if the inference response is string. So client can use utf-8 to decode. 

If a model converts international language string to bytes, client needs to use the codec mechanism specified by the model such as in https://github.com/pytorch/serve/blob/master/examples/nmt_transformer/model_handler_generalized.py#L55

## Deployment and config
Relevant documents.
- [Torchserve configuration](configuration.md)
- [Model zoo](model_zoo.md)
- [Snapshot](snapshot.md)
- [Docker](https://github.com/pytorch/serve/blob/master/docker/README.md#docker-readme)

### Can I run Torchserve APIs on ports other than the default 8080 & 8081?
Yes, Torchserve API ports are configurable using a properties file or environment variable.
Refer to [configuration](configuration.md) for more details.


### How can I resolve model specific python dependency?
You can provide a `requirements.txt` while creating a mar file using "--requirements-file/ -r" flag. Also, you can add dependency files using "--extra-files" flag.
Refer to [configuration](configuration.md) for more details.

### Can I deploy Torchserve in Kubernetes?
Yes, you can deploy Torchserve in Kubernetes using Helm charts.
Refer [Kubernetes deployment ](https://github.com/pytorch/serve/blob/master/kubernetes/README.md#torchserve-kubernetes) for more details.

### Can I deploy Torchserve with AWS ELB and AWS ASG?
Yes, you can deploy Torchserve on a multi-node ASG AWS EC2 cluster. There is a cloud formation template available [here](https://github.com/pytorch/serve/blob/master/examples/cloudformation/ec2-asg.yaml) for this type of deployment. Refer [ Multi-node EC2 deployment behind Elastic LoadBalancer (ELB)](https://github.com/pytorch/serve/tree/master/examples/cloudformation/README.md#multi-node-ec2-deployment-behind-elastic-loadbalancer-elb) more details.

### How can I backup and restore Torchserve state?
TorchServe preserves server runtime configuration across sessions such that a TorchServe instance experiencing either a planned or unplanned service stop can restore its state upon restart. These saved runtime configuration files can be used for backup and restore.
Refer to [TorchServe model snapshot](snapshot.md) for more details.

### How can I build a Torchserve image from source?
Torchserve has a utility [script](https://github.com/pytorch/serve/blob/master/docker/build_image.sh) for creating docker images, the docker image can be hardware-based CPU or GPU compatible. A Torchserve docker image could be CUDA version specific as well.

All these docker images can be created using `build_image.sh` with appropriate options.

Run `./build_image.sh --help` for all available options.

Refer to [Create Torchserve docker image from source](https://github.com/pytorch/serve/blob/master/docker/README.md#create-torchserve-docker-image) for more details.

### How to build a Torchserve image for a specific branch or commit id?
To create a Docker image for a specific branch, use the following command:

`./build_image.sh -b <branch_name>/<commit_id>`

To create a Docker image for a specific branch and specific tag, use the following command:

`./build_image.sh -b <branch_name> -t <tagname:latest>`


### What is the difference between image created using Dockerfile and image created using Dockerfile.dev?
The image created using Dockerfile.dev has Torchserve installed from source where as image created using Dockerfile has Torchserve installed from PyPi distribution.

### What is the order of config.property path?
TorchServe looks for the config.property file according to the order listed in the [doc](https://github.com/pytorch/serve/blob/master/docs/configuration.md#configproperties-file). There is no override mechanism.

### What are model_store, load_models, models?
- model_store: A mandatory argument during TorchServe start. It can be either defined in config.property or overridden by TorchServe command line option "[--model-store](configuration.md)".

- load_models: An optional argument during TorchServe start. It can be either defined in config.property or overridden by TorchServe command line option "[--models](configuration.md)".

- [models](configuration.md): Defines a list of models' configuration in config.property. A model's configuration can be overridden by [management API](management_api.md). It does not decide which models will be loaded during TorchServe start. There is no relationship b.w "models" and "load_models" (ie. TorchServe command line option [--models](configuration.md)).

### 

## API
Relevant documents
- [Torchserve Rest API](rest_api.md)

### What can I use other than *curl* to make requests to Torchserve?
You can use any tool like Postman, Insomnia or even use a python script to do so. Find sample python script [here](https://github.com/pytorch/serve/blob/master/docs/default_handlers.md#torchserve-default-inference-handlers).

### How can I add a custom API to an existing framework?
You can add a custom API using **plugins SDK** available in Torchserve.
Refer to [serving sdk](https://github.com/pytorch/serve/tree/master/serving-sdk) and [plugins](https://github.com/pytorch/serve/tree/master/plugins) for more details.

### How can pass multiple images in Inference request call to my model?
You can provide multiple data in a single inference request to your custom handler as a key-value pair in the `data` object.
Refer to [this issue](https://github.com/pytorch/serve/issues/529#issuecomment-658012913) for more details.

## Handler
Relevant documents
- [Default handlers](default_handlers.md)
- [Custom Handlers](custom_service.md)

### How do I return an image output for a model?
You would have to write a custom handler and modify the postprocessing to return the image
Refer to [custom service documentation](custom_service.md) for more details.

### How to enhance the default handlers?
Write a custom handler that extends the default handler and just override the methods to be tuned.
Refer to [custom service documentation](custom_service.md) for more details.

### Do I always have to write a custom handler or are there default ones that I can use?
Yes, you can deploy your model with no-code/ zero code by using builtin default handlers.
Refer to [default handlers](default_handlers.md) for more details.

### Is it possible to deploy Hugging Face models?
Yes, you can deploy Hugging Face models using a custom handler.
Refer to [HuggingFace_Transformers](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/README.md#huggingface-transformers) for example. 

## Model-archiver
 Relevant documents
 - [Model-archiver ](https://github.com/pytorch/serve/blob/master/model-archiver/README.md#torch-model-archiver-for-torchserve)
 - [Docker Readme](https://github.com/pytorch/serve/blob/master/docker/README.md#docker-readme)

### What is a mar file?
A mar file is a zip file consisting of all model artifacts with the ".mar" extension. The cmd-line utility `torch-model-archiver` is used to create a mar file.

### How can create mar file using Torchserve docker container?
Yes, you create your mar file using a Torchserve container. Follow the steps given [here](https://github.com/pytorch/serve/blob/master/docker/README.md#create-torch-model-archiver-from-container).

### Can I add multiple serialized files in single mar file?
Currently `torch-model-archiver` allows supplying only one serialized file with `--serialized-file` parameter while creating the mar. However, you can supply any number and any type of file with `--extra-files` flag. All the files supplied in the mar file are available in `model_dir` location which can be accessed through the context object supplied to the handler's entry point.

Sample code snippet:

```python
properties = context.system_properties
model_dir = properties.get("model_dir")
```
Refer [Torch model archiver cli](https://github.com/pytorch/serve/blob/master/model-archiver/README.md#torch-model-archiver-command-line-interface) for more details.
Relevant issues: [[#633](https://github.com/pytorch/serve/issues/633)]

### Can I download and register model using s3 presigned v4 url?
You can use both s3 v2 and v4 signature URLs.
Note: For v4 type replace `&` characters in model url with its URL encoding character in the curl command i.e.`%26`.

Relevant issues: [[#669](https://github.com/pytorch/serve/issues/669)]

### Can I host a model on s3
A mar file can be used either locally or be publicly available via http. An S3 URI starting with s3:// will not work but that very same file can be made public and available in the s3 console or aws cli to instead get a public object URL starting with https://

### How to set a model's batch size on SageMaker?  Key parameters for TorchServe performance tuning.
[TorchServe performance tuning example](https://github.com/lxning/torchserve_perf/blob/master/torchserve_perf.ipynb)

## Why is my model initialization so slow?
There's a few reasons why model initialization can be slow
1. `torch.load()` overhead - not something we can improve, this will be more dramatic for larger models
2. CUDA context launch overhead - not something we can control
3. install_py_dep_per_model=true is intended for local development or sagemaker deployments, in other production environment you should pre install your dependencies
4. The model archiver has an overhead to compress and decompress models, the compression is on by default because historically torchserve came out of sagemaker needs which involve loading and unloading tons of models stored in cloud buckets. But for users with smaller deployments choosing `torch-model-archiver --no-archive` is a good bet
