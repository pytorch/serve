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

### Does Torchserve API's  follow some REST API standard?
Torchserve API's are compliant with the [OpenAPI specification 3.0](https://swagger.io/specification/).

### What is not Torchserve?
Torchserve is not a complete web application to serve end to end business use cases. Hence a lot of security aspects should be made available via 3rd party wrapper components in front of Torchserve.

### What's difference between Torchserve and a python web app using web frameworks like Flask, Django?
The Flask app and Torchserve are completely different except the fact that both support handling HTTP requests however using different engines [netty or python apis].

###  Are there any sample Models available?
Various models are provided in Torchserve out of the box. Checkout out Torchserve [Model Zoo](https://github.com/pytorch/serve/blob/master/docs/model_zoo.md) for list of all available models in model zoo. Also check out examples folder all available [examples](https://github.com/pytorch/serve/tree/master/examples).

### Does Torchserve has Windows Support?
Currently, Torchserve supports Windows only via WSL(Windows Subsystem for Linux).
The native Windows support will be added in upcoming releases.

### Can I do streaming service with Torchserve like streaming speech recognition?
Torchserve currently supports only inference through HTTP 1.0 - Request / Response style.
We don't have anything particular in our roadmap - [pytorch/pytorch#27610](https://github.com/pytorch/pytorch/issues/27610)

### Is it possible to deploy model other than Pytorch framework?
Yes, it is possible to deploy(with Python inference logic) however Torchserve has been certified with any other framework hence there can be unknowns.

###  Does Torchserve support other models based on programming languages other than python?
No, As of now only python based models are supported.


### What benefits Torchserve has over AWS Multi-Model-server?
Torchserve is derived from Multi-Model-server, But Torchserve is specifically tuned for Pytorch models. It also has new features like Snapshot and model versioning.

## Deployment and config
Relevant documents.
- [Torchserve configuration](https://github.com/pytorch/serve/blob/master/docs/configuration.md)
- [Model zoo](https://github.com/pytorch/serve/blob/master/docs/model_zoo.md#model-zoo)
- [Snapshot](https://github.com/pytorch/serve/blob/master/docs/snapshot.md)

### Can I run Torchserve APIs on different ports other than 8080 & 8081?
Yes, Torchserve API ports are configurable using a properties file or environment variable.Refer  [configuration.md](https://github.com/pytorch/serve/blob/master/docs/configuration.md) for more details.


### How can I resolve model specific python dependency?
You can provide a requirements.txt while creating a mar file using "--requirements-file/ -r" flag. Also, you can add dependency files using "--extra-files" flag. Refer  [configuration.md](https://github.com/pytorch/serve/blob/master/docs/configuration.md) for more details.

### Can I deploy Torchserve in Kubernetes?
Yes, you can deploy Torchserve in Kubernetes using Helm charts. Refer
[Kubernetes deployment ](https://github.com/pytorch/serve/blob/master/kubernetes/README.md) for more details.

### Can deploy Torchserve with ELB and ASG?
Yes, you can deploy Torchserve on a multinode ASG EC2 cluster. There is a cloud formation template available [here](https://github.com/pytorch/serve/blob/master/cloudformation/ec2-asg.yaml) for this type of deployment. Refer
[ Multi-node EC2 deployment behind Elastic LoadBalancer (ELB)](https://github.com/pytorch/serve/tree/master/cloudformation#multi-node-ec2-deployment-behind-elastic-loadbalancer-elb) more details.

### How can I backup and restore Torchserve state?
TorchServe preserves server runtime configuration across sessions such that a TorchServe instance experiencing either a planned or unplanned service stop can restore its state upon restart. These saved runtime configuration files can be used for backup and restore.
Refer [TorchServe model snapshot](https://github.com/pytorch/serve/blob/master/docs/snapshot.md#torchserve-model-snapshot) for more details.

## API
Relevant documents
- [Torchserve Rest API](https://github.com/pytorch/serve/blob/master/docs/model_zoo.md#model-zoo)
###  What can I use other than *curl* to make requests to Torchserve?
You can use any tool like Postman, Insomnia or even use a python script to do so. Find sample python script [here](https://github.com/pytorch/serve/blob/master/docs/default_handlers.md#torchserve-default-inference-handlers).

### How can add a custom API to an existing framework?
You can a custom API using **plugins SDK** available in Torchserve.
The [Health check API ](https://github.com/pytorch/serve/blob/master/docs/inference_api.md#health-check-api) is an example of a custom API integrated using plugins SDK.
Refer to Plugins Documentation for more details.

## Handler
Relevant documents
- [Default handlers](https://github.com/pytorch/serve/blob/master/docs/model_zoo.md#model-zoo)
- [Custom Handlers](https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handlers)

###  How do I return an image output for a model?
You would have to write a custom handler with the post processing to return image.
Refer [custom  service documentation](https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handlers) for more details.

### How to enhance the default handlers?
Write a custom handler that extends the default handler and just override the methods to be tuned.
Refer  [custom  service documentation](https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handlers) for more details.

### How to serve a model with no-code/zero code?
Yes, you can deploy your model with no-code/ zero code by using builtin default handlers. Refer [default handlers](https://github.com/pytorch/serve/blob/master/docs/default_handlers.md#torchserve-default-inference-handlers) for more details.

### Is it possible to deploy Hugging Face models?
Yes, you can deploy Hugging Face models using a custom handler.
Refer [Huggingface_Transformers](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/README.md) for example. 

## Model-archiver
 Relevant documents
 - [Model-archiver ](https://github.com/pytorch/serve/tree/master/model-archiver#torch-model-archiver-for-torchserve)
 - [Docker Readme](https://github.com/pytorch/serve/blob/master/docker/README.md)

### What is a mar file?
A mar file is a zip file consisting of all model artifacts with the ".mar" extension. The cmd-line utility *torch-model-archiver*  is used to create a mar file.

### How can create mar file using Torchserve docker container?
Yes, you create your mar file using a Torchserve container. Follow the steps given [here](https://github.com/pytorch/serve/blob/master/docker/README.md#create-torch-model-archiver-from-container).

### Can I add multiple serialized files in single mar file?
Currently `TorchModelArchiver` allows supplying only one serialized file with `--serialized-file` parameter while creating the mar. However, you can supply any number and any type of file with `--extra-files` flag. All the files supplied in the mar file are available in `model_dir` location which can be accessed through the context object supplied to the handler's entry point.

Sample code snippet:
```
properties = context.system_properties
model_dir = properties.get("model_dir")
```
Refer [Torch model archiver cli](https://github.com/pytorch/serve/blob/master/model-archiver/README.md#torch-model-archiver-command-line-interface) for more details.