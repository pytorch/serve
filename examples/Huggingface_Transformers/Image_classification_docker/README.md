# HF-torchserve-pipeline
 
This repository contains an example to deploy models with third-party dependencies (like 🤗 Transformers, sparseml etc) on Torchserve servers as ready-for-usage Docker containers on cloud services like AWS.  

For the context of this repository, we would deploy the models on an AWS [`t2.micro`](https://aws.amazon.com/ec2/instance-types/) instance which can be used for free (for 750 hours) on a new AWS account. We work with a 🤗 MobileViT Transformer [model](https://huggingface.co/apple/mobilevit-xx-small) for the task of image classification by using its [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines) feature, the handler code in `scripts` can also be used as a simplistic template to deploy an 🤗 `pipeline`.

This work can also be [extended to deploy *any* 🤗 `pipeline` for *any* supported task with Torchserve](https://github.com/tripathiarpan20/HF-torchserve-pipeline/tree/main/HF-only#instructions-to-use-any--model-from-the-hub-for-any-task-supported-by-the--pipeline).

This work *may* also be extended to deploy the Torchserve Docker containers with HF models at scale with [AWS Cloudformation](https://github.com/pytorch/serve/tree/master/examples/cloudformation) & [AWS EKS](https://github.com/pytorch/serve/tree/master/kubernetes/EKS) as explained in the official Torchserve repo & [AWS Sagemaker](https://github.com/tescal2/TorchServeOnAWS/tree/master/3_torchserve_byoc_with_amazon_sagemaker), incorporating utilities like AWS ELB & Cloudwatch.

We would also benchmark the REST API calls in time units and compare the model performances for the following approaches: 
* Deploying the [MobileViT XX Small](https://huggingface.co/apple/mobilevit-xx-small) Huggingface model with a custom torchserve handler. (refer `HF-only` directory)
* Deploying the [MobileViT XX Small](https://huggingface.co/apple/mobilevit-xx-small) Huggingface model in [scripted mode](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) with a custom torchserve handler. (refer `HF-scripted` directory)

## Latest version
The latest changes are reflected in [this repo](https://github.com/tripathiarpan20/HF-torchserve-pipeline)

## References
* https://github.com/pytorch/serve
* https://github.com/huggingface/transformers
* https://github.com/huggingface/optimum
* https://huggingface.co/docs/transformers/main_classes/pipelines
* [My Torchserve + AWS Notion journal](https://garrulous-saxophone-8a6.notion.site/AWS-Torchserve-resources-52fdfd81fa1c4a5ebb9a5fd7398ed552)
* https://huggingface.co/apple/mobilevit-xx-small
* https://huggingface.co/course/chapter2/2?fw=pt
* https://huggingface.co/docs/transformers/main_classes/pipelines
* https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-siamese-network-with-torchserve
* https://github.com/cceyda/lit-NER
* https://github.com/tescal2/TorchServeOnAWS

## Support
There are many ways to support an open-source work, ⭐ing it is one of them. 

## Issues
In case of bugs or queries, raise an Issue, or even better, raise a PR with fixes.