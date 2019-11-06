# Model Serving with Amazon Elastic Inference 

## Contents of this Document
* [Introduction](#introduction)
* [Custom Service](#custom-service)
* [Creating a EC2 instance with EIA support](#creating-a-ec2-instance-with-eia-support)
* [Custom Service file with EIA](#custom-service-file-with-eia)
* [Running elastic inference on a resnet-152](#running-elastic-inference-on-a-resnet-152)

## Introduction

Amazon Elastic Inference (EI) is a service that allows you to attach low-cost GPU-powered acceleration to Amazon EC2 and Amazon SageMaker 
instances to reduce the cost of running deep learning inference by up to 75%. With MMS it is easy to deploy a MXNet based model, 
taking advantage of the attachable hardware accelerator called Elastic Inference Accelerator (EIA).
In this document, we explore using EIA attached to a Compute Optimized EC2 instance.

## Custom Service

The capability to run model inference with the EIA can be achieved by building a custom service to use the EIA context rather than a GPU or CPU context. An MXNet version with support for EIA is required.

To understand the basics of writing a custom service file refer to the [Custom Service Documentation](https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md).

## Creating a EC2 instance with EIA support 

To Create an EC2 instance with EIA support there are few pre-requisites. These include:
1. [Configuring a Security Group for Amazon EI](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/working-with-ei.html#ei-security).
2. [Configure AWS PrivateLink Endpoint Services](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/working-with-ei.html#eia-privatelink).
3. [Creating a IAM Role with EI instance policy](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/working-with-ei.html#ei-role-policy).
    
The above steps are explored in detail in [AWS Elastic Inference documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/working-with-ei.html)  

On completing the above steps, following steps need to be followed in launching an instance with EIA support

1. Open the Amazon EC2 console at https://console.aws.amazon.com/ec2/.

2. Choose Launch Instance.

3. Choose one of the Deep Learning AMIs, we recommend Deep Learning AMI v20 or later. This is required to use MXNet with EIA.

4. Choose an Instance Type, we recommend a compute optimized EC2 instance such as c5.2xlarge.

5. Choose Next: Configure Instance Details.

6. Under Configure Instance Details, check the configuration settings. Ensure that you are using the VPC with the security groups for the instance and the Amazon EI accelerator that was set up earlier. For more information, see Configuring Your Security Groups for Amazon EI.

7. For IAM role, select the role that you created in the Configuring an Instance Role with an Amazon EI Policy procedure explained in the above documentation.

8. Select Add an Amazon EI accelerator.

9. Select the size of the Amazon EI accelerator. Your options are  eia1.medium, eia1.large, and eia1.xlarge. We recommend selecting the instance size, based on the model size. For larger models, larger instances offer better performance gains. 

10. (Optional) You can choose to add storage and tags by choosing Next at the bottom of the page. Or, you can let the instance wizard complete the remaining configuration steps for you.

11. Review the configuration of your instance and choose Launch.

12. You are prompted to choose an existing key pair for your instance or to create a new key pair. 

**WARNING: Do NOT select the Proceed without a key pair option. If you launch your instance without a key pair, then you can’t connect to it.**

After making your key pair selection, choose Launch Instances.

It can take a few minutes for the instance to be ready so that you can connect to it. Check that your instance has passed its status checks. You can view this information in the Status Checks column.


## Custom Service file with EIA 

You use two different processing contexts with MXNet and EIA
    1. mxnet.cpu() - used for loading up input data
    2. mxnet.eia() - used for binding network symbols and params to an attached EIA instance


        
We modify the [base model service template](https://github.com/awslabs/mxnet-model-server/blob/master/examples/model_service_template/mxnet_model_service.py) to support EIA.


```python
    def initialize(self, context):
        # NOT COMPLETE CODE, refer template above for it.
        #....
        #....
        #....           
        # Load MXNet module
        # Symbol Context set to eia
        self.mxnet_ctx = mx.eia()
        self.data_ctx = mx.cpu()
        sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_prefix, self.epoch)

        # noinspection PyTypeChecker
        self.mx_model = mx.mod.Module(symbol=sym, context=self.mxnet_ctx,
                                      data_names=data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    
    def inference(self, model_input):
        # NOT COMPLETE CODE, refer template above for it.
        #....
        #....
        #....     
        model_input = [item.as_in_context(self.data_ctx) for item in model_input]
```

The above code shows initialization of two contexts, one for data (on CPU) and other for symbols (on EIA).

Once we have the code ready. We can build a model archive consumable by MMS, using the model-archiver. The [model-archiver](https://github.com/awslabs/mxnet-model-server/blob/master/model-archiver/README.md) tool enables to build to an archive understood by MMS.

```bash
model-archiver --model-name <model-name> --handler model_service:handle --export-path <output-dir> --model-path <model_dir> --runtime python
```

This will create file ```<model-name>.mar``` in the directory ```<output-dir>```.

## Running elastic inference on a resnet-152
A pre-built ResNet-152 model archive that uses Amazon Elastic Inference can be downloaded using the following command:
```bash
$ wget https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-152-eia.mar
```
**NOTE:** The above archive will only work on EIA-enabled instances.

Start the EIA-enabled EC2 instance. If using a Deep Learning AMI, there are two Conda environments (one for Python 2 and one for Python 3), both of which come with MXNet built will EI support and MXNet Model Server. Either of the two can be used.
```bash
# python 3 
$ source activate amazonei_mxnet_p36

# python 2 
$ source activate amazonei_mxnet_p27
```

After entering one of the Conda environments, we start MMS, with Resnet-152 EIA model:
```bash
# Start MMS
$ mxnet-model-server --start  --models resnet-152=https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-152-eia.mar
```

Now the model is ready for some inference requests. Let us download a kitten image for classification:
```bash
$ curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
$ curl -X POST http://127.0.0.1:8080/predictions/resnet-152 -T kitten.jpg
```

The predict endpoint will return a prediction response in JSON. It will look something like the following result:

```json
[
  {
    "probability": 0.7148934602737427,
    "class": "n02123045 tabby, tabby cat"
  },
  {
    "probability": 0.22877734899520874,
    "class": "n02123159 tiger cat"
  },
  {
    "probability": 0.04032360762357712,
    "class": "n02124075 Egyptian cat"
  },
  {
    "probability": 0.008370809257030487,
    "class": "n02127052 lynx, catamount"
  },
  {
    "probability": 0.0006728142034262419,
    "class": "n02129604 tiger, Panthera tigris"
  }
]
```

ResNet-152 identified the tabby cat using Elastic Inference Accelerator while being hosted on MMS. Serving on Amazon EI instances reduces inference costs while benefiting from the performance of a GPU for inference tasks.