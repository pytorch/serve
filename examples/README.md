# Contents of this Document
* [Creating mar file for an eager mode model](#creating-mar-file-for-eager-mode-model)
* [Creating mar file for an eager mode model](#creating-mar-file-for-torchscript-mode-model)
* [Serving torchvision image classification models in TorchServe](#serving-torchvision-image-classification-models-in-torchserve)
  * [Serving densenet161 model](#serving-torchvision-image-classification-models)
  * [Serving resnet18 model](#example-to-serve-resnet18-image-classification-model)
* [Serving custom model with custom service handler](#example-to-serve-a-custom-model-with-custom-service-handler)
* [Serving text classification model](#example-to-serve-text-classification-model)
* [Serving object detection model](#example-to-serve-object-detection-model)
* [Serving image segmentation model](#example-to-serve-image-segmentation-model)
* [Serving speech synthesis model](#example-to-serve-speech-synthesis-model)

# TorchServe Examples

The following are examples on how to create and serve model archives with TorchServe.

## Creating mar file for eager mode model

Following are the steps to create a torch-model-archive (.mar) to execute an eager mode torch model in TorchServe :
    
* Pre-requisites to create a torch model archive (.mar) :
    * serialized-file (.pt) : This file represents the state_dict in case of eager mode model.
    * model-file (.py) : This file contains model class extended from torch nn.modules representing the model architecture. This parameter is mandatory for eager mode models. This file must contain only one class definition extended from torch.nn.modules
    * index_to_name.json : This file contains the mapping of predicted index to class. The default TorchServe handles returns the predicted index and probability. This file can be passed to model archiver using --extra-files parameter.
    * version : Model version must be a valid non-negative floating point number
    * handler : TorchServe default handler's name or path to custom inference handler(.py)
* Syntax

    ```bash
    torch-model-archiver --model-name <model_name> --version <model_version_number> --model-file <path_to_model_architecture_file> --serialized-file <path_to_state_dict_file> --handler <path_to_custom_handler_or_default_handler_name> --extra-files <path_to_index_to_name_json_file>
    ```
  
## Creating mar file for torchscript mode model

Following are the steps to create a torch-model-archive (.mar) to execute an eager mode torch model in TorchServe :
    
* Pre-requisites to create a torch model archive (.mar) :
    * serialized-file (.pt) : This file represents the state_dict in case of eager mode model or an executable ScriptModule in case of TorchScript. 
    * index_to_name.json : This file contains the mapping of predicted index to class. The default TorchServe handles returns the predicted index and probability. This file can be passed to model archiver using --extra-files parameter.
    * version : Model version must be a valid non-negative floating point number
    * handler : TorchServe default handler's name or path to custom inference handler(.py)
    
* Syntax

    ```bash
    torch-model-archiver --model-name <model_name> --version <model_version_number> --serialized-file <path_to_executable_script_module> --extra-files <path_to_index_to_name_json_file> --handler <path_to_custom_handler_or_default_handler_name>
    ```  

## Serving torchvision image classification models
 
Following example demonstrates serving an image classification models available in torchvision using TorchServe's default image_classifier handler

* TorchVision Image Classification Models : Download a pre-trained model state_dict for computer vision model that classifies images from the following :

  * [Image Classification with AlexNet](image_classifier/alexnet) - https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
  * [Image Classification with DenseNet161](image_classifier/densenet_161) - https://download.pytorch.org/models/densenet161-8d451a50.pth
  * [Image Classification with ResNet18](image_classifier/resnet_18) - https://download.pytorch.org/models/resnet18-5c106cde.pth
  * [Image Classification with SqueezeNet 1_1](image_classifier/squeezenet) - https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth
  * [Image Classification with VGG11](image_classifier/vgg_11) - https://download.pytorch.org/models/vgg11-bbd30ac9.pth

* Create a model architecture file (model-file) based on selected model or use the sample provided with above examples.

* You can use the [index_to_name.json](image_classifier/index_to_name.json) file for mapping predicted index to class or use a custom one.

* Create a torch model archive file using the above provided syntax command.

#### Sample commands to create a DenseNet161 eager mode model archive, register it on TorchServe and run image prediction

    ```bash
    wget https://download.pytorch.org/models/densenet161-8d451a50.pth
    torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --handler image_classifier --extra-files examples/image_classifier/index_to_name.json
    mkdir model_store
    mv densenet161.mar model_store/
    torchserve --start --model-store model_store --models densenet161=densenet161.mar
    curl -X POST http://127.0.0.1:8080/predictions/densenet161 -T examples/image_classifier/kitten.jpg
    ```

#### TorchScript example using DenseNet161 image classifier:

* Save the Densenet161 model in as an executable script module or a traced script:

1. Save model using scripting


```python
#scripted mode
from torchvision import models
import torch
model = models.densenet161(pretrained=True)
sm = torch.jit.script(model)
sm.save("densenet161.pt")
```

2. Save model using tracing

```python
#traced mode
from torchvision import models
import torch
model = models.densenet161(pretrained=True)
example_input = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("dense161.pt")
```  
 
* Use following commands to register Densenet161 torchscript model on TorchServe and run image prediction

    ```bash
    torch-model-archiver --model-name densenet161_ts --version 1.0 --serialized-file densenet161.pt --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
    mkdir model_store
    mv densenet161_ts.mar model_store/
    torchserve --start --model-store model_store --models densenet161=densenet161_ts.mar
    curl -X POST http://127.0.0.1:8080/predictions/densenet161 -T examples/image_classifier/kitten.jpg
    ```
  
## Example to serve Resnet18 image classification model
The following example demonstrates how to create a Resnet18 image classifier model archive, serve it on TorchServe and run image prediction using TorchServe's default image_classifier handler :

* [Image classification using Resnet18](image_classifier/resnet_18)

## Example to serve a Custom Model with Custom Service Handler

The following example demonstrates how to create and serve a custom NN model with custom handler archives in TorchServe :

* [Digit recognition with MNIST](image_classifier/mnist)

## Example to serve Text classification model

The following example demonstrates how to create and serve a custom text_classification NN model with default text_classifer handler provided by TorchServe :

* [Text classification example](text_classification)

## Example to serve Object Detection model

The following example demonstrates how to create and serve a pretrained fast-rcnn NN model with default object_detector handler provided by TorchServe :

* [Object detection example](object_detector)

## Example to serve Image Segmentation model

The following example demonstrates how to create and serve a pretrained fcn NN model with default image_segmenter handler provided by TorchServe :

* [Image segmentation example](image_segmenter)

## Example to serve Speech Synthesis model

The following example demonstrates how to create and serve a speech synthesis model with default a custom handler provided by TorchServe :

* [Speech synthesis example](text_to_speech_synthesizer)
