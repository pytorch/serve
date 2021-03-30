# Contents of this Document
* [Creating mar file for an eager mode model](#creating-mar-file-for-eager-mode-model)
* [Creating mar file for torchscript mode model](#creating-mar-file-for-torchscript-mode-model)
* [Serving torchvision image classification models](#serving-image-classification-models)
* [Serving custom model with custom service handler](#serving-custom-model-with-custom-service-handler)
* [Serving text classification model](#serving-text-classification-model)
* [Serving object detection model](#serving-object-detection-model)
* [Serving image segmentation model](#serving-image-segmentation-model)
* [Serving huggingface transformers model](#serving-huggingface-transformers)
* [Serving waveglow text to speech synthesizer model](#serving-wavegolw-text-to-speech-synthesizer)

# TorchServe Examples

The following are examples on how to create and serve model archives with TorchServe.

## Creating mar file for eager mode model

Following are the steps to create a torch-model-archive (.mar) to execute an eager mode torch model in TorchServe :

* Pre-requisites to create a torch model archive (.mar) :
    * serialized-file (.pt) : This file represents the state_dict in case of eager mode model.
    * model-file (.py) : This file contains model class extended from torch nn.modules representing the model architecture. This parameter is mandatory for eager mode models. This file must contain only one class definition extended from torch.nn.modules
    * index_to_name.json : This file contains the mapping of predicted index to class. The default TorchServe handles returns the predicted index and probability. This file can be passed to model archiver using --extra-files parameter.
    * version : Model's version.
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
    * version : Model's version.
    * handler : TorchServe default handler's name or path to custom inference handler(.py)

* Syntax

    ```bash
    torch-model-archiver --model-name <model_name> --version <model_version_number> --serialized-file <path_to_executable_script_module> --extra-files <path_to_index_to_name_json_file> --handler <path_to_custom_handler_or_default_handler_name>
    ```  

## Serving image classification models
The following example demonstrates how to create image classifier model archive, serve it on TorchServe and run image prediction using TorchServe's default image_classifier handler :

* [Image classification models](image_classifier)

## Serving custom model with custom service handler

The following example demonstrates how to create and serve a custom NN model with custom handler archives in TorchServe :

* [Digit recognition with MNIST](image_classifier/mnist)

## Serving text classification model

The following example demonstrates how to create and serve a custom text_classification NN model with default text_classifer handler provided by TorchServe :

* [Text classification example](text_classification)

## Serving object detection model

The following example demonstrates how to create and serve a pretrained fast-rcnn NN model with default object_detector handler provided by TorchServe :

* [Object detection example](object_detector)

## Serving image segmentation model

The following example demonstrates how to create and serve a pretrained fcn NN model with default image_segmenter handler provided by TorchServe :

* [Image segmentation example](image_segmenter)


## Serving Huggingface Transformers

The following example demonstrates how to create and serve a pretrained transformer models from Huggingface such as BERT, RoBERTA, XLM :

* [Hugging Face Transformers](Huggingface_Transformers)

## Serving Wavegolw text to speech synthesizer

The following example demonstrates how to create and serve a pretrained transformer models from Huggingface such as BERT, RoBERTA, XLM :

* [Waveglow text to speech synthesizer](text_to_speech_synthesizer)
