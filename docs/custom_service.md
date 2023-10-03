# Custom Service

## Contents of this Document

* [Custom handlers](#custom-handlers)
* [Creating model archive with entry point](#creating-a-model-archive-with-an-entry-point)
* [Handling model execution on GPU](#handling-model-execution-on-multiple-gpus)
* [Installing model specific python dependencies](#installing-model-specific-python-dependencies)

## Custom handlers

Customize the behavior of TorchServe by writing a Python script that you package with
the model when you use the model archiver. TorchServe executes this code when it runs.

Provide a custom script to:
* Initialize the model instance
* Pre-process input data before it is sent to the model for inference or Captum explanations
* Customize how the model is invoked for inference or explanations
* Post-process output from the model before sending back a response

Following is applicable to all types of custom handlers
* **data** - The input data from the incoming request
* **context** - Is the TorchServe [context](https://github.com/pytorch/serve/blob/master/ts/context.py). You can use following information for customization
model_name, model_dir, manifest, batch_size, gpu etc.

### Start with BaseHandler!
[BaseHandler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py) implements most of the functionality you need. You can derive a new class from it, as shown in the examples and default handlers. Most of the time, you'll only need to override `preprocess` or `postprocess`.

### Custom handler with `module` level entry point

The custom handler file must define a module level function that acts as an entry point for execution.
The function can have any name, but it must accept the following parameters and return prediction results.

The signature of a entry point function is:

```python
# Create model object
model = None

def entry_point_function_name(data, context):
    """
    Works on data and context to create model object or process inference request.
    Following sample demonstrates how model object can be initialized for jit mode.
    Similarly you can do it for eager mode models.
    :param data: Input data for prediction
    :param context: context contains model server system properties
    :return: prediction output
    """
    global model

    if not data:
        manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        model = torch.jit.load(model_pt_path)
    else:
        #infer and return result
        return model(data)

```

This entry point is engaged in two cases:

1. TorchServe is asked to scale a model out to increase the number of backend workers (it is done either via a `PUT /models/{model_name}` request
    or a `POST /models` request with `initial-workers` option or during TorchServe startup when you use the `--models` option (`torchserve --start --models {model_name=model.mar}`), ie., you provide model(s) to load)
2. TorchServe gets a `POST /predictions/{model_name}` request.

(1) is used to scale-up or scale-down workers for a model. (2) is used as a standard way to run inference against a model. (1) is also known as model load time.
Typically, you want code for model initialization to run at model load time.
You can find out more about these and other TorchServe APIs in [TorchServe Management API](./management_api.md) and [TorchServe Inference API](./inference_api.md)

### Custom handler with `class` level entry point

You can create custom handler by having class with any name, but it must have an `initialize` and a `handle` method.

NOTE - If you plan to have multiple classes in same python module/file then make sure that handler class is the first in the list

The signature of a entry point class and functions is:

```python
class ModelHandler(object):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        self.initialized = True


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        pred_out = self.model.forward(data)
        return pred_out
```

### Advanced custom handlers

#### Returning custom error codes

To return a custom error code back to the user via custom handler with `module` level entry point.

```python
from ts.utils.util import PredictionException
def handle(data, context):
    # Some unexpected error - returning error code 513
    raise PredictionException("Some Prediction Error", 513)
```

To return a custom error code back to the user via custom handler with `class` level entry point.

```python
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import PredictionException

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def handle(self, data, context):
        # Some unexpected error - returning error code 513
        raise PredictionException("Some Prediction Error", 513)
```

#### Writing a custom handler from scratch for Prediction and Explanations Request

*You should generally derive from BaseHandler and ONLY override methods whose behavior needs to change!* As you can see in the examples, most of the time you only need to override `preprocess` or `postprocess`

Nonetheless, you are able to write a class from scratch. Below is an example. Basically, it follows a typical Init-Pre-Infer-Post pattern to create maintainable custom handler.

```python
# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        preprocessed_data = data[0].get("data")
        if preprocessed_data is None:
            preprocessed_data = data[0].get("body")

        return preprocessed_data


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model.forward(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)

```

Refer [waveglow_handler](https://github.com/pytorch/serve/blob/master/examples/text_to_speech_synthesizer/waveglow_handler.py) for more details.

#### Captum explanations for custom handler

Torchserve returns the captum explanations for Image Classification, Text Classification and BERT models. It is achieved by placing the below request:
 `POST /explanations/{model_name}`

The explanations are written as a part of the explain_handle method of base handler. The base handler invokes this explain_handle_method. The arguments that are passed to the explain handle methods are the pre-processed data and the raw data. It invokes the get insights function of the custom handler that returns the captum attributions. The user should write his own get_insights functionality to get the explanations 

For serving a custom handler the captum algorithm should be initialized in the initialize functions of the handler 

The user can override the explain_handle function in the custom handler.
The user should define their get_insights method for custom handler to get Captum Attributions. 

The above ModelHandler class should have the following methods with captum functionality.

```python

    def initialize(self, context):
        """
        Load the model and its artifacts
        """
        .....
        self.lig = LayerIntegratedGradients(
                captum_sequence_forward, self.model.bert.embeddings
            )

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction/explanation request.
        Do pre-processing of data, prediction using model and postprocessing of prediction/explanations output
        :param data: Input data for prediction/explanation
        :param context: Initial context contains model server system properties.
        :return: prediction/ explanations output
        """
        model_input = self.preprocess(data)
        if not self._is_explain():
                model_output = self.inference(model_input)
                model_output = self.postprocess(model_output)
            else :
                model_output = self.explain_handle(model_input, data)
            return model_output
    
    # Present in the base_handler, so override only when neccessary
    def explain_handle(self, data_preprocess, raw_data):
        """Captum explanations handler

        Args:
            data_preprocess (Torch Tensor): Preprocessed data to be used for captum
            raw_data (list): The unprocessed data to get target from the request

        Returns:
            dict : A dictionary response with the explanations response.
        """
        output_explain = None
        inputs = None
        target = 0

        logger.info("Calculating Explanations")
        row = raw_data[0]
        if isinstance(row, dict):
            logger.info("Getting data and target")
            inputs = row.get("data") or row.get("body")
            target = row.get("target")
            if not target:
                target = 0

        output_explain = self.get_insights(data_preprocess, inputs, target)
        return output_explain

    def get_insights(self,**kwargs):
        """
        Functionality to get the explanations.
        Called from the explain_handle method 
        """
        pass
```

#### Extend default handlers

TorchServe has following default handlers.
- [image_classifier](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_classifier.py)
- [image_segmenter](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_segmenter.py)
- [object_detector](https://github.com/pytorch/serve/blob/master/ts/torch_handler/object_detector.py)
- [text_classifier](https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_classifier.py)

If required above handlers can be extended to create custom handler. Also, you can extend abstract [base_handler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py).

To import the default handler in a python script use the following import statement.

`from ts.torch_handler.<default_handler_name> import <DefaultHandlerClass>`

Following is an example of custom handler extending default image_classifier handler.

```python
from ts.torch_handler.image_classifier import ImageClassifier

class CustomImageClassifier(ImageClassifier):

    def preprocess(self, data):
        """
        Overriding this method for custom preprocessing.
        :param data: raw data to be transformed
        :return: preprocessed data for model input
        """
        # custom pre-procsess code goes here
        return data

```
For more details refer following examples :
- [mnist digit classifier handler](https://github.com/pytorch/serve/tree/master/examples/image_classifier)
- [Huggingface transformer generalized handler](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers/Transformer_handler_generalized.py)
- [Waveglow text to speech synthesizer](https://github.com/pytorch/serve/blob/master/examples/text_to_speech_synthesizer/waveglow_handler.py)

## Creating a model archive with an entry point

TorchServe identifies the entry point to the custom service from a manifest file.
When you create the model archive, specify the location of the entry point by using the `--handler` option.

The [model-archiver](https://github.com/pytorch/serve/tree/master/model-archiver) tool enables you to create a model archive that TorchServe can serve.

```bash
torch-model-archiver --model-name <model-name> --version <model_version_number> --handler model_handler[:<entry_point_function_name>] [--model-file <path_to_model_architecture_file>] --serialized-file <path_to_state_dict_file> [--extra-files <comma_seperarted_additional_files>] [--export-path <output-dir> --model-path <model_dir>] [--runtime python3]
```

NOTE -
1. Options in [] are optional.
2. `entry_point_function_name` can be skipped if it is named as `handle` in your [handler module](#custom-handler-with-module-level-entry-point) or handler is [python class](#custom-handler-with-class-level-entry-point)

This creates the file `<model-name>.mar` in the directory `<output-dir>` for python3 runtime. The `--runtime` parameter enables usage of a specific python version at runtime.
By default it uses the default python distribution of the system.

Example
```bash
torch-model-archiver --model-name waveglow_synthesizer --version 1.0 --model-file waveglow_model.py --serialized-file nvidia_waveglowpyt_fp32_20190306.pth --handler waveglow_handler.py --extra-files tacotron.zip,nvidia_tacotron2pyt_fp32_20190306.pth
```

## Handling model execution on multiple GPUs

TorchServe scales backend workers on vCPUs or GPUs. In case of multiple GPUs TorchServe selects the gpu device in round-robin fashion and passes on this device id to the model handler in context object.
User should use this GPU ID for creating pytorch device object to ensure that all the workers are not created in the same GPU.
The following code snippet can be used in model handler to create the PyTorch device object:

```python
import torch

class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.device = None

    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
```

## Installing model specific python dependencies

Custom models/handlers may depend on different python packages which are not installed by-default as a part of `TorchServe` setup.

Following steps allows user to supply a list of custom python packages to be installed by `TorchServe` for seamless model serving.

1) [Enable model specific python package installation](https://pytorch.org/serve/configuration.html#allow-model-specific-custom-python-packages)
2) [Supply a requirements file with the model-archive](https://github.com/pytorch/serve/tree/master/model-archiver#torch-model-archiver-command-line-interface).
