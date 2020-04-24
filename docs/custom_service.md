# Custom Service

Customize the behavior of TorchServe by writing a Python script that you package with
the model when you use the model archiver. TorchServe executes this code when it runs.

Provide a custom script to:

* Pre-process input data before it is sent to the model for inference
* Customize how the model is invoked for inference
* Post-process output from the model before sending the response to the user

## Contents of this Document

* [Requirements for a custom service file](#requirements-for-custom-service-file)
* [Example Custom Service file](#example-custom-service-file)
* [Creating model archive with entry point](#creating-model-archive-with-entry-point)
* [Handling model execution on GPU](#handling-model-execution-on-multiple-gpus)

## Requirements for custom service file

The custom service file must define a method that acts as an entry point for execution. This function is invoked by TorchServe on a inference request.
The function can have any name, but it must accept the following parameters:

* **data** - The input data from the incoming request
* **context** - Is the TorchServe [context](https://github.com/pytorch/serve/blob/master/ts/context.py) information passed for use with the custom service if required.

The signature of a entry point function is:

```python
def function_name(data,context):
    """
    Works on data and context passed
    """
    # Use parameters passed
```

The following code shows an example custom service.

## Example Custom Service file

```python
# custom service file

# model_handler.py

"""
ModelHandler defines a base model handler.
"""
import logging


class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True

    def preprocess(self, batch):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        return None

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        return None

    def postprocess(self, inference_output):
        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        return ["OK"] * self._batch_size

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)

```

Here the ` handle()` method is our entry point that will be invoked by TorchServe. It accepts parameters `data` and `context`,
and in turn can pass this information to an actual inference class object or handle all the processing in the``handle()` method itself.
The `initialize()` method is used to initialize the model at load time, so after the first time.
The service doesn't need to be re-initialized in the the life cycle of the relevant worker.
We recommend using an `initialize()` method to avoid initialization at prediction time.

This entry point is engaged in two cases:

1. TorchServe is asked to scale a model out to increase the number of backend workers (it is done either via a `PUT /models/{model_name}` request or a `POST /models` request with `initial-workers` option or during TorchServe startup when you use the `--models` option (`torchserve --start --models {model_name=model.mar}`), ie., you provide model(s) to load)
1. TorchServe gets a `POST /predictions/{model_name}` request.

(1) is used to scale-up or scale-down workers for a model. (2) is used as a standard way to run inference against a model. (1) is also known as model load time.
Typically, you want code for model initialization to run at model load time.
You can find out more about these and other TorchServe APIs in [TorchServe Management API](./management_api.md) and [TorchServe Inference API](./inference_api.md)

** For a working example of a custom service handler, see [mnist digit classifier handler](../examples/image_classifier/mnist/mnist_handler.py) **

## Handling model execution on multiple GPUs

TorchServe scales backend workers on vCPUs or GPUs. In case of multiple GPUs TorchServe selects the gpu device in round-robin fashion and passes on this device id to the model handler in context object. User should use this GPU ID for creating pytorch device object to ensure that all the workers are not created in the same GPU.

The following code snippet can be used in model handler to create the PyTorch device object:

```python
class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.device

    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
```

** For more details refer [mnist digit classifier handler](../examples/image_classifier/mnist/mnist_handler.py) **

## Creating a model archive with an entry point

TorchServe identifies the entry point to the custom service from a manifest file.
When you create the model archive, specify the location of the entry point by using the ```--handler``` option.

The [model-archiver](https://github.com/pytorch/serve/blob/master/model-archiver/README.md) tool enables you to create a model archive that TorchServe can serve.
The following is an example that archives a model and specifies a custom handler:

```bash
torch-model-archiver --model-name <model-name> --version <model_version_number> --model-file <path_to_model_architecture_file> --serialized-file <path_to_state_dict_file> --extra-files <path_to_index_to_name_json_file> --handler model_handler:handle --export-path <output-dir> --model-path <model_dir> --runtime python3
```

This creates the file `<model-name>.mar` in the directory `<output-dir>`

This will create a model archive for the python3 runtime. The `--runtime` parameter enables usage of a specific python version at runtime.
By default it uses the default python distribution of the system.
