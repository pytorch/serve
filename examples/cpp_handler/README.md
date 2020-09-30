#### C++ based handler for TorchScript Model using DenseNet161 image classifier

## How do I invoke C++ API from Python based program?
Since handler runs in Python worker process, in order to invoke C++ API, you need to provide the Python torch_cpp_python_bindings
for your C++ code. [torch_cpp_python_bindings.cpp](torch_cpp_python_bindings.cpp) is the C++ file used in this example. We are using Pybind11 library
to do the bindings and we are using Torch provided API to compile this file JIT and use it as python module so that it can be accessed from
C++. Refer the PyTorch CPP extension [documentation](https://pytorch.org/tutorials/advanced/cpp_extension.html) for more details.
In this example, we have provided python bindings for `initialize`, `handle` methods.

## Understanding the [torch_cpp_python_bindings.cpp](torch_cpp_python_bindings.cpp)
If you do not know how to write a custom handler for TorchServe in Python, we recommend you to go through the documentation [here](docs/custom_service.md) first.
The `initialize` method of the handler gets invoked at the time of model registration and the `handle` method gets
invoked at the time of prediction API call. Generally the handle method is optionally modularized into three parts 'sub_process',
`inference` and `post_process`. Here in the CPP file we implement two methods `initialize` and `handle` and both are
mapped to Python class [CPPHandler's](cpp_handler.py) `initialize` and 'handle' methods. The C++ `initialize` method
accepts the model path and returns the Torch Module to Python layer which is kept at class level. The C++ `handle` method
accepts the Torch Module and raw input image data, it pre-processes the raw data into tensors, does the inference and then
returns top k probabilities and classes by doing post processing the prediction results.

## Pre-requisite
1. Install Pytorch for Python Torch and C++ TorchLib

```shell
pip install torch
```

2. Install Ninja to compile the C++ code
```shell
pip install ninja
```

3. Install C++ OpenCV
The OpenCV library is needed to pre process the input raw images into tensors.

On MacOS:
```shell
brew install opencv
#if you have opencv4 installed, create a symlink as follow.
ln -s /usr/local/include/opencv4/opencv2/ /usr/local/include/opencv2
```

On Ubuntu:
```shell
sudo apt-get install libopencv-dev
```


## Save the Densenet161 model in as an executable script module or a traced script:

  * Save model using scripting
   ```python
   #scripted mode
   from torchvision import models
   import torch
   model = models.densenet161(pretrained=True)
   sm = torch.jit.script(model)
   sm.save("densenet161.pt")
   ```

  * Save model using tracing
   ```python
   #traced mode
   from torchvision import models
   import torch
   model = models.densenet161(pretrained=True)
   model.eval()
   example_input = torch.rand(1, 3, 224, 224)
   traced_script_module = torch.jit.trace(model, example_input)
   traced_script_module.save("densenet161.pt")
   ```  

* Use following commands to register Densenet161 torchscript model on TorchServe and run image prediction

    ```bash
    torch-model-archiver --model-name densenet161_cpp --version 1.0  --serialized-file densenet161.pt --extra-files examples/image_classifier/index_to_name.json,examples/cpp_handler/torch_cpp_python_bindings.cpp --handler examples/cpp_handler/cpp_handler.py
    mkdir model_store
    mv densenet161_cpp.mar model_store/
    torchserve --start --model-store model_store
    curl -v -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=densenet161_cpp.mar"
    curl http://127.0.0.1:8080/predictions/densenet161_cpp -T examples/image_classifier/kitten.jpg
    ```

