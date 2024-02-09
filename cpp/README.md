# TorchServe CPP (Experimental Release)
## Requirements
* C++17
* GCC version: gcc-9
## Installation and Running TorchServe CPP

### Install dependencies
```
cd serve
python ts_scripts/install_dependencies.py --cpp [--cuda=cu121|cu118]
```
### Building the backend
```
## Dev Build
cd serve/cpp
./build.sh [-g cu121|cu118]

## Install TorchServe from source
cd serve
python ts_scripts/install_from_src.py
```
### Set Environment Var
#### On Mac
```
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')/../../lib:$(python -c 'import site; print(site.getsitepackages()[0])')/ts/cpp/lib
```
#### On Ubuntu
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/cpp/_build/_deps/libtorch/lib:$(python -c 'import site; print(site.getsitepackages()[0])')/ts/cpp/lib
```
### Run TorchServe
```
cd serve
torchserve torchserve --ncs --start --model-store model_store
```
## Backend
TorchServe cpp backend can run as a process, which is similar to [TorchServe Python backend](https://github.com/pytorch/serve/tree/master/ts). By default, TorchServe supports torch scripted model in cpp backend. Other platforms such as MxNet, ONNX can be supported through custom handlers following the TorchScript example [src/backends/handler/torch_scripted_handler.hh](https://github.com/pytorch/serve/blob/master/src/backends/handler/torch_scripted_handler.hh).
### Custom Handler
By default, TorchServe cpp provides a handler for TorchScript [src/backends/handler/torch_scripted_handler.hh](https://github.com/pytorch/serve/blob/master/src/backends/handler/torch_scripted_handler.hh). Its uses the [BaseHandler](https://github.com/pytorch/serve/blob/master/src/backends/handler/base_handler.hh) which defines the APIs to customize handler.
* [Initialize](serve/blob/cpp_backend/cpp/src/backends/handler/base_handler.hh#L29)
* [LoadModel](serve/blob/cpp_backend/cpp/src/backends/handler/base_handler.hh#L37)
* [Preprocess](serve/blob/cpp_backend/cpp/src/backends/handler/base_handler.hh#L40)
* [Inference](serve/blob/cpp_backend/cpp/src/backends/handler/base_handler.hh#L46)
* [Postprocess](serve/blob/cpp_backend/cpp/src/backends/handler/base_handler.hh#L53)
#### Usage
##### Using TorchScriptHandler
* set runtime as "LSP" in model archiver option [--runtime](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
* set handler as "TorchScriptHandler" in model archiver option [--handler](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
```
 torch-model-archiver --model-name mnist_base --version 1.0 --serialized-file mnist_script.pt --handler TorchScriptHandler --runtime LSP
```
Here is an [example](https://github.com/pytorch/serve/tree/cpp_backend/cpp/test/resources/examples/mnist/base_handler) of unzipped model mar file.
##### Using Custom Handler
* build customized handler shared lib. For example [Mnist handler](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/examples/image_classifier/mnist).
* set runtime as "LSP" in model archiver option [--runtime](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
* set handler as "libmnist_handler:MnistHandler" in model archiver option [--handler](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
```
torch-model-archiver --model-name mnist_handler --version 1.0 --serialized-file mnist_script.pt --handler libmnist_handler:MnistHandler --runtime LSP
```
Here is an [example](https://github.com/pytorch/serve/tree/cpp_backend/cpp/test/resources/examples/mnist/mnist_handler) of unzipped model mar file.

#### Examples
We have created a couple of examples that can get you started with the C++ backend.
The examples are all located under serve/examples/cpp and each comes with a detailed description of how to set it up.
The following examples are available:
* [AOTInductor Llama](../examples/cpp/aot_inductor/llama2/)
* [BabyLlama](../examples/cpp/babyllama/)
* [Llama.cpp](../examples/cpp/llamacpp/)
* [MNIST](../examples/cpp/mnist/)
