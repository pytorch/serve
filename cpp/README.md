# TorchServe CPP (Experimental Release)
## Requirements
* C++17
* GCC version: gcc-9
* cmake version: 3.26.4+
* Linux

For convenience, a [docker container](../docker/README.md#create-torchserve-docker-image) can be used as the development environment to build and install Torchserve CPP
```
cd serve/docker
# For CPU support
./build_image.sh -bt dev -cpp
# For GPU support
./build_image.sh -bt dev -g [-cv cu121|cu118] -cpp
```

Start the container and optionally bind mount a build directory into the container to persist build artifacts across container runs
```
# For CPU support
docker run [-v /path/to/build/dir:/serve/cpp/_build] -it pytorch/torchserve:cpp-dev-cpu /bin/bash
# For GPU support
docker run --gpus all [-v /path/to/build/dir:/serve/cpp/_build] -it pytorch/torchserve:cpp-dev-gpu /bin/bash
```
`Warning`: The dev docker container does not install all necessary dependencies or build Torchserve CPP. Please follow the steps below after starting the container.

## Installation and Running TorchServe CPP
This installation instruction assumes that TorchServe is already installed through pip/conda/source. If this is not the case install it after the `Install dependencies` step through your preferred method.

### Install dependencies
```
cd serve
python ts_scripts/install_dependencies.py --cpp --environment dev [--cuda=cu121|cu118]
```
### Building the backend
Don't forget to install or update TorchServe at this point if it wasn't previously installed. E.g. with:
```
python ts_scripts/install_from_src.py
```

Then build the backend:
```
## Dev Build
cd cpp
./build.sh
```

### Run TorchServe
```
mkdir model_store
export LD_LIBRARY_PATH=`python -c "import torch;from pathlib import Path;p=Path(torch.__file__);print(f\"{(p.parent / 'lib').as_posix()}:{(p.parents[1] / 'nvidia/nccl/lib').as_posix()}\")"`:$LD_LIBRARY_PATH
torchserve --ncs --start --model-store model_store
```

### Clean the build directory
To clean the build directory in order to rebuild from scratch simply delete the cpp/_build directory with
```
rm -rf cpp/_build
```

## Backend
TorchServe cpp backend can run as a process, which is similar to [TorchServe Python backend](https://github.com/pytorch/serve/tree/master/ts). By default, TorchServe supports torch scripted model in cpp backend. Other platforms such as MxNet, ONNX can be supported through custom handlers following the TorchScript example [src/backends/handler/torch_scripted_handler.hh](https://github.com/pytorch/serve/blob/master/cpp/src/backends/handler/torch_scripted_handler.hh).
### Custom Handler
By default, TorchServe cpp provides a handler for TorchScript [src/backends/handler/torch_scripted_handler.hh](https://github.com/pytorch/serve/blob/master/cpp/src/backends/handler/torch_scripted_handler.hh). Its uses the [BaseHandler](https://github.com/pytorch/serve/blob/master/cpp/src/backends/handler/base_handler.hh) which defines the APIs to customize handler.
* [Initialize](https://github.com/pytorch/serve/blob/ba8f96a6e68ca7f63b55d72a21aad364334e4d8e/cpp/src/backends/handler/base_handler.hh#L34)
* [LoadModel](https://github.com/pytorch/serve/blob/ba8f96a6e68ca7f63b55d72a21aad364334e4d8e/cpp/src/backends/handler/base_handler.hh#L41)
* [Preprocess](https://github.com/pytorch/serve/blob/ba8f96a6e68ca7f63b55d72a21aad364334e4d8e/cpp/src/backends/handler/base_handler.hh#L43)
* [Inference](https://github.com/pytorch/serve/blob/ba8f96a6e68ca7f63b55d72a21aad364334e4d8e/cpp/src/backends/handler/base_handler.hh#L49)
* [Postprocess](https://github.com/pytorch/serve/blob/ba8f96a6e68ca7f63b55d72a21aad364334e4d8e/cpp/src/backends/handler/base_handler.hh#L55)
#### Usage
##### Using TorchScriptHandler
* set runtime as "LSP" in model archiver option [--runtime](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
* set handler as "TorchScriptHandler" in model archiver option [--handler](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
```
 torch-model-archiver --model-name mnist_base --version 1.0 --serialized-file mnist_script.pt --handler TorchScriptHandler --runtime LSP
```
Here is an [example](https://github.com/pytorch/serve/tree/master/cpp/test/resources/examples/mnist/base_handler) of unzipped model mar file.
##### Using Custom Handler
* build customized handler shared lib. For example [Mnist handler](https://github.com/pytorch/serve/blob/cpp_backend/cpp/src/examples/image_classifier/mnist).
* set runtime as "LSP" in model archiver option [--runtime](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
* set handler as "libmnist_handler:MnistHandler" in model archiver option [--handler](https://github.com/pytorch/serve/tree/master/model-archiver#arguments)
```
torch-model-archiver --model-name mnist_handler --version 1.0 --serialized-file mnist_script.pt --handler libmnist_handler:MnistHandler --runtime LSP
```
Here is an [example](https://github.com/pytorch/serve/tree/master/cpp/test/resources/examples/mnist/mnist_handler) of unzipped model mar file.

#### Examples
We have created a couple of examples that can get you started with the C++ backend.
The examples are all located under serve/examples/cpp and each comes with a detailed description of how to set it up.
The following examples are available:
* [AOTInductor Llama](../examples/cpp/aot_inductor/llama2/)
* [BabyLlama](../examples/cpp/babyllama/)
* [Llama.cpp](../examples/cpp/llamacpp/)
* [MNIST](../examples/cpp/mnist/)

#### Developing
When making changes to the cpp backend its inconvenient to reinstall TorchServe using ts_scripts/install_from_src.py after every compilation.
To automatically update the model_worker_socket located in ts/cpp/bin/ we can install TorchServe once from source with the `--environment dev`.
This will make the TorchServe installation editable and the updated cpp backend binary is automatically picked up when starting a worker (No restart of TorchServe required).
```
python ts_scripts/install_from_src.py --environment dev
```

#### FAQs
Q: After running ./build.sh TorchServe can not find model_worker_socket
A:
1. See if the binary `model_worker_socket` exists by running:
```bash
python -c "import ts; from pathlib import Path; print((Path(ts.__file__).parent / 'cpp/bin/model_worker_socket').exists())
```
2. Look if ./build.sh was actually successful and if the tests ran without any error at the end. If a test failed the binary will not be copied into the appropriate directory.
3. Make sure you have the right conda/venv environment activated during building that you're also using to run TorchServe.

Q: Build on Mac fails with `Library not loaded: @rpath/libomp.dylib`
A: Install libomp with brew and link in /usr/local/lib
```bash
brew install libomp
sudo ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib /usr/local/lib/libomp.dylib
```

Q: When loading a handler which uses a model exported with torch._export.aot_compile the handler dies with "error: Error in dlopen: MODEL.SO : undefined symbol: SOME_SYMBOL".
A: Make sure that you are using matching libtorch and Pytorch versions for inference and export, respectively.
