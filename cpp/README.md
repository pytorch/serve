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
#### Example
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
##### BabyLLama Example
The babyllama example can be found [here](https://github.com/pytorch/serve/blob/master/cpp/src/examples/babyllama/).
To run the example we need to download the weights as well as tokenizer files:
```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
```
Subsequently, we need to adjust the paths according to our local file structure in [config.json](https://github.com/pytorch/serve/blob/master/serve/cpp/test/resources/examples/babyllama/babyllama_handler/config.json).
```bash
{
"checkpoint_path" : "/home/ubuntu/serve/cpp/stories15M.bin",
"tokenizer_path" : "/home/ubuntu/serve/cpp/src/examples/babyllama/tokenizer.bin"
}
```
Then we can create the mar file and deploy it with:
```bash
cd serve/cpp/test/resources/examples/babyllama/babyllama_handler
torch-model-archiver --model-name llm --version 1.0 --handler libbabyllama_handler:BabyLlamaHandler --runtime LSP --extra-files config.json
mkdir model_store && mv llm.mar model_store/
torchserve --ncs --start --model-store model_store

curl -v -X POST "http://localhost:8081/models?initial_workers=1&url=llm.mar"
```
The handler name `libbabyllama_handler:BabyLlamaHandler` consists of our shared library name (as defined in our [CMakeLists.txt](https://github.com/pytorch/serve/blob/master/serve/cpp/src/examples/CMakeLists.txt)) as well as the class name we chose for our [custom handler class](https://github.com/pytorch/serve/blob/master/serve/cpp/src/examples/babyllama/baby_llama_handler.cc) which derives its properties from BaseHandler.

To test the model we can run:
```bash
cd serve/cpp/test/resources/examples/babyllama/
curl http://localhost:8080/predictions/llm -T prompt.txt
```
##### Mnist example
* Transform data on client side. For example:
```
import torch
from PIL import Image
from torchvision import transforms

image_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
image = Image.open("examples/image_classifier/mnist/test_data/0.png")
image = image_processing(image)
torch.save(image, "0_png.pt")
```
* Run model registration and prediction: [Using BaseHandler](serve/cpp/test/backends/torch_scripted/torch_scripted_backend_test.cc#L54) or [Using customized handler](serve/cpp/test/backends/torch_scripted/torch_scripted_backend_test.cc#L72).
