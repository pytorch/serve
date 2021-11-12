# Torchprep

A CLI tool to prepare your Pytorch models for efficient inference. The only prerequisite is a model trained and saved with `torch.save(model_name, model_path)`. See `example.py` for an example.

**Be warned**: `torchprep` is an experimental tool so expect bugs, deprecations and limitations. That said if you like the project and would like to improve it please open up a Github issue!

## Install from source

Create a virtual environment 

```sh
apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
```

Install `poetry`

```sh
sudo python3 -m pip install -U pip
sudo python3 -m pip install -U setuptools
pip install poetry
```

Install `torchprep`

```sh
cd torchprep
poetry install
```

## Install from Pypi (Coming soon)

```sh
pip install torchprep
```

## Usage

```sh
torchprep quantize --help
```

### Example

```sh
# Install example dependencies
pip install torchvision transformers

# Download resnet example
python example.py

# quantize a cpu model with int8 on cpu and profile with a float tensor of shape [64,3,7,7]
torchprep quantize models/resnet152.pt int8 --input-shape 64,3,7,7

# profile a model for a 100 iterations
torchprep profile models/resnet152.pt --iterations 100 --device cpu --input-shape 64,3,7,7

# set omp threads to 1 to optimize cpu inference
torchprep env --device cpu

# Prune 30% of model weights
torchprep prune models/resnet152.pt --prune-amount 0.3
```


### Available commands


```
Usage: torchprep [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.
  --help                Show this message and exit.

Commands:
  distill        Create a smaller student model by setting a distillation...
  prune          Zero out small model weights using l1 norm
  env-variables  Set environment variables for optimized inference.
  fuse           Supports optimizations including conv/bn fusion, dropout...
  profile        Profile model latency 
  quantize       Quantize a saved torch model to a lower precision float...
```

### Usage instructions for a command

`torchprep <command> --help`

```
Usage: torchprep quantize [OPTIONS] MODEL_PATH PRECISION:{int8|float16}

  Quantize a saved torch model to a lower precision float format to reduce its
  size and latency

Arguments:
  MODEL_PATH                [required]
  PRECISION:{int8|float16}  [required]

Options:
  --device [cpu|gpu]  [default: Device.cpu]
  --input-shape TEXT  Comma seperated input tensor shape
  --help              Show this message and exit.
```

### Create binaries

To create binaries and test them out locally

```sh
poetry build
pip install --user /path/to/wheel
```

### Upload to Pypi

```sh
poetry config pypi-token.pypi <SECRET_KEY>
poetry publish --build
```

## Roadmap
* Supporting add custom model names and output paths
* Support multiple input tensors for models like BERT that expect a batch size and sequence length
* Support multiple input tensor types
* Automatic distillation example: Reduce parameter count by 1/3 `torchprep distill model.pt 1/3`
* Automated release with github actions
* TensorRT, IPEX, AMP and autocast support
* Training aware optimizations
* Get model input shape using fx instead of asking user for it
* Refactor profiling, loading and saving into seperate functions
* More environment variable setting and a way to reverse environment variables (e.g: save current ones in user file)
