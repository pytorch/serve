# Torchprep

A CLI tool to prepare your Pytorch models for inference

## Installation

```sh
pip install -r requirements.txt
```

## Usage
* Float16 precision on GPU `python main.py quantize --precision float16 --device gpu`
* Float16 quantization on CPU `python main.py quantize --float16`
* Int8 quantization on CPU profiled with a random matrix of size 2x2: `python main.py quantize --precision int8 --profile [2,2]`

## Coming soon
* Reduce parameter count by 1/3 `python main.py distil model.pt -distil 1/3`
* And much more
* pip installation instructions
* Baseline models resnet18 and BERT 
* Automated release with github actions