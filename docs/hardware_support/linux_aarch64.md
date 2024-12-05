# TorchServe on linux aarch64 - Experimental

TorchServe has been tested to be working on linux aarch64 for some of the examples.
- Tested this on Amazon Graviton 3 instance(m7g.4x.large)

## Installation

Currently installation from PyPi or installing from source works

```
python ts_scripts/install_dependencies.py
pip install torchserve torch-model-archiver torch-workflow-archiver
```

## Optimizations

You can also enable this optimizations for Graviton 3 to get an improved performance. More details can be found in this [blog](https://pytorch.org/blog/optimized-pytorch-w-graviton/)
```
export DNNL_DEFAULT_FPMATH_MODE=BF16
export LRU_CACHE_CAPACITY=1024
```

## Example

This [example](https://github.com/pytorch/serve/tree/master/examples/text_to_speech_synthesizer/SpeechT5) on Text to Speech synthesis was verified to be working on Graviton 3

## To Dos
- CI
- Regression tests
