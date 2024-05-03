# Text to Speech synthesis with SpeechT5

This is an example showing text to speech synthesis using SpeechT5 model. This has been verified to work on (linux-aarch64) Graviton 3 instance

While  running this model on `linux-aarch64`, you can enable these optimizations

```
export DNNL_DEFAULT_FPMATH_MODE=BF16
export LRU_CACHE_CAPACITY=1024
```
More details can be found in this [blog](https://pytorch.org/blog/optimized-pytorch-w-graviton/)


## Pre-requisites
```
chmod +x setup.sh
./setup.sh
```

## Download model

This saves the model artifacts to `model_artifacts` directory
```
huggingface-cli login
python download_model.py
```

## Create model archiver

```
mkdir model_store

torch-model-archiver --model-name SpeechT5-TTS --version 1.0 --handler text_to_speech_handler.py --config-file model-config.yaml --archive-format no-archive --export-path model_store -f

mv model_artifacts/* model_store/SpeechT5-TTS/
```

## Start TorchServe

```
torchserve --start --ncs --model-store model_store --models SpeechT5-TTS
```

## Send Inference request

```
curl http://127.0.0.1:8080/predictions/SpeechT5-TTS -T sample_input.txt  -o speech.wav
```

This generates an audio file `speech.wav` corresponding to the text in `sample_input.txt`
