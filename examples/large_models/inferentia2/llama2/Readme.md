# Large model inference on Inferentia2

This document briefs on serving the [Llama 2](https://huggingface.co/meta-llama) model on [AWS Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/) with [micro batching](https://github.com/pytorch/serve/tree/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/examples/micro_batching) and [streaming response](https://github.com/pytorch/serve/blob/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/docs/inference_api.md#curl-example-1) support.

Inferentia2 uses [Neuron SDK](https://aws.amazon.com/machine-learning/neuron/) which is built on top of PyTorch XLA stack. For large model inference [`transformers-neuronx`](https://github.com/aws-neuron/transformers-neuronx) package is used that takes care of model partitioning and running inference.

Let's take a look at the steps to prepare our model for inference on Inf2 instances.

**Note** To run the model on an Inf2 instance, the model gets compiled as a preprocessing step. As part of the compilation process, to generate the model graph, a specific batch size is used. Following this, when running inference, we need to pass the same batch size that was used during compilation. This batch size and micro batch size for this example are present in `model-config.yaml`.

### Step 1: Inf2 instance

Get an Inf2 instance(Note: This example was tested on instance type:`inf2.24xlarge`), ssh to it, make sure to use the following DLAMI as it comes with PyTorch and necessary packages for AWS Neuron SDK pre-installed.
DLAMI Name: ` Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) 20230720 Amazon Machine Image (AMI)`

### Step 1: Package Installations

Follow the steps below to complete package installations

```bash
sudo apt-get update
sudo apt-get upgrade

# Update Neuron Runtime
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Activate Python venv
source /opt/aws_neuron_venv_pytorch/bin/activate

# Clone Torchserve git repository
git clone https://github.com/pytorch/serve.git
cd serve

# Install dependencies
python ts_scripts/install_dependencies.py --neuronx

# Install torchserve and torch-model-archiver
python ts_scripts/install_from_src.py

# Set pip repository pointing to the Neuron repository
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Update Neuron Compiler, Framework and Transformers
python -m pip install --upgrade neuronx-cc torch-neuronx transformers-neuronx

# Install additional necessary packages
python -m pip install --upgrade transformers tokenizers sentencepiece

```



### Step 2: Save the model split checkpoints compatible with `transformers-neuronx`
Login to Huggingface
```bash
huggingface-cli login
```

Navigate to `examples/large_models/inferentia2/llama2` directory
```bash
cd examples/large_models/inferentia2/llama2/
```

Run the `inf2_save_split_checkpoints.py` script
```bash
python ../util/inf2_save_split_checkpoints.py --model_name meta-llama/Llama-2-13b-hf --save_path './llama-2-13b-split'
```


### Step 3: Generate Tar/ MAR file

```bash
torch-model-archiver --model-name llama-2-13b --version 1.0 --handler inf2_handler.py --extra-files ./llama-2-13b-split  -r requirements.txt --config-file model-config.yaml --archive-format no-archive
```

### Step 4: Add the mar file to model store

```bash
mkdir model_store
mv llama-2-13b model_store
```

### Step 5: Start torchserve

```bash
torchserve --ncs --start --model-store model_store
```

### Step 6: Register model

```bash
curl -X POST "http://localhost:8081/models?url=llama-2-13b"
```

### Step 7: Run inference

```bash
python test_stream_response.py
```

### Step 8: Stop torchserve

```bash
torchserve --stop
```
