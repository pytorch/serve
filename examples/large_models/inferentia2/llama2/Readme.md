# Large model inference on Inferentia2

This document briefs on serving the [Llama 2](https://huggingface.co/meta-llama) model on [AWS Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/) for text completion with TorchServe's features:

* demo1: [micro batching](https://github.com/pytorch/serve/tree/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/examples/micro_batching) and [streaming response](https://github.com/pytorch/serve/blob/96450b9d0ab2a7290221f0e07aea5fda8a83efaf/docs/inference_api.md#curl-example-1) support in folder streamer.
* demo2: continuous batching support in folder continuous_batching

Inferentia2 uses [Neuron SDK](https://aws.amazon.com/machine-learning/neuron/) which is built on top of PyTorch XLA stack. For large model inference [`transformers-neuronx`](https://github.com/aws-neuron/transformers-neuronx) package is used that takes care of model partitioning and running inference.

This example folder demonstrates the utilization of neuronx cache to store inf2 model compilation artifacts using the `NEURONX_CACHE` and `NEURON_COMPILE_CACHE_URL` environment variables in the custom handler.
When the model is loaded for the first time, the model is compiled for the configured micro batch size and the compilation artifacts are saved to the neuronx cache.
On subsequent model load, the compilation artifacts in the neuronx cache serves as `Ahead of Time(AOT)` compilation artifacts and significantly reduces the model load time.
For convenience, the compiled model artifacts for this example are made available on the Torchserve model zoo: `s3://torchserve/mar_files/llama-2-13b-neuronx-b4`\
Instructions on how to use the AOT compiled model artifacts is shown below.

### Step 1: Inf2 instance

Get an Inf2 instance(Note: This example was tested on instance type:`inf2.24xlarge`), ssh to it, make sure to use the following DLAMI as it comes with PyTorch and necessary packages for AWS Neuron SDK pre-installed.
DLAMI Name: ` Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) 20230720 Amazon Machine Image (AMI)` or higher.

**Note**: The `inf2.24xlarge` instance consists of 6 neuron chips with 2 neuron cores each. The total accelerator memory is 192GB.
Based on the configuration used in [model-config.yaml](streamer/model-config.yaml), with `tp_degree` set to 6, 3 of the 6 neuron chips are used, i.e 6 neuron cores.
On loading the model, the accelerator memory consumed is 38.1GB (12.7GB per chip).

### Step 2: Package Installations

Follow the steps below to complete package installations

```bash
sudo apt-get update
sudo apt-get upgrade

# Install Neuron libraries, SDK 2.12.2: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/prev/content.html#id8
sudo apt-get install aws-neuronx-dkms=2.11.9.0
sudo apt-get install aws-neuronx-collectives=2.15.16.0*
sudo apt-get install aws-neuronx-runtime-lib=2.15.14.0*

# Activate Python venv
source /opt/aws_neuron_venv_pytorch/bin/activate

# Clone Torchserve git repository
git clone https://github.com/pytorch/serve.git
cd serve

# Install dependencies
python ts_scripts/install_dependencies.py --neuronx --environment=dev

# Install torchserve and torch-model-archiver
python ts_scripts/install_from_src.py

# Install additional neuron packages, SDK 2.12.2: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/prev/content.html#id8
python -m pip install neuronx-cc==2.8.0.25 torch-neuronx==1.13.1.1.9.1 transformers-neuronx==0.5.58

# Navigate to `examples/large_models/inferentia2/llama2` directory
cd examples/large_models/inferentia2/llama2/

# Install additional necessary packages
python -m pip install -r requirements.txt
```

### Step 3: Save the model artifacts compatible with `transformers-neuronx`
In order to use the pre-compiled model artifacts, copy them from the model zoo using the command shown below and skip to **Step 5**
```bash
aws s3 cp s3://torchserve/mar_files/llama-2-13b-neuronx-b4/ llama-2-13b --recursive
```

In order to download and compile the Llama2 model from scratch for support on Inf2:\
Request access to the Llama2 model\
https://huggingface.co/meta-llama/Llama-2-13b-hf

Login to Huggingface
```bash
huggingface-cli login
```

Run the `inf2_save_split_checkpoints.py` script
```bash
python ../util/inf2_save_split_checkpoints.py --model_name meta-llama/Llama-2-13b-hf --save_path './llama-2-13b-split'
```


### Step 4: Package model artifacts

```bash
torch-model-archiver --model-name llama-2-13b --version 1.0 --handler /PATH/TO/inf2_handler.py -r requirements.txt --config-file /PATH/TO/model-config.yaml --archive-format no-archive
mv llama-2-13b-split llama-2-13b
```

### Step 5: Add the model artifacts to model store

```bash
mkdir model_store
mv llama-2-13b model_store
```

### Step 6: Start torchserve

```bash
torchserve --ncs --start --model-store model_store --ts-config config.properties
```

### Step 7: Register model

```bash
curl -X POST "http://localhost:8081/models?url=llama-2-13b"
```

### Step 8: Run inference

```bash
python test_stream_response.py
```

### Step 9: Stop torchserve

```bash
torchserve --stop
```
