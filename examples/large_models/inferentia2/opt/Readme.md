# Large model inference on Inferentia2

This document briefs on serving large HuggingFace (HF) models on [AWS Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/) instances.

Inferentia2 uses [Neuron SDK](https://aws.amazon.com/machine-learning/neuron/) which is build on top of PyTorch XLA stack. For large model inference [`transformers-neuronx`](https://github.com/aws-neuron/transformers-neuronx) package is used that takes care of model partitioning and running inference.

Let's take a look at the steps to prepare our model for inference on Inf2 instances.

**Note** To run the model on an Inf2 instance, the model gets compiled as a preprocessing step. As part of the compilation process, to generate the model graph, a specific batch size is used. Following this, when running inference, we need to pass the same batch size that was used during compilation. This is taken care of by the [custom handler](inf2_handler.py) in this example.

### Step 1: Inf2 instance

Get an Inf2 instance, ssh to it, make sure to use the following DLAMI as it comes with PyTorch and necessary packages for AWS Neuron SDK pre-installed.
DLAMI Name: ` Deep Learning AMI Neuron PyTorch 1.13.0 (Ubuntu 20.04) 20230226 Amazon Machine Image (AMI)`

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

# Install torchserve and torch-model-archiver
python -m pip install --upgrade torch-model-archiver torchserve

# Clone Torchserve git repository
git clone https://github.com/pytorch/serve.git
cd serve

# Install dependencies
python ts_scripts/install_dependencies.py --neuronx

# Set pip repository pointing to the Neuron repository
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Update Neuron Compiler, Framework and Transformers
python -m pip install --upgrade neuronx-cc torch-neuronx transformers-neuronx

# Install additional necessary packages
python -m pip install --upgrade transformers

```



### Step 2: Save the model split checkpoints compatible with `transformers-neuronx`

Navigate to `examples/large_models/inferentia2/opt` directory
```bash
cd examples/large_models/inferentia2/opt/
```

Run the `inf2_save_split_checkpoints.py` script
```bash
 python ../util/inf2_save_split_checkpoints.py --model_name facebook/opt-6.7b --save_path './opt-6.7b-split'
```


### Step 3: Generate Tar/ MAR file

```bash
torch-model-archiver --model-name opt --version 1.0 --handler inf2_handler.py --extra-files ./opt-6.7b-split  -r requirements.txt --config-file model-config.yaml --archive-format no-archive
```

### Step 4: Add the mar file to model store

```bash
mkdir model_store
mv opt model_store
```

### Step 5: Start torchserve

```bash
torchserve --ncs --start --model-store model_store
```

### Step 6: Register model

```bash
curl -X POST "http://localhost:8081/models?url=opt"
```

### Step 7: Run inference

```bash
curl "http://localhost:8080/predictions/opt" -T sample_text.txt
```

### Step 8: Stop torchserve

```bash
torchserve --stop
```
