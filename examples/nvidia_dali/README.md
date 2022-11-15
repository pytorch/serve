# DALI Optimization integration with Torchserve models

The NVIDIA Data Loading Library (DALI) is a library for data loading and pre-processing to accelerate deep learning applications. It provides a collection of highly optimized building blocks for loading and processing image, video and audio data.

Here, we serve torchserve models with DALI pipeline for optimizing the pre-processing

Refer to [NVIDIA-DALI-Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) for detailed information

### Install dependencies :

Navigate to `serve/examples/nvidia_dali` directory and run the below command to install the dependencies

```bash
pip install -r requirements.txt
```

### Define and Build DALI Pipeline

In DALI, any data processing task has a central object called Pipeline.
Refer to [NVIDIA-DALI](https://github.com/NVIDIA/DALI) for more details on DALI pipeline.

Navigate to `cd ./serve/examples/nvidia_dali`.

Change the `dali_config.json`

`batch_size` - Maximum batch size of pipeline.

`num_threads` - Number of CPU threads used by the pipeline.

`device_id` - ID of GPU device used by pipeline.

Run the python file which serializes the Dali Pipeline and saves it to `model.dali`

```bash
python serialiize_dali_pipeline.py --config dali_config.json
```

**__Note__**:

- Makesure the serialized file should have the extension `.dali`
- The Torchserve batchsize should match the DALI batch size.

### Download the resnet .pth file

Navigate to `serve` directory and run the below commands

```bash
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

### Create model-archive file

The following command will create a .mar extension file where we also include the `model.dali` file and `dali_config.json` file in it.

```bash
torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./examples/image_classifier/resnet_18/model.py --serialized-file resnet18-f37072fd.pth --handler image_classifier --extra-files ./examples/image_classifier/index_to_name.json,./examples/nvidia_dali/model.dali,./examples/nvidia_dali/dali_config.json
```

Create a new directory `model_store` and move the model-archive file

```bash
mkdir model_store
mv resnet-18.mar model_store/
```

### Set environment for DALI

Run the following command in your terminal to set the environment variable for DALI_PREPROCESSING

```bash
export DALI_PREPROCESSING=true
```


### Start the torchserve

```bash
torchserve --start --model-store model_store --models resnet-18=resnet-18.mar
```

### Run Inference

Get the inference for a sample image using the below command

```bash
curl http://127.0.0.1:8080/predictions/resnet-18 -T ./examples/image_classifier/kitten.jpg
```
