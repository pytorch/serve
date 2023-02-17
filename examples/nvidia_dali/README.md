# DALI Optimization integration with Torchserve models

The NVIDIA Data Loading Library (DALI) is a library for data loading and pre-processing to accelerate deep learning applications. It provides a collection of highly optimized building blocks for loading and processing image, video and audio data.

In this example, we use NVIDIA DALI for pre-processing image input for inference in resnet-18 model.

Refer to [NVIDIA-DALI-Documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) for detailed information

### Install dependencies

Navigate to `serve/examples/nvidia_dali` directory and run the below command to install the dependencies

```bash
pip install -r requirements.txt
```

### Define and Build DALI Pipeline

In DALI, any data processing task has a central object called Pipeline.
Refer to [NVIDIA-DALI](https://github.com/NVIDIA/DALI) for more details on DALI pipeline.

Navigate to `cd ./serve/examples/nvidia_dali`.

Change the `dali_config.json` variables

`batch_size` - Maximum batch size of pipeline.

`num_threads` - Number of CPU threads used by the pipeline.

`device_id` - ID of GPU device used by pipeline.

Run the python file which serializes the Dali Pipeline and saves it to `model.dali`

```bash
python serialize_dali_pipeline.py --config dali_config.json
```

**__Note__**:

- Make sure that the serialized file has the extension `.dali`
- The Torchserve batch size should match the DALI batch size.

### Download the resnet .pth file

```bash
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

### Create model-archive file

The following command will create a .mar extension file where we also include the `model.dali` file and `dali_config.json` file in it.

```bash
torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ../image_classifier/resnet_18/model.py --serialized-file resnet18-f37072fd.pth --handler custom_handler.py --extra-files ../image_classifier/index_to_name.json,./model.dali,./dali_config.json
```

Navigate to `serve` directory and run the below commands

Create a new directory `model_store` and move the model-archive file

```bash
mkdir model_store
mv resnet-18.mar model_store/
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

```json
{
  "tabby": 0.408751517534256,
  "tiger_cat": 0.35404905676841736,
  "Egyptian_cat": 0.12418942898511887,
  "lynx": 0.025347290560603142,
  "bucket": 0.011393273249268532
}
```
