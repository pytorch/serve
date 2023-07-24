# DALI Optimization integration with Torchserve models

The NVIDIA Data Loading Library (DALI) is a library for data loading and pre-processing to accelerate deep learning applications. It provides a collection of highly optimized building blocks for loading and processing image, video and audio data.

In this example, we use NVIDIA DALI for pre-processing image input for inference in mnist model.

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

Change the `model-config.yaml` variables

`batch_size` - Maximum batch size of pipeline.

`num_threads` - Number of CPU threads used by the pipeline.

`device_id` - ID of GPU device used by pipeline.

`pipeline_filename` - Pipeline filename

Run the python file which serializes the Dali Pipeline and saves it to `model.dali`

```bash
python serialize_dali_pipeline.py --config model-config.yaml
```

**__Note__**:

- Make sure that the serialized file has the extension `.dali`
- The Torchserve batch size should match the DALI batch size.

### Create model-archive file

The following command will create a .mar extension file where we also include the `model.dali` file in it.

```bash
torch-model-archiver --model-name  mnist --version 1.0 --model-file ../image_classifier/mnist/mnist.py --serialized-file ../image_classifier/mnist/mnist_cnn.pt --handler custom_handler.py --extra-files ./model.dali --config-file model-config.yaml
```

Navigate to `serve` directory and run the below commands

Create a new directory `model_store` and move the model-archive file

```bash
mkdir model_store
mv mnist.mar model_store/
```

### Start the torchserve

```bash
torchserve --start --model-store model_store --models mnist=mnist.mar
```

### Run Inference

Get the inference for a sample image using the below command

```bash
curl http://127.0.0.1:8080/predictions/mnist -T ./examples/image_classifier/mnist/test_data/0.png
```
