# Digit recognition model with MNIST dataset

In this example, we show how to use a pre-trained custom MNIST model to performing real time Digit recognition with TorchServe and TorchData.

The inference service would return the digit inferred by the model in the input image.

We used the following pytorch example to train the basic MNIST model for digit recognition :
https://github.com/pytorch/examples/tree/master/mnist

# Objective
1. Demonstrate how to use torchdata and torchserve together.

# Prerequisites
Install aiohttp package
```bash
pip install aiohttp
```

Download MNIST dataset
```bash
cd examples/image_classifier/mnist/torchdata
mkdir mnist_dataset

wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -P ./mnist_dataset
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -P ./mnist_dataset
```

# Serve a custom model on TorchServe

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path

 * Step - 1: Create a torch model archive using the torch-model-archiver utility to archive the above files.

    ```bash
    torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/torchdata/mnist_handler.py
    ```

 * Step - 2: Register the model on TorchServe using the above model archive file and run digit recognition inference

    ```bash
    mkdir model_store
    mv mnist.mar model_store/
    torchserve --start --model-store model_store --models mnist=mnist.mar --ts-config config.properties
    curl http://127.0.0.1:8080/predictions/mnist -T examples/image_classifier/mnist/test_data/0.png
    ```

* Step - 3: Run `inference.py` script that loads MNIST dataset using torchdata and sends REST API requests to torchserve for inference.
    ```bash
    python inference.py
    ```

### Running KServe

Refer the [MNIST Readme for KServe](https://github.com/pytorch/serve/blob/master/kubernetes/kserve/kserve_wrapper/README.md) to run it locally.

Refer the [End to End KServe document](https://github.com/pytorch/serve/blob/master/kubernetes/kserve/README.md) to run it in the cluster.
