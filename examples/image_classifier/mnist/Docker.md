# Digit recognition model with MNIST dataset using Docker container

In this example, we show how to use a pre-trained custom MNIST model to performing real time Digit recognition with TorchServe.
We will be serving the model using a Docker container.

The inference service would return the digit inferred by the model in the input image.

We used the following pytorch example to train the basic MNIST model for digit recognition :
https://github.com/pytorch/examples/tree/master/mnist

## Serve an MNIST model on TorchServe docker container

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

 ### Create a torch model archive using the torch-model-archiver utility to archive the above files.

    ```bash
    torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
    ```

  ### Move .mar file into model_store directory

    ```bash
    mkdir model_store
    mv mnist.mar model_store/
    ```

  ### Start a docker container with torchserve

  ```bash
  docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -v $(pwd)/model_store:/home/model-server/model-store pytorch/torchserve:latest-cpu
  ```

  ### Register the model on TorchServe using the above model archive file

  ```bash
  curl -X POST "localhost:8081/models?model_name=mnist&url=mnist.mar&initial_workers=4"
  ```

  If this succeeeds, you will see a message like below

  ```bash
  {
  "status": "Model \"mnist\" Version: 1.0 registered with 4 initial workers"
  }
  ```

  ### Run digit recognition inference outside the container

    ```bash
    curl http://127.0.0.1:8080/predictions/mnist -T examples/image_classifier/mnist/test_data/0.png
    ```

   The output in this case will be a `0`
