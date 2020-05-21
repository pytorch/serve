# Digit recognition model with MNIST dataset

In this example, we show how to use a pre-trained custom MNIST model to performing real time Digit recognition with TorchServe.

The inference service would return the digit inferred by the model in the input image.

We used the following pytorch example to train the basic MNIST model for digit recognition :
https://github.com/pytorch/examples/tree/master/mnist

# Objective
1. Demonstrate how to package a custom trained model with custom handler into torch model archive (.mar) file
2. Demonstrate how to create model handler code
3. Demonstrate how to load model archive (.mar) file into TorchServe and run inference.

# Serve a custom model on TorchServe

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path

 * Step - 1: Create a new model architecture file which contains model class extended from torch.nn.modules. In this example we have created [mnist model file](mnist.py).
 * Step - 2: Train a MNIST digit recognition model using https://github.com/pytorch/examples/blob/master/mnist/main.py and save the state dict of model. We have added the pre-created [state dict](mnist_cnn.pt) of this model.
 * Step - 3: Write a custom handler to run the inference on your model. In this example, we have added a [custom_handler](mnist_handler.py) which runs the inference on the input greyscale images using the above model and recognizes the digit in the image.
 * Step - 4: Create a torch model archive using the torch-model-archiver utility to archive the above files.
 
    ```bash
    torch-model-archiver --model-name mnist --version 1.0 --model-file ./serve/examples/image_classifier/mnist/mnist.py --serialized-file ./serve/examples/image_classifier/mnist/mnist_cnn.pt --handler  ./serve/examples/image_classifier/mnist/mnist_handler.py
    ```
   
 * Step - 5: Register the model on TorchServe using the above model archive file and run digit recognition inference
   
    ```bash
    mkdir model_store
    mv mnist.mar model_store/
    torchserve --start --model-store model_store --models mnist=mnist.mar
    curl http://127.0.0.1:8080/predictions/mnist -T ./serve/examples/image_classifier/mnist/test_data/0.png
    ```

