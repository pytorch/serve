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
    torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
    ```
   
 * Step - 5: Register the model on TorchServe using the above model archive file and run digit recognition inference
   
    ```bash
    mkdir model_store
    mv mnist.mar model_store/
    torchserve --start --model-store model_store --models mnist=mnist.mar
    curl http://127.0.0.1:8080/predictions/mnist -T examples/image_classifier/mnist/test_data/0.png
    ```

For captum Explanations on the Torchserve side, use the below curl request:
```bash
curl http://127.0.0.1:8080/explanations/mnist -T examples/image_classifier/mnist/test_data/0.png
```

#Serve a custom model on Torchserve with KFServing API Spec for Inference and Captum Explanations:



KFServing makes use of Image Transformer - it converts bytes array into float tensor. So the input request should take the shape of a Bytes Array.


To serve the model in KFserving for Inference, follow the below steps :

* Step 1 : specify kfserving as the envelope in the config.properties file as below :

```bash
service_envelope=kfserving
```

* Step 2 : Create a .mar file by invoking the below command :

```bash
torch-model-archiver --model-name mnist --version 1.0 --model-file serve/examples/image_classifier/mnist/mnist.py --serialized-file serve/examples/image_classifier/mnist/mnist_cnn.pt --handler  serve/examples/image_classifier/mnist/mnist_handler.py
```

* Step 3 : Ensure that the docker image for Torchserve and the Image Transformer is created and accessible by the KFServing Environment. 
	     Refer the document for creating torchserve image with kfserving wrapper 

* Step 4 : Create an Inference Service in the Kubeflow, refer to the doc below to initiate the process:
[End to End Torchserve KFServing Model Serving](https://github.com/pytorch/serve/blob/master/kf_predictor_docker/README.md)

* Step 5 : Make the curl request as below for Inference:
```bash
 curl -H "Content-Type: application/json" --data @examples/image_classifier/mnist/mnist_kf.json http://127.0.0.1:8085/v1/models/mnist:predict
```

The Prediction response is as below :

```bash
{
	"predictions" : [

	2
		]
}
```

* Step 6 : Make the curl request as below for Explanations:
```bash
 curl -H "Content-Type: application/json" --data @examples/image_classifier/mnist/mnist_kf.json http://127.0.0.1:8085/v1/models/mnist:explain
```

The Explanation response is as below :

```bash
{
  "explanations": [
    [
      [
        [
          0.004570948731989492,
          0.006216969640322402,
          0.008197565423679522,
          0.009563574612830427,
          0.008999274832810742,
          0.009673474804303854,
          0.007599905146155397,
          ------,
	  ------

        ]
      ]
    ]
  ]
}
```


