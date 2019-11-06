# MXNet Vision Service

In this example, we show how to use a pre-trained MXNet model to performing real time Image Classification with MMS

We choose squeezenet in this example: [Iandola, et al.](https://arxiv.org/pdf/1602.07360v4.pdf). But the same should work for other MXNet Image Classification models.

The inference service would return the response in the json format.

# Objective

1. Demonstrate how to package a pre-trained squeezenet into model archive (.mar) file
2. Demonstrate how to create model service code based on provided service template
3. Demonstrate how to load model archive (.mar) file into MMS and run inference.

## Step 1 - Download the pre-trained squeezenet Model

You will need the model files in this example. Check this example's directory in case they're already downloaded. Otherwise, you can `curl` the files or download them via your browser:

```bash
cd mxnet-model-server/examples/mxnet_vision
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-symbol.json
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-0000.params
```

Alternatively, use these links to download the Symbol and Params files via your browser:
1. <a href="https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-symbol.json" download>squeezenet_v1.1-symbol.json</a>
2. <a href="https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/squeezenet_v1.1-0000.params" download>squeezenet_v1.1-0000.params</a>

## Step 2 - Prepare the signature file

Define Input and Output name, type and shape in `signature.json` file. The signature for this example looks like below:

```json
{
  "inputs": [
    {
      "data_name": "data",
      "data_shape": [
        0,
        3,
        224,
        224
      ]
    }
  ]
}
```

In this pre-trained model, input name is 'data' and shape is '(1,3,224,224)'. Where, the expected input is a color image (3 channels - RGB) of shape 224*224. We also expect input type is a binary JPEG images. In provided mxnet_vision_service.py, you will see the code that take care of converting binary images to tensor NDArray used by MXNet.

*Note:* Typically, if you train your own model, you define the Input and Output Layer name and shape when defining the Neural Network. If you are using a pre-trained MXNet model, to get these Input and Output name and dimensions, you can load the Model and extract the Input and Output layer details. Unfortunately, there are no APIs or easy way to extract the Input shape. Example code below:

```python
>>> import mxnet as mx
>>> load_symbol, args, auxs = mx.model.load_checkpoint("squeezenet_v1.1", 0)
>>> mod = mx.mod.Module(load_symbol, label_names=None, data_names=['data'], context=mx.cpu())
>>> mod.data_names
['data']
>>> mod.bind(data_shapes=[('data', (1, 3, 224, 224))])
>>> mod.set_params(args, auxs)
>>> print(mod.data_names)
>>> print(mod.data_shapes)
>>> print(mod.output_names)
>>> print(mod.output_shapes)
['data']
[DataDesc[data,(1, 3, 224, 224),<class 'numpy.float32'>,NCHW]]
['detection_output']
[('detection_output', (1, 6132, 6))]
```

## Step 3 - Prepare synset.txt with list of class names

[synset.txt](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt) is where we define list of all classes detected by the model. The list of classes in synset.txt will be loaded by MMS as list of labels in inference logic.

You can use `curl` to download it.
```bash
cd mxnet-model-server/examples/mxnet_vision

curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt
```

Alternatively, use following link to download:
<a href="https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt" download>synset.txt</a>

## Step 4 - Create custom service class

We provided custom service class template code in [model_service_template](../model_service_template) folder:
1. [model_handler.py](../model_service_template/model_handler.py) - A generic based service class.
2. [mxnet_model_service.py](../model_service_template/mxnet_model_service.py) - A MXNet base service class.
3. [mxnet_vision_service.py](../model_service_template/mxnet_vision_service.py) - A MXNet Vision service class.
4. [mxnet_utils](../model_service_template/mxnet_utils) - A python package that contains utility classes.

In this example, you can simple copy them into mxnet_vision folder, as use provided mxnet_vision_service.py as user model archive entry point.

```bash
cd mxnet-model-server/examples
cp -r model_service_template/* mxnet_vision/
```

## Step 5 - Package the model with `model-archiver` CLI utility

In this step, we package the following:
1. pre-trained MXNet Model we downloaded in Step 1.
2. '[signature.json](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/signature.json)' file we prepared in step 2.
3. '[synset.txt](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/squeezenet_v1.1/synset.txt)' file we prepared in step 3.
4. custom model service files we prepared in step 4.

We use `model-archiver` command line utility (CLI) provided by MMS.
Install `model-archiver` in case you have not:

```bash
pip install model-archiver
```

This tool create a .mar file that will be provided to MMS for serving inference requests. In following command line, we specify 'mxnet_model_service:handle' as model archive entry point.

```bash
cd mxnet-model-server/examples
model-archiver --model-name squeezenet_v1.1 --model-path mxnet_vision --handler mxnet_vision_service:handle
```

## Step 6 - Start the Inference Service

Start the inference service by providing the 'squeezenet_v1.1.mar' file we created in Step 5.

By default, the server is started on the localhost at port 8080.

```bash
cd mxnet-model-server
mxnet-model-server --start --model-store examples --models squeezenet_v1.1.mar
```

Awesome! we have successfully packaged a pre-trained MXNet model and started a inference service.

`Note:` In this example, MMS loads the .mar file from the local file system. However, you can also store the model archive (.mar file) over a network-accessible storage such as AWS S3, and use a URL such as http:// or https:// to indicate the model location. MMS is capable of loading the model archive over such URLs as well.

## Step 7 - Test sample inference

Let us try the inference server we just started. Use curl to make a prediction call by passing a JPEG image as input to the prediction request.

```bash
cd mxnet-model-server
curl -X POST http://127.0.0.1:8080/predictions/squeezenet_v1.1 -T docs/images/kitten_small.jpg
```

You can expect the response similar to below. The output format is in json.

```json
[
  {
    "class": "n02127052 lynx, catamount", 
    "probability": 0.5721369385719299
  }, 
  {
    "class": "n02124075 Egyptian cat", 
    "probability": 0.4079437255859375
  }, 
  {
    "class": "n02123045 tabby, tabby cat", 
    "probability": 0.013694713823497295
  }, 
  {
    "class": "n02123394 Persian cat", 
    "probability": 0.004954110365360975
  }, 
  {
    "class": "n02123159 tiger cat", 
    "probability": 0.0012674571480602026
  }
]
```

A consumer application can use this response to identify the objects in the input image and their bounding boxes.

## Step 8 - Clean up and stop MMS

MMS will keep running in background. And .mar file will be extracted to system temp directory.
You can clean up temp directory by unregister model and use CLI to stop MMS

```bash
curl -X DELETE http://127.0.0.1:8081/models/squeezenet_v1.1

mxnet-model-server --stop
```
