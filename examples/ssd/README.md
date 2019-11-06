# Single Shot Multi Object Detection Inference Service

In this example, we show how to use a pre-trained Single Shot Multi Object Detection (SSD) MXNet model for performing real time inference using MMS

The pre-trained model is trained on the [Pascal VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) The network is a SSD model built on Resnet50 as base network to extract image features. The model is trained to detect the following entities (classes): ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']. For more details about the model, you can refer [here](https://github.com/apache/incubator-mxnet/tree/master/example/ssd).

The inference service would return the response in the format - '[(object_class, xmin, ymin, xmax, ymax)]. Where, xmin, ymin, xmax and ymax are the bounding box coordinates of the detected object.

# Objective

1. Demonstrate how to package a a pre-trained MXNet model in MMS
2. Demonstrate how to create custom service with pre-processing and post-processing

## Step 1 - Download the pre-trained SSD Model

You will need the model files to use for the export. Check this example's directory in case they're already downloaded. Otherwise, you can `curl` the files or download them via your browser:

```bash
cd mxnet-model-server/examples/ssd

curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/resnet50_ssd_model-symbol.json
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/resnet50_ssd_model-0000.params
```

Alternatively, use these links to download the Symbol and Params files via your browser:
1. <a href="https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/resnet50_ssd_model-symbol.json" download>resnet50_ssd_model-symbol.json</a>
2. <a href="https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/resnet50_ssd_model-0000.params" download>resnet50_ssd_model-0000.params</a>

**Note** params file is around 125 MB.

## Step 2 - Prepare the signature file

Define model input name and shape in `signature.json` file. The signature for this example looks like below:

```json
{
  "inputs": [
    {
      "data_name": "data",
      "data_shape": [
        1,
        3,
        512,
        512
      ]
    }
  ]
}
```

In the pre-trained model, input name is 'data' and shape is '(1,3,512,512)'. Where, the expected input is a color image (3 channels - RGB) of shape 512*512. We also expect input type is a binary JPEG images. In provided mxnet_vision_service.py, you will see the code that take care of converting binary images to tensor NDArray used by MXNet.

*Note:* Typically, if you train your own model, you define the Input and Output Layer name and shape when defining the Neural Network. If you are using a pre-trained MXNet model, to get these Input and Output name and dimensions, you can load the Model and extract the Input and Output layer details. Unfortunately, there are no APIs or easy way to extract the Input shape. Example code below:

```python
>>> import mxnet as mx
>>> load_symbol, args, auxs = mx.model.load_checkpoint("resnet50_ssd_model", 000)
>>> mod = mx.mod.Module(load_symbol, label_names=None, context=mx.cpu())
>>> mod.data_names
['data']
>>> mod.bind(data_shapes=[('data', (1, 3, 512, 512))])
>>> mod.set_params(args, auxs)
>>> print(mod.data_names)
>>> print(mod.data_shapes)
>>> print(mod.output_names)
>>> print(mod.output_shapes)
['data']
[DataDesc[data,(1, 3, 512, 512),<class 'numpy.float32'>,NCHW]]
['detection_output']
[('detection_output', (1, 6132, 6))]
```

*Note:* The network generates 6132 detections because we use MXNet's [MultiboxPrior](https://mxnet.incubator.apache.org/api/python/symbol.html#mxnet.contrib.symbol.MultiBoxPrior) to generate the anchor boxes with the following 'Ratios and 'Sizes':

```python
    sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
    ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
```

To understand more about the MultiboxPrior, anchor boxes, sizes and ratios, please read [this tutorial](http://gluon.mxnet.io/chapter08_computer-vision/object-detection.html)

## Step 3 - Prepare synset.txt with list of class names

`synset.txt` is where we define list of all classes detected by the model. The pre-trained SSD model used in the example is trained to detect 20 classes - person, car, aeroplane, bicycle and more. See synset.txt file for list of all classes.

The list of classes in synset.txt will be loaded by MMS as list of labels in inference logic.

You can use `curl` to download it.
```bash
cd mxnet-model-server/examples/ssd

curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/synset.txt
```

Alternatively, use following link to download:
<a href="https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/synset.txt" download>synset.txt</a>


## Step 4 - Create custom service class

We provided custom service class template code in [template](../template) folder:
1. [model_handler.py](../model_service_template/model_handler.py) - A generic based service class.
2. [mxnet_model_service.py](../model_service_template/mxnet_model_service.py) - A MXNet base service class.
3. [mxnet_vision_service.py](../model_service_template/mxnet_vision_service.py) - A MXNet Vision service class.
4. [mxnet_utils](../model_service_template/mxnet_utils) - A python package that contains utility classes.

In this example, you can simple copy them into ssd folder, as use provided mxnet_vision_service.py as user model archive entry point.

```bash
cd mxnet-model-server/examples
cp -r model_service_template/* ssd/
```

In this example, we extend `MXNetVisionService`, provided by MMS for vision inference use-cases, and reuse its input image preprocess functionality to resize and transform the image shape. We only add custom pre-processing and post-processing steps. See [ssd_service.py](ssd_service.py) for more details on how to extend the base service and add custom pre-processing and post-processing.

## Step 5 - Package the model with `model-archiver` CLI utility

In this step, we package the following:
1. pre-trained MXNet Model we downloaded in Step 1.
2. '[signature.json](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/signature.json)' file we prepared in step 2.
3. '[synset.txt](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/ssd/synset.txt)' file we prepared in step 3.
4. custom model service files we prepared in step 4.

We use `model-archiver` command line utility (CLI) provided by MMS.
Install `model-archiver` in case you have not:

```bash
pip install model-archiver
```

This tool create a .mar file that will be provided to MMS for serving inference requests. In following command line, we specify 'ssd_service:handle' as model archive entry point.

```bash
cd mxnet-model-server/examples
model-archiver --model-name resnet50_ssd_model --model-path ssd --handler ssd_service:handle
```

## Step 6 - Start the Inference Service

Start the inference service by providing the 'resnet50_ssd_model.mar' file we created in Step 5.

MMS then extracts the resources (signature, synset, model symbol and params) we have packaged into .mar file and uses the extended custom service, to start the inference server.

By default, the server is started on the localhost at port 8080.

```bash
cd mxnet-model-server
mxnet-model-server --start --model-store examples --models ssd=resnet50_ssd_model.mar
```

Awesome! we have successfully exported a pre-trained MXNet model, extended MMS with custom preprocess/postprocess and started a inference service.

**Note**: In this example, MMS loads the .mar file from the local file system. However, you can also store the archive (.mar file) over a network-accessible storage such as AWS S3, and use a URL such as http:// or https:// to indicate the model archive location. MMS is capable of loading the model archive over such URLs as well.

## Step 7 - Test sample inference

Let us try the inference server we just started. Open another terminal on the same host. Download a sample image, or try any jpeg image that contains the one or more of the object classes mentioned earlier: 'aeroplane', 'bicycle', 'bird', 'boat', etc...

You can also use this image of three dogs on a beach.
![3 dogs on beach](../../docs/images/3dogs.jpg)

Use curl to make a prediction call by passing the downloaded image as input to the prediction request.

```bash
cd mxnet-model-server
curl -X POST http://127.0.0.1:8080/predictions/ssd -T docs/images/3dogs.jpg
```

You can expect the response similar to below. The output format is `[(object_class, xmin, ymin, xmax, ymax)]`.
Where, xmin, ymin, xmax and ymax are the bounding box coordinates of the detected object.

```json
[
  [
    "dog", 
    399, 
    128, 
    570, 
    290
  ], 
  [
    "dog", 
    278, 
    196, 
    417, 
    286
  ], 
  [
    "cow", 
    205, 
    116, 
    297, 
    272
  ]
]
```

A consumer application can use this response to identify the objects in the input image and their bounding boxes.

For better visualization on the input and how we can use the inference output, see below:

Input Image

![Street Input Image](../../docs/images/dogs-before.jpg)

Output Image

![Street Output Image](../../docs/images/dogs-after.jpg)


See [More example outputs](example_outputs.md)

# References
1. Adapted code and pre-trained model from - https://github.com/apache/incubator-mxnet/tree/master/example/ssd
2. Learn more about SSD in this tutorial - http://gluon.mxnet.io/chapter08_computer-vision/object-detection.html
