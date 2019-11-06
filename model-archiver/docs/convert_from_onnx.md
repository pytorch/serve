# Converting an ONNX Model

## Install model-archiver with ONNX support
ONNX support is optional in `model-archiver` tool. It's not installed by default with `model-archiver`.

To install MMS with ONNX support, you will need to have the [protobuf compiler](https://github.com/onnx/onnx#installation) installed:

for Ubuntu run:

```bash
sudo apt-get install protobuf-compiler libprotoc-dev

pip install model-archiver[onnx]
```

Or for Mac run:

```bash
conda install -c conda-forge protobuf numpy

pip install model-archiver[onnx]
```

MXNet is also required for conversion. You can choose different flavor is mxnet:

```bash
pip install mxnet

or

pip install mxnet-mkl

or

pip install mxnet-cu90mkl
```

## ONNX model archive example

You can download a model from the [ONNX Model Zoo](https://github.com/onnx/models) then use `model-archiver` to covert it to a `.mar` file.

**Note**: Some ONNX model authors upload their models to the zoo in the `.pb` or `.pb2` format. Just change the extension to `.onnx` before attempting to convert.

Let's use the SqueezeNet ONNX model as an example. 

### Prepare ONNX model and labels

To create a model archive for MMS, you can get `.onnx` file and optionally a labels file (synset.txt) from our S3:

* [SqueezeNet ONNX model](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-squeezenet/squeezenet.onnx): a `.onnx` model file from the [ONNX Model Zoo](https://github.com/onnx/models)
* [label file](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-squeezenet/synset.txt): has the labels for 1,000 ImageNet classes

```bash
cd mxnet-model-server/examples
mkdir onnx-squeezenet
cd onnx-squeezenet

curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-squeezenet/squeezenet.onnx
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-squeezenet/synset.txt
```

###  Prepare your model custom service code

You can implement your own model customer service code as model archive entry point. In this example we just copy provided mxnet vision service template:

```bash
cd mxnet-model-server/examples

cp -r model_service_template/* onnx-squeezenet/
```

The mxnet_vision_service.py assume there is a signature.json file that describes input parameter name and shape. You can download example from: [signature file](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-squeezenet/signature.json).


```bash
cd mxnet-model-server/examples/onnx-squeezenet

curl -o signature.json https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-squeezenet/signature.json
```

### Create a `.mar` file from onnx model

Since the model has the `.onnx` extension, it will be detected and the converted to mxnet models accordingly.

Now you can use the `model-archiver` command to output `onnx-squeezenet.mar` file.

```bash
cd mxnet-model-server/examples

model-archiver --model-name onnx-squeezenet --model-path onnx-squeezenet --handler mxnet_vision_service:handle
```

Now start the server:

```bash
cd mxnet-model-server

mxnet-model-server --start --model-store examples --models squeezenet=onnx-squeezenet.mar
```

After your server starts, you can use the following command to see the prediction results.

```bash
curl -X POST http://127.0.0.1:8080/predictions/squeezenet -T docs/images/kitten_small.jpg
```
