# Loading and serving Gluon models on MXNet Model Server (MMS)
MXNet Model Server (MMS) supports loading and serving MXNet Imperative and Hybrid Gluon models.
This is a short tutorial on how to write a custom Gluon model, and then serve it with MMS.

This tutorial covers the following:
1. [Prerequisites](#prerequisites)
2. [Serve a Gluon model](#load-and-serve-a-gluon-model)
  * [Load and serve a pre-trained Gluon model](#load-and-serve-a-pre-trained-gluon-model)
  * [Load and serve a custom Gluon model](#load-and-serve-a-custom-gluon-imperative-model)
  * [Load and serve a custom hybrid Gluon model](#load-and-serve-a-hybrid-gluon-model)
3. [Conclusion](#conclusion)

## Prerequisites
* **Basic Gluon knowledge**. If you are using Gluon for the first
time, but are familiar with creating a neural network with MXNet or another framework, you may refer this 10 min Gluon crash-course: [Predict with a pre-trained model](http://gluon-crash-course.mxnet.io/predict.html).
* **Gluon naming**. Fine-tuning pre-trained Gluon models requires some understanding of how the naming conventions work. Take a look at the [Naming of Gluon Parameter and Blocks](https://mxnet.incubator.apache.org/tutorials/gluon/naming.html) tutorial for more information.
* **Basic MMS knowledge**. If you are using MMS for the first time, you should take advantage of the [MMS QuickStart tutorial](https://github.com/awslabs/mxnet-model-server#quick-start).
* **MMS installed**. If you haven't already, [install MMS with pip](https://github.com/awslabs/mxnet-model-server/blob/master/docs/install.md#install-mms-with-pip) or [install MMS from source](https://github.com/awslabs/mxnet-model-server/blob/master/docs/install.md#install-mms-from-source-code). Either installation will also install MXNet.

Refer to the [MXNet model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html) documentation for examples of accessing other models.

## Load and serve a Gluon model
There are three scenarios for serving a Gluon model with MMS:

1. Load and serve a pre-trained Gluon model.
2. Load and serve a custom imperative Gluon model.
3. Load and serve a custom hybrid Gluon model.

To learn more about the differences between gluon and hybrid gluon models refer to [the following document](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html)

### Load and serve a pre-trained Gluon model
Loading and serving a pre-trained Gluon model is the simplest of the three scenarios. These models don't require you to provide `symbols` and `params` files.

It is easy to access a model with a couple of lines of code. The following code snippet shows how to load and serve a pretrained Gluon model.

```python
class PretrainedAlexnetService(GluonBaseService):
    """
    Pretrained alexnet Service
    """
    def initialize(self, params):
        self.net = mxnet.gluon.model_zoo.vision.alexnet(pretrained=True)
        self.param_filename = "alexnet.params"
        super(PretrainedAlexnetService, self).initialize(params)

    def postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [[{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability':
                float(data[0, int(i.asscalar())].asscalar())} for i in idx]]


svc = PretrainedAlexnetService()


def pretrained_gluon_alexnet(data, context):
    res = None
    if not svc.initialized:
        svc.initialize(context)

    if data is not None:
        res = svc.predict(data)

    return res
```

For an actual code implementation, refer to the custom-service code which uses the [pre-trained Alexnet](https://github.com/awslabs/mxnet-model-server/blob/master/examples/gluon_alexnet/gluon_pretrained_alexnet.py)

### Serve pre-trained model with MMS
To serve pre-trained models with MMS we would need to create an model archive file. Follow the below steps to get the example custom service
loaded and served with MMS.

1. Create a `models` directory
```bash
mkdir /tmp/models
```
2. Copy the [example code](https://github.com/awslabs/mxnet-model-server/blob/master/examples/gluon_alexnet/gluon_pretrained_alexnet.py)
and other required artifacts to this folder
```bash
cp ../model_service_template/gluon_base_service.py ../model_service_template/mxnet_utils/ndarray.py  gluon_pretrained_alexnet.py synset.txt signature.json /tmp/models/.
```
3. Run the model-export tool on this folder.
```bash
model-archiver --model-name alexnet --model-path /tmp/models --handler gluon_pretrained_alexnet:pretrained_gluon_alexnet --runtime python --export-path /tmp
```
This creates a model-archive file `/tmp/alexnet.mar`.

4. You could run the server with this model file to serve the pre-trained alexnet.
```bash
mxnet-model-server --start --models alexnet.mar --model-store /tmp
```
5. Test your service
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/alexnet/predict -F "data=@kitten.jpg"
```

## Load and serve a custom Gluon imperative model
To load an imperative model for use with MMS, you must activate the network in a MMS custom service. Once activated, MMS
can load the pre-trained parameters and start serving the imperative model. You also need to handle pre-processing and
post-processing of the image input.

We created a custom imperative model using Gluon. Refer to
[custom service code](https://github.com/awslabs/mxnet-model-server/examples/gluon_alexnet/examples/gluon_alexnet/gluon_alexnet.py)
The network definition, which is defined in the example, is as follows

```python
class GluonImperativeAlexNet(gluon.Block):
    """
    Fully imperative gluon Alexnet model
    """
    def __init__(self, classes=1000, **kwargs):
        super(GluonImperativeAlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.Sequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
            self.output = nn.Dense(classes)

    def forward(self, x):
	x = self.features(x)
        x = self.output(x)
        return x
```

The pre-process, inference and post-process steps are similar to the service code that we saw in the [above section](#load-and-serve-a-pre-trained-gluon-model).
```python
class ImperativeAlexnetService(GluonBaseService):
    """
    Gluon alexnet Service
    """

    def initialize(self, params):
        self.net = GluonImperativeAlexNet()
        self.param_filename = "alexnet.params"
        super(ImperativeAlexnetService, self).initialize(params)

    def postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [[{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability':
                 float(data[0, int(i.asscalar())].asscalar())} for i in idx]]


svc = ImperativeAlexnetService()


def imperative_gluon_alexnet_inf(data, context):
    res = None
    if not svc.initialized:
        svc.initialize(context)

    if data is not None:
        res = svc.predict(data)

    return res
```

### Test your imperative Gluon model service
To serve imperative Gluon models with MMS we would need to create an model archive file.
Follow the below steps to get the example custom service loaded and served with MMS.

1. Create a `models` directory
```bash
mkdir /tmp/models
```
2. Copy the [example code](https://github.com/awslabs/mxnet-model-server/examples/gluon_alexnet/gluon_imperative_alexnet.py)
and other required artifacts to this folder
```bash
cp ../model_service_template/gluon_base_service.py ../model_service_template/mxnet_utils/ndarray.py  gluon_imperative_alexnet.py synset.txt signature.json /tmp/models/.
```
3. Download/copy the parameters to this `/tmp/models` directory. For this example, we have the parameters file in a S3 bucket.
```bash
wget https://s3.amazonaws.com/gluon-mms-model-files/alexnet.params
mv alexnet.params /tmp/models
```
4. Run the model-export tool on this folder.
```bash
model-archiver --model-name alexnet --model-path /tmp/models --handler gluon_imperative_alexnet:imperative_gluon_alexnet_inf --runtime python --export-path /tmp
```
This creates a model-archive file `/tmp/alexnet.mar`.

5. You could run the server with this model file to serve the pre-trained alexnet.
```bash
mxnet-model-server --start --models alexnet.mar --model-store /tmp
```
6. Test your service
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/alexnet/predict -F "data=@kitten.jpg"
```

The output should be close to the following:

```json
{"prediction":[{"class":"lynx,","probability":0.9411474466323853},{"class":"leopard,","probability":0.016749195754528046},{"class":"tabby,","probability":0.012754007242619991},{"class":"Egyptian","probability":0.011728651821613312},{"class":"tiger","probability":0.008974711410701275}]}
```

## Load and serve a hybrid Gluon model
To serve hybrid Gluon models with MMS, let's consider `gluon_imperative_alexnet.py` in `mxnet-model-server/examples/gluon_alexnet` folder.
We first convert the model to a `Gluon` hybrid block.
For additional background on using `HybridBlocks` and the need to `hybridize` refer to this Gluon [hybridize](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html#) tutorial.

The above network, after this conversion, would look as follows:
```python
class GluonHybridAlexNet(HybridBlock):
    """
    Hybrid Block gluon model
    """
    def __init__(self, classes=1000, **kwargs):
        super(GluonHybridAlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x
```
We could use the same custom service code as in the above section,

```python
class HybridAlexnetService(GluonBaseService):
    """
    Gluon alexnet Service
    """
    def initialize(self, params):
        self.net = GluonHybridAlexNet()
        self.param_filename = "alexnet.params"
        super(HybridAlexnetService, self).initialize(params)
        self.net.hybridize()

    def postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [[{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability':
                 float(data[0, int(i.asscalar())].asscalar())} for i in idx]]


svc = HybridAlexnetService()


def hybrid_gluon_alexnet_inf(data, context):
    res = None
    if not svc.initialized:
        svc.initialize(context)

    if data is not None:
        res = svc.predict(data)

    return res
```
Similar to imperative models, this model doesn't require `Symbols` as the call to `.hybridize()` compiles the neural net.
This would store the `symbols` implicitly.

### Test your hybrid Gluon model service
To serve Hybrid Gluon models with MMS we would need to create an model archive file.
Follow the below steps to get the example custom service loaded and served with MMS.

1. Create a `models` directory
```bash
mkdir /tmp/models
```
2. Copy the [example code](https://github.com/awslabs/mxnet-model-server/examples/gluon_alexnet/gluon_imperative_alexnet.py)
and other required artifacts to this folder
```bash
cp ../model_service_template/gluon_base_service.py ../model_service_template/mxnet_utils/ndarray.py  gluon_hybrid_alexnet.py synset.txt signature.json /tmp/models/.
```
3. Download/copy the parameters to this `/tmp/models` directory. For this example, we have the parameters file in a S3 bucket.
```bash
wget https://s3.amazonaws.com/gluon-mms-model-files/alexnet.params
mv alexnet.params /tmp/models
```
4. Run the model-export tool on this folder.
```bash
model-archiver --model-name alexnet --model-path /tmp/models --handler gluon_hybrid_alexnet:hybrid_gluon_alexnet_inf --runtime python --export-path /tmp
```
This creates a model-archive file `/tmp/alexnet.mar`.

5. You could run the server with this model file to serve the pre-trained alexnet.
```bash
mxnet-model-server --start --models alexnet.mar --model-store /tmp
```
6. Test your service
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/alexnet/predict -F "data=@kitten.jpg"
```

The output should be close to the following:

```json
{"prediction":[{"class":"lynx,","probability":0.9411474466323853},{"class":"leopard,","probability":0.016749195754528046},{"class":"tabby,","probability":0.012754007242619991},{"class":"Egyptian","probability":0.011728651821613312},{"class":"tiger","probability":0.008974711410701275}]}
```

## Conclusion
In this tutorial you learned how to serve Gluon models in three unique scenarios: a pre-trained imperative model directly from the model zoo, a custom imperative model, and a hybrid model. For further examples of customizing gluon models, try the Gluon tutorial for [Transferring knowledge through fine-tuning](http://gluon.mxnet.io/chapter08_computer-vision/fine-tuning.html). For an advanced custom service example, try the MMS [SSD example](https://github.com/awslabs/mxnet-model-server/tree/master/examples/ssd).
