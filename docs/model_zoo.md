# Model Zoo

This page lists model archives that are pre-trained and pre-packaged, ready to be served for inference with MMS.
To propose a model for inclusion, please submit a [pull request](https://github.com/awslabs/mxnet-model-server/pulls).

*Special thanks to the [Apache MXNet](https://mxnet.incubator.apache.org) community whose Model Zoo and Model Examples were used in generating these model archives.*


| Model File | Type | Dataset | Source | Size | Download |
| --- | --- | --- | --- | --- | --- |
| [AlexNet](#alexnet) | Image Classification | ImageNet | ONNX | 233 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/alexnet.mar) |
| [ArcFace-ResNet100](#arcface-resnet100_onnx) | Face Recognition | Refined MS-Celeb1M | ONNX | 236.4 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-arcface-resnet100.mar) |
| [Character-level Convolutional Networks for Text Classification](#crepe) | Text Classification | [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/) | Gluon | 40 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/crepe.mar) |
| [CaffeNet](#caffenet) | Image Classification | ImageNet | MXNet | 216 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/caffenet/caffenet.mar) |
| [FERPlus](#ferplus_onnx) | Emotion Detection | FER2013 | ONNX | 35MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/FERPlus.mar) |
| [Inception v1](#inception_v1) | Image Classification | ImageNet | ONNX | 27 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-inception_v1.mar) |
| [Inception v3 w/BatchNorm](#inception_v3) | Image Classification | ImageNet | MXNet | 45 MB |  [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/inception-bn.mar) |
| [LSTM PTB](#lstm-ptb) | Language Modeling | PennTreeBank | MXNet | 16 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/lstm_ptb.mar) |
| [MobileNetv2-1.0](#mobilenetv2-1.0_onnx) | Image Classification | ImageNet | ONNX | 13.7 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-mobilenet.mar) |
| [Network in Network (NiN)](#nin) | Image Classification | ImageNet | MXNet | 30 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/nin.mar) |
| [ResNet-152](#resnet-152) | Image Classification | ImageNet | MXNet | 241 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-152.mar) |
| [ResNet-18](#resnet-18) | Image Classification | ImageNet | MXNet | 43 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-18.mar) |
| [ResNet50-SSD](#resnet50-ssd) | SSD (Single Shot MultiBox Detector) | ImageNet | MXNet | 124 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/resnet50_ssd.mar) |
| [ResNext101-64x4d](#resnext101) | Image Classification | ImageNet | MXNet | 334 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/resnext-101-64x4d.mar) |
| [ResNet-18v1](#resnet-18v1) | Image Classification | ImageNet | ONNX | 45 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet18v1.mar) |
| [ResNet-34v1](#resnet-34v1) | Image Classification | ImageNet | ONNX | 83 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet34v1.mar) |
| [ResNet-50v1](#resnet-50v1) | Image Classification | ImageNet | ONNX | 98 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet50v1.mar) |
| [ResNet-101v1](#resnet-101v1) | Image Classification | ImageNet | ONNX | 171 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet101v1.mar) |
| [ResNet-152v1](#resnet-152v1) | Image Classification | ImageNet | ONNX | 231 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet152v1.mar) |
| [ResNet-18v2](#resnet-18v2) | Image Classification | ImageNet | ONNX | 45 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet18v2.mar) |
| [ResNet-34v2](#resnet-34v2) | Image Classification | ImageNet | ONNX | 83 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet34v2.mar) |
| [ResNet-50v2](#resnet-50v2) | Image Classification | ImageNet | ONNX | 98 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet50v2.mar) |
| [ResNet-101v2](#resnet-101v2) | Image Classification | ImageNet | ONNX | 171 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet101v2.mar) |
| [ResNet-152v2](#resnet-152v2) | Image Classification | ImageNet | ONNX | 231 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet152v2.mar) |
| [Shufflenet](#shufflenet) | Image Classification | ImageNet | ONNX | 8.1   MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/shufflenet.mar) |
| [SqueezeNet_v1.1](#squeezenet_v1.1_onnx) | Image Classification | ImageNet | ONNX | 5 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-squeezenet.mar) |
| [SqueezeNet v1.1](#squeezenet_v1.1) | Image Classification | ImageNet | MXNet | 5 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar) |
| [VGG16](#vgg16) | Image Classification | ImageNet | MXNet | 490 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/vgg16.mar) |
| [VGG16](#vgg16_onnx) | Image Classification | ImageNet | ONNX | 527 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-vgg16.mar) |
| [VGG16_bn](#vgg16_bn_onnx) | Image Classification | ImageNet | ONNX | 527 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-vgg16_bn.mar) |
| [VGG19](#vgg19) | Image Classification | ImageNet | MXNet | 509 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/vgg19.mar) |
| [VGG19](#vgg19_onnx) | Image Classification | ImageNet | ONNX | 548 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-vgg19.mar) |
| [VGG19_bn](#vgg19_bn_onnx) | Image Classification | ImageNet | ONNX | 548 MB | [.mar](https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-vgg19_bn.mar) |


## Details on Each Model
Each model below comes with a basic description, and where available, a link to a scholarly article about the model.

Many of these models use a kitten image to test inference. Use the following to get one that will work:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
```


## <a name="alexnet"></a>AlexNet
* **Type**: Image classification trained on ImageNet

* **Reference**: [Krizhevsky, et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models alexnet=https://s3.amazonaws.com/model-server/model_archive_1.0/alexnet.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/alexnet -T kitten.jpg
```

## <a name="arcface-resnet100_onnx"></a>ArcFace-ResNet100 (from ONNX model zoo)
* **Type**: Face Recognition model trained on refined MS-Celeb1M dataset (model imported from ONNX)

* **Reference**: [Deng et al.](https://arxiv.org/abs/1801.07698)

* **Model Service**:
    * [arcface_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-arcface-resnet100/arcface_service.py)
    * [mtcnn_detector.py](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-arcface-resnet100/mtcnn_detector.py)

* **Install dependencies**:
```bash
pip install opencv-python
pip install scikit-learn
pip install easydict
pip install scikit-image
pip install numpy
```

* **Start Server**:
```bash
mxnet-model-server --start --models arcface=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-arcface-resnet100.mar
```

* **Get two test images**:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/arcface-input1.jpg

curl -O https://s3.amazonaws.com/model-server/inputs/arcface-input2.jpg
```


* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/arcface -F "img1=@arcface-input1.jpg" -F "img2=@arcface-input2.jpg"
```

## <a name="caffenet"></a>CaffeNet
* **Type**: Image classification trained on ImageNet

* **Reference**: [Krizhevsky, et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models caffenet=https://s3.amazonaws.com/model-server/model_archive_1.0/caffenet.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/caffenet -T kitten.jpg
```

## <a name="crepe"></a>Character-level Convolutional Networks for text Classification
* **Type**: Character-level Convolutional network for text classification trained on [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/).

* **Reference**: [R. He, J. McAuley et al.](https://arxiv.org/abs/1602.01585), [J. McAuley, C. Targett, J. Shi, A. van den Hengel et al.](https://arxiv.org/abs/1506.04757) 

* **Model Service**: [gluon_crepe.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/gluon_character_cnn/gluon_crepe.py)

* **Start Server**:
```bash
mxnet-model-server --start --models crepe=https://s3.amazonaws.com/model-server/model_archive_1.0/crepe.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/crepe -F "data=[{\"review_title\":\"Inception is the best\",\"review\": \"great direction and story\"}]"
```

## <a name="duc-resnet101_onnx"></a>DUC-ResNet101 (from ONNX model zoo)
* **Type**: Semantic Segmentation model trained on the Cityscapes dataset (model imported from ONNX)

* **Reference**: [Wang et al.](https://arxiv.org/abs/1702.08502)

* **Model Service**:
    * [duc_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-duc/duc_service.py)
    * [cityscapes_labels.py](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/onnx-duc/cityscapes_labels.py)

* **Install dependencies**:
```bash
pip install opencv-python
pip install pillow
```

* **Start Server**:
```bash
mxnet-model-server --models duc=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-duc.mar
```

* **Get the test image**:
```bash
curl -O https://s3.amazonaws.com/mxnet-model-server/onnx-duc/city1.jpg
```

* **Download inference script**:

The script makes an inference call to the server using the test image, displays the colorized segmentation map and prints the confidence score.
```bash
curl -O https://s3.amazonaws.com/mxnet-model-server/onnx-duc/duc-inference.py
```

* **Run Prediction**:
```bash
python duc-inference.py city1.jpg
```

## <a name="ferplus_onnx"></a>FERPlus
* **Type**: Emotion detection trained on FER2013 dataset (model imported from ONNX)

* **Reference**: [Barsoum et al.](https://arxiv.org/abs/1608.01041)

* **Model Service**: [emotion_detection_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/FERPlus/emotion_detection_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models FERPlus=https://s3.amazonaws.com/model-server/model_archive_1.0/FERPlus.mar
```

* **Get a test image**:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/ferplus-input.jpg
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/FERPlus -T ferplus-input.jpg
```


## <a name="inception_v1"></a>Inception v1
* **Type**: Image classification trained on ImageNet

* **Reference**: [Szegedy, et al., Google](https://arxiv.org/pdf/1512.00567.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models onnx-inception-v1=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-inception_v1.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/onnx-inception-v1 -T kitten.jpg
```


## <a name="inception_v3"></a>Inception v3
* **Type**: Image classification trained on ImageNet

* **Reference**: [Szegedy, et al., Google](https://arxiv.org/pdf/1512.00567.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models inception-bn=https://s3.amazonaws.com/model-server/model_archive_1.0/inception-bn.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/inception-bn -T kitten.jpg
```


## <a name="lstm-ptb"></a>LSTM PTB
Long short-term memory network trained on the PennTreeBank dataset.

* **Reference**: [Hochreiter, et al.](http://www.bioinf.jku.at/publications/older/2604.pdf)

* **Model Service**: [lstm_ptb_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/lstm_ptb/lstm_ptb_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models lstm_ptb=https://s3.amazonaws.com/model-server/model_archive_1.0/lstm_ptb.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/lstm_ptb -H "Content-Type: application/json" -d '[{"input_sentence": "on the exchange floor as soon as ual stopped trading we <unk> for a panic said one top floor trader"}]'
```

## <a name="mobilenetv2-1.0_onnx"></a>MobileNetv2-1.0 (from ONNX model zoo)
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Sandler et al.](https://arxiv.org/abs/1801.04381)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models mobilenet=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-mobilenet.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/mobilenet -T kitten.jpg
```


## <a name="nin"></a>Network in Network
* **Type**: Image classification trained on ImageNet

* **Reference**: [Lin, et al.](https://arxiv.org/pdf/1312.4400v3.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models nin=https://s3.amazonaws.com/model-server/model_archive_1.0/nin.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/nin -T kitten.jpg
```


## <a name="resnet-152"></a>ResNet-152
* **Type**: Image classification trained on ImageNet

* **Reference**: [Lin, et al.](https://arxiv.org/pdf/1312.4400v3.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet-152=https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-152.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet-152 -T kitten.jpg
```


## <a name="resnet-18"></a>ResNet-18
* **Type**: Image classification trained on ImageNet

* **Reference**: [He, et al.](https://arxiv.org/pdf/1512.03385v1.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet-18=https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-18.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet-18 -T kitten.jpg
```


## <a name="resnet50-ssd"></a>ResNet50-SSD
* **Type**: Image classification trained on ImageNet

* **Reference**: [Liu, et al.](https://arxiv.org/pdf/1512.02325v4.pdf)

* **Model Service**: [ssd_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/ssd/ssd_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models SSD=https://s3.amazonaws.com/model-server/model_archive_1.0/resnet50_ssd.mar
```

* **Run Prediction**:
```bash
curl -O https://www.dphotographer.co.uk/users/21963/thm1024/1337890426_Img_8133.jpg

curl -X POST http://127.0.0.1:8080/predictions/SSD -T 1337890426_Img_8133.jpg
```


## <a name="resnext101"></a>ResNext101-64x4d
* **Type**: Image classification trained on ImageNet

* **Reference**: [Xie, et al.](https://arxiv.org/pdf/1611.05431.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnext101=https://s3.amazonaws.com/model-server/model_archive_1.0/resnext-101-64x4d.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnext101 -T kitten.jpg
```

## <a name="resnet_header"></a>ResNet (from ONNX model zoo)

### <a name="resnet-18v1"></a>ResNet18-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet18-v1=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet18v1.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet18-v1 -T kitten.jpg
```

### <a name="resnet-34v1"></a>ResNet34-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet34-v1=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet34v1.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet34-v1 -T kitten.jpg
```

### <a name="resnet-50v1"></a>ResNet50-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet50-v1=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet50v1.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet50-v1 -T kitten.jpg
```

### <a name="resnet-101v1"></a>ResNet101-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet101-v1=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet101v1.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet101-v1 -T kitten.jpg
```

### <a name="resnet-152v1"></a>ResNet152-v1
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1512.03385)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet152-v1=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet152v1.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet152-v1 -T kitten.jpg
```

### <a name="resnet-18v2"></a>ResNet18-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet18-v2=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet18v2.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet18-v2 -T kitten.jpg
```

### <a name="resnet-34v2"></a>ResNet34-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet34-v2=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet34v2.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet34-v2 -T kitten.jpg
```

### <a name="resnet-50v2"></a>ResNet50-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet50-v2=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet50v2.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet50-v2 -T kitten.jpg
```

### <a name="resnet-101v2"></a>ResNet101-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet101-v2=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet101v2.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet101-v2 -T kitten.jpg
```

### <a name="resnet-152v2"></a>ResNet152-v2
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [He, et al.](https://arxiv.org/abs/1603.05027)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models resnet152-v2=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-resnet152v2.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/resnet152-v2 -T kitten.jpg
```

## <a name="Shufflenet"></a>Shufflenet_v2
* **Type**: Image classification trained on ImageNet

* **Reference**: [Zhang, et al.](https://arxiv.org/abs/1707.01083)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models shufflenet=https://s3.amazonaws.com/model-server/model_archive_1.0/shufflenet.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/shufflenet -T kitten.jpg
```

## <a name="squeezenet_v1.1"></a>SqueezeNet v1.1
* **Type**: Image classification trained on ImageNet

* **Reference**: [Iandola, et al.](https://arxiv.org/pdf/1602.07360v4.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models squeezenet_v1.1=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/squeezenet_v1.1 -T kitten.jpg
```

## <a name="squeezenet_v1.1_onnx"></a>SqueezeNet v1.1 (from ONNX model zoo)
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Iandola, et al.](https://arxiv.org/pdf/1602.07360v4.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models onnx-squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-squeezenet.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/onnx-squeezenet -T kitten.jpg
```


## <a name="vgg16"></a>VGG16
* **Type**: Image classification trained on ImageNet

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models vgg16=https://s3.amazonaws.com/model-server/model_archive_1.0/vgg16.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/vgg16 -T kitten.jpg
```

## <a name="vgg19"></a>VGG19
* **Type**: Image classification trained on ImageNet

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf)

* **Model Service**: [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/master/examples/mxnet_vision/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models vgg19=https://s3.amazonaws.com/model-server/model_archive_1.0/vgg19.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/vgg19 -T kitten.jpg
```

## <a name="vgg_header"></a>VGG (from ONNX model zoo)

### <a name="vgg16_onnx"></a>VGG16
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models onnx-vgg16=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-vgg16.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/onnx-vgg16 -T kitten.jpg
```

### <a name="vgg16_bn_onnx"></a>VGG16_bn
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf) (Batch normalization applied after each conv layer of VGG16)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models onnx-vgg16_bn=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-vgg16_bn.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/onnx-vgg16_bn -T kitten.jpg
```

### <a name="vgg19_onnx"></a>VGG19
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models onnx-vgg19=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-vgg19.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/onnx-vgg19 -T kitten.jpg
```

### <a name="vgg19_bn_onnx"></a>VGG19_bn
* **Type**: Image classification trained on ImageNet (imported from ONNX)

* **Reference**: [Simonyan, et al.](https://arxiv.org/pdf/1409.1556v6.pdf) (Batch normalization applied after each conv layer of VGG19)

* **Model Service**: [mxnet_vision_service.py](https://s3.amazonaws.com/model-server/model_archive_1.0/mxnet_vision_service.py)

* **Start Server**:
```bash
mxnet-model-server --start --models onnx-vgg19_bn=https://s3.amazonaws.com/model-server/model_archive_1.0/onnx-vgg19_bn.mar
```

* **Run Prediction**:
```bash
curl -X POST http://127.0.0.1:8080/predictions/onnx-vgg19_bn -T kitten.jpg
```
