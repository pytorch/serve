# Model Zoo

This page lists model archives that are pre-trained and pre-packaged, ready to be served for inference with TorchServe.
To propose a model for inclusion, please submit a [pull request](https://github.com/pytorch/serve/pulls).

*Special thanks to the [PyTorch](https://pytorch.org/) community whose Model Zoo and Model Examples were used in generating these model archives.*


| Model | Type | Dataset | Size | Download | Sample Input| Model mode |
| --- | --- | --- | --- | --- | --- | --- |
| AlexNet | Image Classification | ImageNet | 216 MB | [.mar](https://torchserve.pytorch.org/mar_files/alexnet.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Eager|
| Densenet161 | Image Classification | ImageNet | 106 MB | [.mar](https://torchserve.pytorch.org/mar_files/densenet161.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Eager|
| Resnet18 | Image Classification | ImageNet | 41 MB | [.mar](https://torchserve.pytorch.org/mar_files/resnet-18.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Eager|
| VGG16 | Image Classification | ImageNet | 489 MB | [.mar](https://torchserve.pytorch.org/mar_files/vgg16.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Eager|
| Squeezenet 1_1 | Image Classification | ImageNet | 4.4 MB | [.mar](https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Eager|
| MNIST digit classifier | Image Classification | MNIST | 4.3 MB | [.mar](https://torchserve.pytorch.org/mar_files/mnist_v2.mar) | [0.png](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist/test_data) |Eager|
| Resnet 152 |Image Classification | ImageNet | 214 MB | [.mar](https://torchserve.pytorch.org/mar_files/resnet-152-batch_v2.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Eager|
| Faster RCNN | Object Detection | COCO | 148 MB | [.mar](https://torchserve.pytorch.org/mar_files/fastrcnn.mar) | [persons.jpg](https://github.com/pytorch/serve/blob/master/examples/object_detector/persons.jpg?raw=true) |Eager|
| MASK RCNN | Object Detection | COCO | 158 MB | [.mar](https://torchserve.pytorch.org/mar_files/maskrcnn.mar) | [persons.jpg](https://github.com/pytorch/serve/blob/master/examples/object_detector/persons.jpg?raw=true) |Eager|
| Text classifier | Text Classification | AG_NEWS | 169 MB | [.mar](https://torchserve.pytorch.org/mar_files/my_text_classifier_v4.mar) | [sample_text.txt](https://github.com/pytorch/serve/blob/master/examples/text_classification/sample_text.txt) |Eager|
| FCN ResNet 101 | Image Segmentation | COCO | 193 MB | [.mar](https://torchserve.pytorch.org/mar_files/fcn_resnet_101.mar) | [persons.jpg](https://github.com/pytorch/serve/blob/master/examples/image_segmenter/persons.jpg?raw=true) |Eager|
| DeepLabV3 ResNet 101 | Image Segmentation | COCO | 217 MB | [.mar](https://torchserve.pytorch.org/mar_files/deeplabv3_resnet_101_eager.mar) | [persons.jpg](https://github.com/pytorch/serve/blob/master/examples/image_segmenter/persons.jpg) |Eager|
| AlexNet Scripted | Image Classification | ImageNet | 216 MB | [.mar](https://torchserve.pytorch.org/mar_files/alexnet_scripted.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Torchscripted |
| Densenet161 Scripted| Image Classification | ImageNet | 105 MB | [.mar](https://torchserve.pytorch.org/mar_files/densenet161_scripted.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Torchscripted |
| Resnet18 Scripted| Image Classification | ImageNet | 42 MB | [.mar](https://torchserve.pytorch.org/mar_files/resnet-18_scripted.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Torchscripted |
| VGG16 Scripted| Image Classification | ImageNet | 489 MB | [.mar](https://torchserve.pytorch.org/mar_files/vgg16_scripted.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Torchscripted |
| Squeezenet 1_1 Scripted | Image Classification | ImageNet | 4.4 MB | [.mar](https://torchserve.pytorch.org/mar_files/squeezenet1_1_scripted.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Torchscripted |
| MNIST digit classifier Scripted | Image Classification | MNIST | 4.3 MB | [.mar](https://torchserve.pytorch.org/mar_files/mnist_scripted_v2.mar) | [0.png](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist/test_data) |Torchscripted |
| Resnet 152 Scripted |Image Classification | ImageNet | 215 MB | [.mar](https://torchserve.pytorch.org/mar_files/resnet-152-scripted_v2.mar) | [kitten.jpg](https://github.com/pytorch/serve/blob/master/examples/image_classifier/kitten.jpg?raw=true) |Torchscripted |
| Text classifier Scripted | Text Classification | AG_NEWS | 169 MB | [.mar](https://torchserve.pytorch.org/mar_files/my_text_classifier_scripted_v3.mar) | [sample_text.txt](https://github.com/pytorch/serve/blob/master/examples/text_classification/sample_text.txt) |Torchscripted |
| FCN ResNet 101 Scripted | Image Segmentation | COCO | 193 MB | [.mar](https://torchserve.pytorch.org/mar_files/fcn_resnet_101_scripted.mar) | [persons.jpg](https://github.com/pytorch/serve/blob/master/examples/image_segmenter/persons.jpg?raw=true) |Torchscripted |
| DeepLabV3 ResNet 101 Scripted | Image Segmentation | COCO | 217 MB | [.mar](https://torchserve.pytorch.org/mar_files/deeplabv3_resnet_101_scripted.mar) | [persons.jpg](https://github.com/pytorch/serve/blob/master/examples/image_segmenter/persons.jpg) |Torchscripted |

Refer [example](https://github.com/pytorch/serve/tree/master/examples) for more details on above models.
