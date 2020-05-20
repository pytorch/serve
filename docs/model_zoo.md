# Model Zoo

This page lists model archives that are pre-trained and pre-packaged, ready to be served for inference with TorchServe.
To propose a model for inclusion, please submit a [pull request](https://github.com/pytorch/serve/pulls).

*Special thanks to the [PyTorch](https://pytorch.org/) community whose Model Zoo and Model Examples were used in generating these model archives.*


| Model File | Type | Dataset | Size | Download |
| --- | --- | --- | --- | --- |
| AlexNet | Image Classification | ImageNet | 216 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/alexnet.mar)|
| Densenet161 | Image Classification | ImageNet | 106 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/densenet161.mar)|
| Resnet18 | Image Classification | ImageNet | 41 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar)|
| VGG11 | Image Classification | ImageNet | 471 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/vgg11.mar)|
| Squeezenet 1_1 | Image Classification | ImageNet | 4.4 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar)|
| MNIST digit classifier | Image Classification | MNIST | 4.3 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/mnist.mar)|
| Resnet 152 |Image Classification | ImageNet | 214 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/resnet-152-batch.mar)
| Faster RCNN | Object Detection | COCO | 148 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/fastrcnn.mar)|
| MASK RCNN | Object Detection | COCO | 158 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/maskrcnn.mar)|
| Text classifier | Text Classification | AG_NEWS | 169 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/my_text_classifier.mar)|
| FCN Resenet 101 | Image Segmentation | COCO | 193 MB | [.mar](https://torchserve.s3.amazonaws.com/mar_files/fcn_resnet_101.mar)|
Refer [example](../examples) for more details on above models.