# TorchServe default inference handlers

TorchServe provides following inference handlers out of box:

## image_classifier

* Description : Handles image classification models trained on the ImageNet dataset.
* Input : RGB image
* Output : Top 5 predictions and their respective probability of the image

For more details see [examples](https://github.com/pytorch/serve/tree/master/examples/image_classifier)

## image_segmenter

* Description : Handles image segmentation models trained on the ImageNet dataset.
* Input : RGB image
* Output : Output shape as [CL H W], CL - number of classes, H - height and W - width.

For more details see [examples](https://github.com/pytorch/serve/tree/master/examples/image_segmenter)

## object_detector

* Description : Handles object detection models.
* Input : RGB image
* Output : List of detected classes and bounding boxes respectively 

Note : For torchvision version lower than 0.6, the object_detector default handler runs on only default GPU device in GPU based environment.

For more details see [examples](https://github.com/pytorch/serve/tree/master/examples/object_detector) 

## text_classifier

* Description : Handles models trained on the ImageNet dataset.
* Input : text file
* Output : Class of input text

For more details see [examples](https://github.com/pytorch/serve/tree/master/examples/text_classification)
