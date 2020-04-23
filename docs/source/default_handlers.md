# TorchServe default inference handlers

TorchServe provides following inference handlers out of box :

## image_classifier

 * Description : Handles image classification models trained on imagenet dataset.
 * Input : RGB image
 * Output : Top 5 predictions and their respective probability of the image

For more details refer [examples](../examples/image_classifier)
 
## image_segmenter

 * Description : Handles image segmentation models trained on imagenet dataset.
 * Input : RGB image
 * Output : Output shape as [CL H W], CL - number of classes, H - height and W - width.

For more details refer [examples](../examples/image_segmenter)

## object_detector

 * Description : Handles object detection models.
 * Input : RGB image
 * Output : List of detected classes and bounding boxes respectively 

For more details refer [examples](../examples/object_detector) 

## text_classifier

 * Description : Handles models trained on imagenet dataset.
 * Input : text file
 * Output : Class of input text
 
For more details refer [examples](../examples/text_classification)
 
