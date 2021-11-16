# TorchServe default inference handlers

TorchServe provides following inference handlers out of box. It's expected that the models consumed by each support batched inference.

## image_classifier

* Description : Handles image classification models trained on the ImageNet dataset.
* Input : RGB image
* Output : Batch of top 5 predictions and their respective probability of the image

For more details see [examples](https://github.com/pytorch/serve/tree/master/examples/image_classifier)

## image_segmenter

* Description : Handles image segmentation models trained on the ImageNet dataset.
* Input : RGB image
* Output : Output shape as [N, CL H W], N - batch size, CL - number of classes, H - height and W - width.

For more details see [examples](https://github.com/pytorch/serve/tree/master/examples/image_segmenter)

## object_detector

* Description : Handles object detection models.
* Input : RGB image
* Output : Batch of lists of detected classes and bounding boxes respectively

Note : We recommend running `torchvision>0.6` otherwise the object_detector default handler will only run on the default GPU device

For more details see [examples](https://github.com/pytorch/serve/tree/master/examples/object_detector)

## text_classifier

* Description : Handles models trained on the AG_NEWS dataset.
* Input : text file
* Output : Class of input text. (No batching supported)

For more details see [examples](https://github.com/pytorch/serve/tree/master/examples/text_classification)

For a more comprehensive list of available handlers make sure to check out the [examples page](https://github.com/pytorch/serve/tree/master/examples)

## Common features

### index_to_name.json

`image_classifier`, `text_classifier` and `object_detector` can all automatically map from numeric classes (0,1,2...) to friendly strings. To do this, simply include in your model archive a file, `index_to_name.json`, that contains a mapping of class number (as a string) to friendly name (also as a string). You can see some examples here:
- [image_classifier](https://github.com/pytorch/serve/tree/master/examples/image_classifier/index_to_name.json)
- [text_classifier](https://github.com/pytorch/serve/tree/master/examples/text_classification/index_to_name.json)
- [object_detector](https://github.com/pytorch/serve/tree/master/examples/object_detector/index_to_name.json)

### Contributing
We welcome new contributed handlers, if your usecase isn't covered by one of the existing default handlers please follow the below steps to contribute it
1. Write a new class derived from [BaseHandler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py). Add it as a separate file in `ts/torch_handler/`
2. Update `model-archiver/model_packaging.py` to add in your classes name
3. Run and update the unit tests in [unit_tests](https://github.com/pytorch/serve/tree/master/ts/torch_handler/unit_tests). As always, make sure to run [torchserve_sanity.py](https://github.com/pytorch/serve/tree/master/torchserve_sanity.py) before submitting.
