# Image Segmentation using torchvision's pretrained deeplabv3_resnet_101_coco model.

* Download the pre-trained deeplabv3_resnet_101_coco image segmentation model's state_dict from the following URL:

https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth

```bash
wget https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth
```

* Create a model archive file and serve the deeplabv3 model in TorchServe using below commands

    ```bash
    torch-model-archiver --model-name deeplabv3_resnet_101 --version 1.0 --model-file examples/image_segmenter/deeplabv3/model.py --serialized-file deeplabv3_resnet101_coco-586e9e4e.pth --handler image_segmenter --extra-files examples/image_segmenter/deeplabv3/deeplabv3.py,examples/image_segmenter/deeplabv3/intermediate_layer_getter.py,examples/image_segmenter/deeplabv3/fcn.py
    mkdir model_store
    mv deeplabv3_resnet_101.mar model_store/
    torchserve --start --model-store model_store --models deeplabv3=deeplabv3_resnet_101.mar --disable-token-auth  --enable-model-api
    curl http://127.0.0.1:8080/predictions/deeplabv3 -T examples/image_segmenter/persons.jpg
    ```
* Output
An array of shape [Batch, Height, Width, 2] where the final dimensions are [class, probability]

```json
[[[0.0, 0.9988763332366943], [0.0, 0.9988763332366943], [0.0, 0.9988763332366943], [0.0, 0.9988763332366943], [0.0, 0.9988666772842407], [0.0, 0.9988440275192261], [0.0, 0.9988170862197876], [0.0, 0.9987859725952148] ... ]]
```
