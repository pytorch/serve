# Image Segmentation using torchvision's pretrained fcn_resnet_101_coco model.

* Download the pre-trained fcn_resnet_101_coco image segmentation model's state_dict from the following URL:

https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth

```bash
wget https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth
```

* Create a model archive file and serve the fcn model in TorchServe using below commands

    ```bash
    torch-model-archiver --model-name fcn_resnet_101 --version 1.0 --model-file examples/image_segmenter/fcn/model.py --serialized-file fcn_resnet101_coco-7ecb50ca.pth --handler image_segmenter --extra-files examples/image_segmenter/fcn/fcn.py,examples/image_segmenter/fcn/intermediate_layer_getter.py
    mkdir model_store
    mv fcn_resnet_101.mar model_store/
    torchserve --start --model-store model_store --models fcn=fcn_resnet_101.mar --disable-token-auth  --enable-model-api
    curl http://127.0.0.1:8080/predictions/fcn -T examples/image_segmenter/persons.jpg
    ```
* Output
An array of shape [Batch, Height, Width, 2] where the final dimensions are [class, probability]

```json
[[[0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993864297866821], [0.0, 0.999385416507721], [0.0, 0.9993811845779419], [0.0, 0.9993740320205688] ... ]]
```
