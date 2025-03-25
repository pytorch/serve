# Object Detection using torchvision's pretrained fast-rcnn model

* Download the pre-trained fast-rcnn object detection model's state_dict from the following URL :

https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

```bash
wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
```

* Create a model archive file and serve the fastrcnn model in TorchServe using below commands

    ```bash
    torch-model-archiver --model-name fastrcnn --version 1.0 --model-file examples/object_detector/fast-rcnn/model.py --serialized-file fasterrcnn_resnet50_fpn_coco-258fb6c6.pth --handler object_detector --extra-files examples/object_detector/index_to_name.json
    mkdir model_store
    mv fastrcnn.mar model_store/
    torchserve --start --model-store model_store --models fastrcnn=fastrcnn.mar --disable-token-auth  --enable-model-api
    curl http://127.0.0.1:8080/predictions/fastrcnn -T examples/object_detector/detectron2/person.jpg
    ```
* Note : The objects detected have scores greater than "0.5". This threshold value is set in object_detector handler.

* Output

```json
[
  {
    "person": [
      362.34539794921875,
      161.9876251220703,
      515.53662109375,
      385.2342834472656
    ],
    "score": 0.9977679252624512
  },
  {
    "handbag": [
      67.37423706054688,
      277.63787841796875,
      111.6810073852539,
      400.26470947265625
    ],
    "score": 0.9925485253334045
  },
  {
    "handbag": [
      228.7159423828125,
      145.87753295898438,
      303.5065612792969,
      231.10513305664062
    ],
    "score": 0.9921919703483582
  }
]
```
