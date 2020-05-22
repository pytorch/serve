# Object Detection using torchvision's pretrained fast-rcnn model.

* Download the pre-trained fast-rcnn object detection model's state_dict from the following URL :

https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

```bash
wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
```

* Create a model archive file and serve the fastrcnn model in TorchServe using below commands

    ```bash
    torch-model-archiver --model-name fastrcnn --version 1.0 --model-file serve/examples/object_detector/fast-rcnn/model.py --serialized-file fasterrcnn_resnet50_fpn_coco-258fb6c6.pth --handler object_detector --extra-files serve/examples/object_detector/index_to_name.json
    mkdir model_store
    mv fastrcnn.mar model_store/
    torchserve --start --model-store model_store --models fastrcnn=fastrcnn.mar
    curl http://127.0.0.1:8080/predictions/fastrcnn -T serve/examples/object_detector/persons.jpg
    ```
* Output

```json
[
  {
    "person": "[(167.6395, 56.781574), (301.6996, 437.15158)]"
  },
  {
    "person": "[(89.61491, 64.89805), (191.40207, 446.66052)]"
  },
  {
    "person": "[(362.3454, 161.98763), (515.5366, 385.23428)]"
  },
  {
    "handbag": "[(67.37424, 277.63788), (111.68101, 400.2647)]"
  },
  {
    "handbag": "[(228.71594, 145.87753), (303.50656, 231.10513)]"
  },
  {
    "handbag": "[(379.42468, 259.9776), (419.01486, 317.95105)]"
  },
  {
    "person": "[(517.9014, 149.55002), (636.5953, 365.52505)]"
  },
  {
    "bench": "[(268.99918, 217.2433), (423.95178, 390.4785)]"
  },
  {
    "person": "[(539.68317, 157.81715), (616.1689, 253.0961)]"
  },
  {
    "person": "[(477.1378, 147.92549), (611.0255, 297.92764)]"
  },
  {
    "bench": "[(286.66885, 216.35751), (550.45374, 383.19562)]"
  },
  {
    "person": "[(627.4468, 177.199), (640.0, 247.35138)]"
  },
  {
    "handbag": "[(406.96024, 261.82846), (453.762, 357.5365)]"
  },
  {
    "bench": "[(83.19274, 226.72401), (560.9781, 422.55176)]"
  },
  {
    "chair": "[(451.366, 207.4905), (504.65698, 287.66193)]"
  },
  {
    "chair": "[(454.38974, 207.96115), (487.7692, 270.3133)]"
  }
]
```