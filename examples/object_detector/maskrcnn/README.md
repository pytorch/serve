# Object Detection using torchvision's pretrained maskrcnn model.

* Download the pre-trained maskrcnn object detection model's state_dict from the following URL :

https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth

```bash
wget https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth
```

* Create a model archive file and serve the maskrcnn model in TorchServe using below commands

    ```bash
    torch-model-archiver --model-name maskrcnn --version 1.0 --model-file serve/examples/object_detector/maskrcnn/model.py --serialized-file maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth --handler object_detector --extra-files serve/examples/object_detector/index_to_name.json
    mkdir model_store
    mv maskrcnn.mar model_store/
    torchserve --start --model-store model_store --models maskrcnn=maskrcnn.mar
    curl http://127.0.0.1:8080/predictions/maskrcnn -T serve/examples/object_detector/persons.jpg
    ```
* Output

```json
[
  {
    "person": "[(169.61879, 50.145702), (300.844, 442.49292)]"
  },
  {
    "person": "[(90.418335, 66.83669), (194.21136, 437.27753)]"
  },
  {
    "person": "[(362.38925, 158.00893), (521.066, 385.55084)]"
  },
  {
    "handbag": "[(68.58448, 279.28394), (111.14233, 400.91205)]"
  },
  {
    "person": "[(473.854, 147.2746), (638.38654, 364.52316)]"
  },
  {
    "handbag": "[(225.6044, 142.74402), (302.4504, 230.29791)]"
  },
  {
    "handbag": "[(380.28204, 259.18207), (419.5152, 318.27216)]"
  },
  {
    "bench": "[(273.41745, 217.40706), (441.2533, 396.36096)]"
  },
  {
    "person": "[(541.3647, 156.64714), (620.07886, 249.49536)]"
  },
  {
    "chair": "[(455.21783, 207.56235), (491.11472, 274.75076)]"
  },
  {
    "person": "[(626.24615, 178.66173), (640.0, 246.10945)]"
  },
  {
    "dog": "[(557.7418, 202.89917), (611.424, 256.95578)]"
  },
  {
    "person": "[(359.27438, 161.60461), (493.7587, 296.96854)]"
  },
  {
    "person": "[(548.9496, 177.09854), (640.0, 364.51437)]"
  },
  {
    "bench": "[(297.377, 208.0484), (563.4821, 380.4136)]"
  },
  {
    "handbag": "[(412.6865, 272.41565), (459.14282, 363.98538)]"
  },
  {
    "bench": "[(444.64893, 204.42014), (627.00635, 359.8998)]"
  }
]
```