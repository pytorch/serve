# Object Detection using torchvision's pretrained fast-rcnn model.

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
    curl http://127.0.0.1:8080/predictions/fastrcnn -T examples/object_detector/persons.jpg
    ```
* Note : The objects detected have scores greater than "0.5". This threshold value is set in object_detector handler.

* Output

```json
[
  {
    "person": [
      167.4222869873047,
      57.03825378417969,
      301.305419921875,
      436.68682861328125
    ],
    "score": 0.9995299577713013
  },
  {
    "person": [
      89.61490631103516,
      64.8980484008789,
      191.40206909179688,
      446.6605224609375
    ],
    "score": 0.9995074272155762
  },
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
  },
  {
    "handbag": [
      379.4246826171875,
      259.97760009765625,
      419.0148620605469,
      317.9510498046875
    ],
    "score": 0.9896279573440552
  },
  {
    "person": [
      517.9014282226562,
      149.55001831054688,
      636.5952758789062,
      365.5250549316406
    ],
    "score": 0.9828333854675293
  },
  {
    "bench": [
      268.9991760253906,
      217.24330139160156,
      423.9517822265625,
      390.4784851074219
    ],
    "score": 0.9581767916679382
  },
  {
    "person": [
      539.6831665039062,
      157.81715393066406,
      616.1688842773438,
      253.09609985351562
    ],
    "score": 0.8993930816650391
  },
  {
    "person": [
      477.1377868652344,
      147.9254913330078,
      611.0255126953125,
      297.9276428222656
    ],
    "score": 0.8726601600646973
  },
  {
    "bench": [
      286.6688537597656,
      216.35751342773438,
      550.4537353515625,
      383.19561767578125
    ],
    "score": 0.8438199162483215
  },
  {
    "person": [
      627.44677734375,
      177.19900512695312,
      640.0,
      247.35137939453125
    ],
    "score": 0.8364201188087463
  },
  {
    "bench": [
      88.39929962158203,
      226.47962951660156,
      560.918701171875,
      421.661865234375
    ],
    "score": 0.7469933032989502
  },
  {
    "handbag": [
      406.9602355957031,
      261.8284606933594,
      453.7619934082031,
      357.5364990234375
    ],
    "score": 0.7322059273719788
  },
  {
    "chair": [
      451.3659973144531,
      207.49049377441406,
      504.656982421875,
      287.66192626953125
    ],
    "score": 0.6674202084541321
  },
  {
    "chair": [
      454.3897399902344,
      207.96115112304688,
      487.7691955566406,
      270.31329345703125
    ],
    "score": 0.5939609408378601
  }
]
```
