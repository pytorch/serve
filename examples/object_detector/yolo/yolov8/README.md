# Object Detection using Ultralytics's pretrained YOLOv8(yolov8n) model.


Install `ultralytics` using
```
python -m pip install -r requirements.txt
```

In this example, we are using the YOLOv8 Nano model from ultralytics. Download the pretrained weights from [Ultralytics](https://docs.ultralytics.com/models/yolov8/#supported-modes)

```
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

We need a custom handler to load the YOLOv8n model. The default `initialize` function loads `.pt` file using `torch.jit.load`. This doesn't work for YOLOv8n model. Hence, we need a custom handler with an `initialize` method where we load the model using ultralytics.

## Create a model archive file for Yolov8n model

```
torch-model-archiver --model-name yolov8n --version 1.0 --serialized-file yolov8n.pt --handler custom_handler.py
```

```
mkdir model_store
mv yolov8n.mar model_store/.
```

## Start TorchServe and register the model


```
torchserve --start --model-store model_store --ncs --disable-token-auth  --enable-model-api
curl -X POST "localhost:8081/models?model_name=yolov8n&url=yolov8n.mar&initial_workers=4&batch_size=2"
```

results in

```
{
  "status": "Model \"yolov8n\" Version: 1.0 registered with 4 initial workers"
}
```

## Run Inference

Here we are counting the number of detected objects in the image. You can change the post-process method in the handler to return the bounding box coordinates

```
curl http://127.0.0.1:8080/predictions/yolov8n -T persons.jpg  & curl http://127.0.0.1:8080/predictions/yolov8n -T bus.jpg
```

gives the output

```
{
  "person": 4,
  "handbag": 3,
  "bench": 3
}{
  "person": 4,
  "bus": 1,
  "stop sign": 1
}
```
