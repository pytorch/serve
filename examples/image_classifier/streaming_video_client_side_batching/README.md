#### Sample commands to create a resnet-18 eager mode model archive, register it on TorchServe and run inference on a streaming video

In this example, we are using OpenCV to send frames on the client side.
Install opencv with the following command
```
pip install opencv-python
```

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path

```bash
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./examples/image_classifier/streaming_video_client_side_batching/model.py --serialized-file resnet18-f37072fd.pth --handler ./examples/image_classifier/streaming_video_client_side_batching/custom_handler.py --extra-files ./examples/image_classifier/index_to_name.json
mkdir model_store
mv resnet-18.mar model_store/
torchserve --start --model-store model_store --models resnet-18=resnet-18.mar
python examples/image_classifier/streaming_video_client_side_batching/request.py
```

If you have a camera connected, you can run inference on streaming video from the camera as follows

```
python examples/image_classifier/streaming_video_client_side_batching/request.py --batch_size 10 --input 0
```

