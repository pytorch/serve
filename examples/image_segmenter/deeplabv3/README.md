# Image Segmentation using torchvision's pretrained deeplabv3_resnet_101_coco model.

* Download the pre-trained deeplabv3_resnet_101_coco image segmentation model's state_dict from the following URL:

https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth

```bash
wget https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth
```

* Create a model archive file and serve the deeplabv3 model in TorchServe using below commands

    ```bash
    torch-model-archiver --model-name deeplabv3_resnet_101 --version 1.0 --model-file examples/image_segmenter/deeplabv3/model.py --serialized-file deeplabv3_resnet101_coco-586e9e4e.pth --handler image_segmenter --extra-files examples/image_segmenter/deeplabv3/deeplabv3.py,examples/image_segmenter/deeplabv3/intermediate_layer_getter.py
    mkdir model_store
    mv deeplabv3_resnet_101.mar model_store/
    torchserve --start --model-store model_store --models deeplabv3=deeplabv3_resnet_101.mar
    curl http://127.0.0.1:8080/predictions/deeplabv3 -T examples/image_segmenter/persons.jpg
    ```
* Output
An array of shape [Batch, Height, Width, 2] where the final dimensions are [class, probability]

# TODO: test and update output
```json
[[[[0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993857145309448], [0.0, 0.9993864297866821], [0.0, 0.999385416507721], [0.0, 0.9993811845779419], [0.0, 0.9993740320205688] ... ]]]
```

# DEPLOYMENT NOTES:

```
conda create --name torch
conda activate torch
conda install python=3.7
conda install pytorch torchvision -c pytorch
conda install torchserve torch-model-archiver -c pytorch
pip install torchserve torch-model-archiver
---
wget https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth
torch-model-archiver --model-name deeplabv3 --version 1.0 --model-file examples/image_segmenter/deeplabv3/model.py --serialized-file examples/image_segmenter/deeplabv3_resnet101_coco-586e9e4e.pth --handler image_segmenter --extra-files examples/image_segmenter/deeplabv3/deeplabv3.py,examples/image_segmenter/deeplabv3/intermediate_layer_getter.py
mv deeplabv3.mar model-store/
---
docker pull pytorch/torchserve
docker run -p 8080:8080 -p 8081:8081 -v /home/ubuntu/model-store:/home/model-server/model-store pytorch/torchserve:latest
---
docker ps
docker exec -u root:root -it $CONTAINER_ID /bin/bash 
apt update && apt install wget curl
curl -X POST "http://localhost:8081/models?url=deeplabv3.mar" # https://github.com/pytorch/serve/issues/712
curl -v -X PUT "http://localhost:8081/models/deeplabv3?min_worker=3" # https://pytorch.org/serve/management_api.html#scale-workers
---
wget "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
curl "http://localhost:8080/predictions/deeplabv3" -T dog.jpg
exit
---
docker ps
docker kill $CONTAINER_ID
```