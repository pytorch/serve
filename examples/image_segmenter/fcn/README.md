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
    torchserve --start --model-store model_store --models fcn=fcn_resnet_101.mar
    curl http://127.0.0.1:8080/predictions/fcn -T examples/image_segmenter/persons.jpg
    ```
* Output
An array of shape [Batch, Height, Width, 2] where the final dimensions are [class, probability]

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
wget https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth
torch-model-archiver --model-name fcn --version 1.0 --model-file examples/image_segmenter/fcn/model.py --serialized-file examples/image_segmenter/fcn_resnet101_coco-7ecb50ca.pth --handler image_segmenter --extra-files examples/image_segmenter/fcn/fcn.py,examples/image_segmenter/fcn/intermediate_layer_getter.py
mv fcn.mar model-store/
---
docker pull pytorch/torchserve
docker run -p 8080:8080 -p 8081:8081 -v /home/ubuntu/model-store:/home/model-server/model-store pytorch/torchserve:latest
---
docker ps
docker exec -u root:root -it $CONTAINER_ID /bin/bash 
apt update && apt install wget curl
curl -X POST "http://localhost:8081/models?url=fcn.mar" # https://github.com/pytorch/serve/issues/712
curl -v -X PUT "http://localhost:8081/models/fcn?min_worker=3" # https://pytorch.org/serve/management_api.html#scale-workers
---
wget "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
curl "http://localhost:8080/predictions/fcn" -T dog.jpg
exit
---
docker ps
docker kill $CONTAINER_ID
```