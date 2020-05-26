# Image Segmentation using torchvision's pretrained fcn_resnet_101_coco model.

* Download the pre-trained fcn_resnet_101_coco image segmentation model's state_dict from the following URL :

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
    curl http://127.0.0.1:8080/predictions/fcn -T examples/image_segmenter/fcn/persons.jpg -o out_mask_persons.jpg

    'out_mask_persons.jpg' is a masked output image which can be opened using any image viewer/browser.
    ```

* Run inference and download audio output using python script :

```python
import requests

files = {'data': open('examples/image_segmenter/fcn/persons.jpg','rb')}
response = requests.post('http://localhost:8080/predictions/fcn', files=files)
data = response.content

with open('out_mask_persons.jpg', 'wb') as imagefile:
    imagefile.write(data)
```

* Change the host and port in above samples as per your server configuration.

* Response : An out_mask_persons.jpg file gets downloaded.