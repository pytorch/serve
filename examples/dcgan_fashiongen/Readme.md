# GAN(Generative Adversarial Networks) models using TorchServe
- In this example we will demonstrate how to serve a GAN model using TorchServe.
- We have used a pretrained DCGAN model from [facebookresearch/pytorch_GAN_zoo](https://github.com/facebookresearch/pytorch_GAN_zoo)  
  (Introduction to [DCGAN on FashionGen](https://pytorch.org/hub/facebookresearch_pytorch-gan-zoo_dcgan/))

### 1. Create a Torch Model Archive
Execute the following command to create _dcgan_fashiongen.mar_ :
```
./create_mar.sh
```
This command does the following :
- Download the model's source code, extract the relevant directory and zip it. (`--extra-files`)
- Download a checkpoint file _DCGAN_fashionGen-1d67302.pth_.  (`--serialized-file`)
- Provide a custom handler - [dcgan_fashiongen_handler.py](dcgan_fashiongen_handler.py). (`--handler`)


### 2. Start TorchServe and Register Model
```
mkdir modelstore
mv dcgan_fashiongen.mar modelstore/
torchserve --start --ncs --model-store ./modelstore --models dcgan_fashiongen.mar
```

### 3. Generate Images
Invoke the predictions API and pass following payload(JSON  
**number_of_images** :  Number of images to generate  
**input_gender** : OPTIONAL; If specified, needs to be one of - `Men`, `Women`  
**input_category** : OPTIONAL; If specified, needs to be one of - One of - `SHIRTS`, `SWEATERS`, `JEANS`, `PANTS`, `TOPS`, `SUITS & BLAZERS`, `SHORTS`, `JACKETS & COATS`, `SKIRTS`, `JUMPSUITS`, `SWIMWEAR`, `DRESSES`  
**input_pose** : OPTIONAL; If specified, needs to be one of - `id_gridfs_1`, `id_gridfs_2`, `id_gridfs_3`, `id_gridfs_4`  

#### Example
```
curl -X POST -H "Content-Type: application/json" -d '{"number_of_images":64,"input_gender":"Men","input_category":"SHIRTS", "input_pose":"id_gridfs_1"}' http://localhost:8080/predictions/dcgan_fashiongen -o test_img1.jpg

curl -X POST -H "Content-Type: application/json" -d '{"number_of_images":32,"input_gender":"Women","input_category":"DRESSES", "input_pose":"id_gridfs_3"}' http://localhost:8080/predictions/dcgan_fashiongen -o test_img2.jpg

curl -X POST -H "Content-Type: application/json" -d '{"number_of_images":4}' http://localhost:8080/predictions/dcgan_fashiongen -o myimage3.jpg

```
