# PyToch serving  
This example shows how to serve PyTorch trained models for crocodile species recognition..
The custom handler is implemented in `densenet_service.py`. For simplicity, we'll use a pre-trained model.

We will use existing model archive [mar] file  with following contents - 
1. densent_service.py - Custom handler code
2. index_to_name.json - class to name mapping
3. model.pth - state_dict file

Downloading mar file
```bash
  curl -O https://github.com/pytorch/serve/temp/example/densent_vision/densent161.jpeg
```

Start the server from inside the container:
```bash
  torchserve --model-store=/Users/dhaniram_kshirsagar/projects/neo-sagemaker/mms/model-store --models densenet161=densenet161.mar
```

Now we can download a sample crocodile's image
```bash
  curl -O https://github.com/pytorch/serve/temp/example/densent_vision/croco.jpg
```
Get the inference of the model with the following:
```bash
  curl -X POST http://127.0.0.1:8080/predictions/densenet161_pytorch -T croco.jpg
```
```json
[
  {
    "African_crocodile": 0.9915103912353516
  },
  {
    "American_alligator": 0.00846970733255148
  },
  {
    "hippopotamus": 5.692463673767634e-06
  },
  {
    "Komodo_dragon": 2.0629049686249346e-06
  },
  {
    "common_iguana": 1.4585045846615685e-06
  }
]
```

For more information on MAR files and the built-in REST APIs, see:
* https://github.com/pytorch/serve/temp/docs
