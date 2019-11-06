# PyToch serving  
This example shows how to serve PyTorch trained models for flower species recognition..
The custom handler is implemented in `densenet_service.py`.
For simplicity, we'll use a pre-trained model. For simplicity we will use docker container to run Model Server.

## Getting Started With Docker
Build the docker image with pytorch as backend engine:
```bash
  cd examples/densenet_pytorch/
  docker build . -t mms_with_pytorch
```

Run the container that you have built in previous step.
```bash
  docker run -it --entrypoint bash mms_with_pytorch
```

Start the server from inside the container:
```bash
  mxnet-model-server --models densenet161_pytorch=https://s3.amazonaws.com/model-server/model_archive_1.0/examples/PyTorch+models/densenet/densenet161_pytorch.mar
```

Now we can download a sample flower's image
```bash
  curl -O https://s3.amazonaws.com/model-server/inputs/flower.jpg
```
Get the status of the model with the following:
```bash
  curl -X POST http://127.0.0.1:8080/predictions/densenet161_pytorch -T flower.jpg
```
```json
[
  {
    "canna lily": 0.01565943844616413
  },
  {
    "water lily": 0.015515935607254505
  },
  {
    "purple coneflower": 0.014358781278133392
  },
  {
    "globe thistle": 0.014226051047444344
  },
  {
    "ruby-lipped cattleya": 0.014212552458047867
  }
  ]
```

For more information on MAR files and the built-in REST APIs, see:
* https://github.com/awslabs/mxnet-model-server/tree/master/docs
