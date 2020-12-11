# Predict on a InferenceService using PyTorch Server and Transformer

Most of the model servers expect tensors as input data, so a pre-processing step is needed before making the prediction call if the user is sending in raw input format. Transformer is a service we orchestrated from InferenceService spec for user implemented pre/post processing code. In the [pytorch](../README.md) example we call the prediction endpoint with tensor inputs, and in this example we add additional pre-processing step to allow the user send raw image data.

## Setup
1. Your ~/.kube/config should point to a cluster with [KFServing installed](https://github.com/kubeflow/kfserving/#install-kfserving).
2. Your cluster's Istio Ingress gateway must be [network accessible](https://istio.io/latest/docs/tasks/traffic-management/ingress/ingress-control/).

##  Build Transformer image

### Extend KFModel and implement pre/post processing functions
```python
import kfserving
from typing import List, Dict
from PIL import Image
import torchvision.transforms as transforms
import logging
import io
import numpy as np
import base64

logging.basicConfig(level=kfserving.constants.KFSERVING_LOGLEVEL)

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def image_transform(instance):
    byte_array = base64.b64decode(instance['image_bytes']['b64'])
    image = Image.open(io.BytesIO(byte_array))
    a = np.asarray(image)
    im = Image.fromarray(a)
    res = transform(im)
    logging.info(res)
    return res.tolist()


class ImageTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        return {'instances': [image_transform(instance) for instance in inputs['instances']]}

    def postprocess(self, inputs: List) -> List:
        return inputs
```

### Build Transformer docker image
This step can be part of your CI/CD pipeline to continuously build the transformer image version. 
```shell
docker build -t gcr.io/kubeflow-ci/kfserving/image-transformer:latest -f transformer.Dockerfile .
```
