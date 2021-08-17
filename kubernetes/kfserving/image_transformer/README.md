# Predict on a InferenceService using PyTorch Server and Transformer

Most of the model servers expect tensors as input data, so a pre-processing step is needed before making the prediction call if the user is sending in raw input format. Transformer is a service for users to implement pre/post processing code before making the prediction call. In this example we add additional pre-processing step to allow the user send raw image data and convert it into json array.


##  Build Transformer image

### Extend KFModel and implement pre/post processing functions
```python
EXPLAINER_URL_FORMAT = "http://{0}/v1/models/{1}:explain"

image_processing = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def image_transform(instance):
    byte_array = base64.b64decode(instance["data"])
    image = Image.open(io.BytesIO(byte_array))
    instance["data"] = image_processing(image).tolist()
    logging.info(instance)
    return instance


class ImageTransformer(kfserving.KFModel):

    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.explainer_host = predictor_host
        logging.info("MODEL NAME %s", name)
        logging.info("PREDICTOR URL %s", self.predictor_host)
        logging.info("EXPLAINER URL %s", self.explainer_host)
        self.timeout = 100

    def preprocess(self, inputs: Dict) -> Dict:

        return {'instances': [image_transform(instance) for instance in inputs['instances']]}

    def postprocess(self, inputs: List) -> List:
        return inputs
```
## Steps to run Image Transformer in local environment
### Install the Image Transformer and KFServing Repo

* Clone the KFServing Git Repository
```bash
git clone -b master https://github.com/kubeflow/kfserving.git
```

* Install KFServing as below:
```bash
pip install -e ./kfserving/python/kfserving
``` 

* Run the Install Dependencies script
```bash
python ./ts_scripts/install_dependencies.py --environment=dev
```

* Run the Install from Source command
```bash
python ./ts_scripts/install_from_src.py
```

* Create a directory to place the config.properties in the folder structure below:
```bash
sudo  mkdir -p /mnt/models/config/
```

* Move the model to /mnt/models/model-store

* Move the config.properties to /mnt/models/config/.

The config.properties file is as below:

```bash
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8085
metrics_address=http://0.0.0.0:8082
enable_metrics_api=true
metrics_format=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
enable_envvars_config=true
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"mnist":{"1.0":{"defaultVersion":true,"marName":"mnist.mar","minWorkers":1,"maxWorkers":5,"batchSize":5,"maxBatchDelay":200,"responseTimeout":60}}}}
```

* Create a directory to place the .mar file in the folder structure below
```bash
sudo  mkdir -p /mnt/models/model-store
```

* Install Image Transformer with the below command
```bash
pip install -e ./serve/kubernetes/kfserving/image_transformer/
```

* Run the Image Transformer with the below command
```bash
python3 -m image_transformer --predictor_host 0.0.0.0:8085
```
The transformer will hit the predictor host after pre-processing.
The predictor host is the inference url of torchserve.

* Set service envelope environment variable

The 
```export TS_SERVICE_ENVELOPE=kfserving``` or ```TS_SERVICE_ENVELOPE=kfservingv2``` envvar is for choosing between
KFServing v1 and v2 protocols

* Start torchserve using config.properties in /mnt/models/config/
```
torchserve --start --ts-config /mnt/models/config/config.properties
```

Please note that Model runs at port 8085,Image transformer runs at port 8080.
The request first comes to the image transformer at port 8080 and in turn requests the torchserve at port 8085. So our request should be made at port 8080.

* The curl request for inference is as below:
```
curl -H "Content-Type: application/json" --data @serve/kubernetes/kfserving/kf_request_json/mnist.json http://0.0.0.0:8080/v1/models/mnist:predict
```

output:
```json
{"predictions": [2]}
```

## Build Transformer docker image
This step can be used to continuously build the transformer image version. 
```shell
docker build -t <image_name>:<tag> -f transformer.Dockerfile .
```


