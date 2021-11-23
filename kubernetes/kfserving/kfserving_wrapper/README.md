# KFServing Wrapper

The KFServing wrapper folder contains three files :

1) __main__.py
2) TorchserveModel.py
3) TSModelRepository.py

The KFServing wrapper files were created to enable the Torchserve integration with KFServing. 

1) the __main__.py file parses the model snapshot from the config.properties present in the KFServing side and passes the parameters like inference address, management address and the model address to the KFServing side to handle the input request and response. 


2) The TorchserveModel.py file contains the methods to handle the request and response that comes from the Torchserve side and passes it on to the KFServing side.

3) TSModelRepository.py file contains the intialize method for the parameters that gets passed on to the Torchservemodel.py. 

## The Local Testing of KFServing Wrapper for MNIST

Run KFServer locally to test it before creating a docker image. 
Torchserve makes use of port 8085 and the kfserver runs at port 8080
We will hit kfserve , which inturn hit torch serve for inference and explanations request. 
Follow the below steps to serve the MNIST Model :

* Step 1 : Install python3.6.9

* Step 2 : Clone the KFServing Git Repository as below:
```bash
git clone -b master https://github.com/kubeflow/kfserving.git
```

* Step 3 : Install KFserving as below:
```bash
pip install -e ./kfserving/python/kfserving
```

* Step 4 :  Run the Install Dependencies script 
```bash
python ./ts_scripts/install_dependencies.py --environment=dev
```

* Step 5: Run the Install from Source command
```bash
python ./ts_scripts/install_from_src.py
```

* Step 6: Create a directory to place the config.properties in the folder structure below:
```bash
sudo  mkdir -p /mnt/models/config/
```

* Step 7:  Create a directory to place the .mar file in the folder structure below:
```bash
sudo  mkdir -p /mnt/models/model-store
```

* Step 8: Move the model to /mnt/models/model-store

* Step 9: Set service_envelope in config.properties to switch between kfserving v1 and v2 protocols

For v1 protocol

````export TS_SERVICE_ENVELOPE=kfserving```

For v2 protocol

````export TS_SERVICE_ENVELOPE=kfservingv2```

* Step 10: Move the config.properties to /mnt/models/config/.
The config.properties file is as below :
```bash
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8085
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_format=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"mnist":{"1.0":{"defaultVersion":true,"marName":"mnist.mar","minWorkers":1,"maxWorkers":5,"batchSize":5,"maxBatchDelay":200,"responseTimeout":60}}}}
```

* Step 11: Run the below command to start the KFServer
```bash
python3 serve/kubernetes/kfserving/kfserving_wrapper/__main__.py
```

Output:
```bash
[I 201211 18:29:29 __main__:69] Wrapper : Model names ['mnist'], inference address http//0.0.0.0:8085, management address http://0.0.0.0:8085, model store /mnt/models/model-store
[I 201211 18:29:29 TorchserveModel:48] kfmodel Predict URL set to 0.0.0.0:8085
[I 201211 18:29:29 TorchserveModel:50] kfmodel Explain URL set to 0.0.0.0:8085
[I 201211 18:29:29 TSModelRepository:26] TSModelRepo is initialized
[I 201211 18:29:29 kfserver:115] Registering model: mnist
[I 201211 18:29:29 kfserver:96] Listening on port 8080
[I 201211 18:29:29 kfserver:98] Will fork 1 workers
```


* Step 12: Start torchserve using config.properties in /mnt/models/config/
```bash
torchserve --start --ts-config /mnt/models/config/config.properties
```
Please note that Model runs at 8085,KFserving at 8080.The request first comes to the KFServing Wrapper at 8080 in turn requests the torchserve at 8085. So our request should be made at 8080.


For v1 protocol

The curl request for inference is as below:
```bash
curl -H "Content-Type: application/json" --data @serve/kubernetes/kfserving/kf_request_json/mnist.json http://0.0.0.0:8080/v1/models/mnist:predict
```
Output:
```json
{"predictions": [2]}
```

The curl request for explain is as below:
```
curl -H "Content-Type: application/json" --data @serve/kubernetes/kfserving/kf_request_json/mnist.json http://0.0.0.0:8080/v1/models/mnist:explain
```
Output:
```json
{"explanations": [[[[0.004570948726580721,
             ...
             ...
            ]]]]
}
```

For v2 protocol

The curl request for inference is as below:

```bash
curl -H "Content-Type: application/json" --data @serve/kubernetes/kfserving/kf_request_json/mnist_v2.json http://0.0.0.0:8080/v2/models/mnist/infer
```

Response:

```json
{
  "id": "87522347-c21c-4904-9bd3-f6d3ddcc06b0",
  "model_name": "mnist",
  "model_version": "1.0",
  "outputs": [{
    "name": "predict",
    "shape": [1],
    "datatype": "INT64",
    "data": [1]
  }]
}
```

The curl request for explain is as below:

```
curl -H "Content-Type: application/json" --data @serve/kubernetes/kfserving/kf_request_json/mnist.json http://0.0.0.0:8080/v2/models/mnist/explain
```

Response:

```json
{
  "id": "3482b766-0483-40e9-84b0-8ce8d4d1576e",
  "model_name": "mnist",
  "model_version": "1.0",
  "outputs": [{
    "name": "explain",
    "shape": [1, 28, 28],
    "datatype": "FP64",
    "data": [-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0
    ...
    ...
    ]
  }]
}
```
## KFServing Wrapper Testing in Local for BERT

* Step 1: Follow the same steps from to 10 as what was done for MNIST.

* Step 2: Use this config.properties- Change the mode_snaphot to bert
```bash
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8085
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_format=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"bert":{"1.0":{"defaultVersion":true,"marName":"bert.mar","minWorkers":1,"maxWorkers":5,"batchSize":5,"maxBatchDelay":200,"responseTimeout":60}}}}
```

* Step 3: Start the KFServer as below:
```
python3 serve/kubernetes/kfserving/kfserving_wrapper/__main__.py 
```
* Step 4: Start TorchServe:
```
torchserve --start --ts-config /mnt/models/config/config.properties
```

For v1 protocol

The curl request for inference is as below:

```bash
curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/bert.json http://0.0.0.0:8080/v1/models/bert:predict
```

The curl request for Explain is as below:

```bash
curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/bert.json http://0.0.0.0:8080/v1/models/bert:explain
```

For v2 protocol

The curl request for inference is as below:

```bash
curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/bert.json http://0.0.0.0:8080/v2/models/bert/infer
```

The curl request for Explain is as below:

```bash
curl -H "Content-Type: application/json" --data @kubernetes/kfserving/kf_request_json/bert.json http://0.0.0.0:8080/v2/models/bert/explain
```