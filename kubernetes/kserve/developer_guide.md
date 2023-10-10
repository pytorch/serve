# Developer Guide

The documentation covers the steps to run Torchserve along with the KServe for the mnist model in a local machine without kubernetes. This serves the purpose of developing and debugging Kserve wrapper, service envelope for Torchserve.

## Prerequisites

Install dependencies

```bash
pip install torchserve torch-model-archiver kserve
```

## Steps to run Torchserve with Kserve

### Generating marfile and config.properties file

Navigate to the cloned serve repo and run

```bash
torch-model-archiver --model-name mnist_kf --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
```

The above command generates mnist_kf.mar

Copy the below contents to config.properties and change the model_store path

```bash
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
metrics_mode=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
model_store=<path-to-the-mar-file>
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"mnist_kf":{"1.0":{"defaultVersion":true,"marName":"mnist_kf.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}
```

### Running Torchserve

Enable KServe service envelope

```bash
# For v1 protocol
export TS_SERVICE_ENVELOPE=kserve
# For v2 protocol
export TS_SERVICE_ENVELOPE=kservev2
```

Start Torchserve

```bash
torchserve --start --ts-config config.properties
```

### Running KServe

Start KServe

Navigate to serve/kubernetes/kserve/kserve_wrapper

```bash
python __main__.py
```

### Make inference request

```bash
# v1 protocol
cd serve/kubernetes/kserve/kf_request_json/v1
# Infer request
curl http://localhost:8080/v1/models/mnist:predict -d @./mnist.json
```

```json
{"predictions": [2]}
```

```bash
# v2 protocol
cd serve/kubernetes/kserve/kf_request_json/v2/mnist
# Infer request
curl http://localhost:8080/v2/models/mnist/infer -d @./mnist_v2_bytes.json
```

```json
{"id": "7a02adc8-e1f2-4218-a2ad-f4a29dfa9b16", "model_name": "mnist", "model_version": "1.0", "outputs": [{"name": "predict", "shape": [], "datatype": "INT64", "data": [0]}]}
```

### Make explain request

```bash
# v1 protocol
cd serve/kubernetes/kserve/kf_request_json/v1
# Explain request
curl http://localhost:8080/v1/models/mnist:explain -d @./mnist.json
```

```json
{"explanations": [[[[0.004570948731989492, 0.006216969640322402,
.
.
0.0036409566641309194, -0.0005390934328924084, 0.0064513327328557446, 0.0027735805355367277, 0.006060840367244276]]]]}
```

```bash
# v2 protocol
cd serve/kubernetes/kserve/kf_request_json/v2/mnist
# Explain request
curl http://localhost:8080/v1/models/mnist/explain -d @./mnist_v2_bytes.json
```

```json
{"id": "67ebd591-9528-4172-8aef-e500bc45112c", "model_name": "mnist", "model_version": "1.0", "outputs": [{"name": "explain", "shape": [1, 28, 28], "datatype": "FP64", "data": [-0.0003991945059908792,
-0.0006029826446001493, -0004584260323458741,
.
.
 -0.00020735189345109017, -0.00043539764473153527, ]}]}
````
