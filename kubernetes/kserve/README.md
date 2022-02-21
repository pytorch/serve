## End to End Documentation for Torchserve - KServe Model Serving

The documentation covers the steps to run Torchserve inside the KServe environment for the mnist model.

Currently, KServe supports the Inference API for all the existing models but text to speech synthesizer and it's explain API works for the eager models of MNIST,BERT and text classification only.

### Docker Image Dev Build

```
DOCKER_BUILDKIT=1 docker build -f Dockerfile.dev -t pytorch/torchserve-kfs:latest-dev .
```

### Docker Image Building

- To create a CPU based image

```
./build_image.sh
```

- To create a CPU based image with custom tag

```
./build_image.sh -t <repository>/<image>:<tag>
```

- To create a GPU based image

```
./build_image.sh -g
```

- To create a GPU based image with custom tag

```
./build_image.sh -g -t <repository>/<image>:<tag>
```

### Running Torchserve inference service in KServe cluster

Please follow the below steps to deploy Torchserve in KServe Cluster

- Step - 1 : Create the .mar file for mnist by invoking the below command

Run the below command inside the serve folder

```bash
torch-model-archiver --model-name mnist_kf --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
```

For BERT and Text Classifier models, to generate a .mar file refer to the "Generate mar file" section of [BERT Readme file](kf_request_json/v2/bert/README.md)

- Step - 2 : Create a config.properties file and place the contents like below:

```bash
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8081
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
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"<model_name>":{"1.0":{"defaultVersion":true,"marName":"<name of the mar file.>","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}
```

Please note that, the port for inference address should be set at 8085 since KServe by default makes use of 8080 for its inference service.

When we make an Inference Request, in Torchserve it makes use of port 8080, whereas on the KServe side it makes use of port 8085.

Ensure that the KServe envelope is specified in the config file as shown above. The path of the model store should be mentioned as /mnt/models/model-store because KServe mounts the model store from that path.

The below sequence of steps need to be executed in the KServe cluster.

- Step - 3 : Create PV, PVC and PV pods in KServe

Follow the instructions in the link below for creating PV and copying the config files

[Steps for creating PVC](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/torchserve/model-archiver/README.md)


* Step - 4 : Create the Inference Service

Refer the following linn to create an inference service

[Creating inference service](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/torchserve/README.md#create-the-inferenceservice)

```bash
DEPLOYMENT_NAME=torch-pred
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${DEPLOYMENT_NAME}
 -n KServe-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

* Step - 5 : Generating input files

KServe supports different types of inputs (ex: tensor, bytes). Use the following instructions to generate input files based on its type.

1. Preparing input Section - [MNIST input generation](kf_request_json/v2/mnist/README.md) 
2. Preparing input Section - [Bert input generation](kf_request_json/v2/bert/README.md)


* Step - 6 : Hit the Curl Request to make a prediction as below :

For v1 protocol

```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v1/models/<model-name>:predict -d @<path-to-input-file>
```

For v2 protocol

```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v2/models/<model-name>/infer -d @<path-to-input-file>
```

* Step - 7 : Hit the Curl Request to make an explanation as below:

For v1 protocol

```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v1/models/<model-name>:explain -d @<path-to-input-file>
```

For v2 protocol

```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v2/models/<model-name>/explain -d @<path-to-input-file>
```

Refer the individual Readmes for KServe :

* [BERT](https://github.com/kserve/kserve/tree/master/docs/samples/v1beta1/torchserve/bert#readme)
* [MNIST](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/torchserve/README.md)

Sample input JSON file for v1 and v2 protocols 

For v1 protocol

```json
{
  "instances": [
    {
      "data": "iVBORw0eKGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
    }
  ]
}
```

For v2 protocol

```json
{
  "id": "d3b15cad-50a2-4eaf-80ce-8b0a428bd298",
  "inputs": [{
    "name": "4b7c7d4a-51e4-43c8-af61-04639f6ef4bc",
    "shape": -1,
    "datatype": "BYTES",
    "data": "this year business is good"
  }]
}
```

For the request and response of BERT and Text Classifier models, refer the "Request and Response" section of section of [BERT Readme file](kf_request_json/v2/bert/README.md).

### Troubleshooting guide for KServe :

1. Check if the pod is up and running :

```bash
kubectl get pods -n kserve-test
```

2. Check pod events :

```bash
kubectl describe pod <pod-name> -n kserve-test
```

3. Getting pod logs to track errors :

```bash
kubectl log torch-pred -c kserve-container -n kserve-test
```

4. To get the Ingress Host and Port use the following two commands :

```bash
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
```

5. To get the service host by running the following command:

```bash
DEPLOYMENT_NAME=_HOSTNAME=$(kubectl get inferenceservice ${DEPLOYMENT_NAME}
 -n kserve-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

## Autoscaling
One of the main serverless inference features is to automatically scale the replicas of an `InferenceService` matching the incoming workload.
KServe by default enables [Knative Pod Autoscaler](https://knative.dev/docs/serving/autoscaling/) which watches traffic flow and scales up and down
based on the configured metrics.

[Autoscaling Example](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/torchserve/autoscaling/README.md)

## Canary Rollout
Canary rollout is a deployment strategy when you release a new version of model to a small percent of the production traffic.

[Canary Deployment](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/torchserve/canary/README.md)

## Monitoring
[Expose metrics and setup grafana dashboards](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/torchserve/metrics/README.md)
