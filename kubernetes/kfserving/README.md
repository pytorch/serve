## End to End Documentation for Torchserve - KFserving Model Serving

The documentation covers the steps to run Torchserve inside the KFServing environment for the mnist model. 

Currently, KFServing supports the Inference API for all the existing models but text to speech synthesizer and it's explain API works for the eager models of MNIST,BERT and text classification only.

### Docker Image Dev Build

```
DOCKER_BUILDKIT=1 docker build -f Dockerfile.dev -t pytorch/torchserve-kfs:latest-dev .
```

### Docker Image Building

* To create a CPU based image

```
./build_image.sh 
```

* To create a CPU based image with custom tag

```
./build_image.sh -t <repository>/<image>:<tag>
```

* To create a GPU based image

```
./build_image.sh -g 
```

* To create a GPU based image with custom tag

```
./build_image.sh -g -t <repository>/<image>:<tag>
```

Please follow the below steps to deploy Torchserve in Kubeflow Cluster as kfpredictor:

* Step - 1 : Create the .mar file for mnist by invoking the below command :

Run the below command inside the serve folder
```bash
torch-model-archiver --model-name mnist_kf --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
```
For BERT and Text Classifier models, to generate a .mar file refer to the ".mar file creation" section of [BERT Readme file](https://github.com/pytorch/serve/tree/master/kubernetes/kfserving/Huggingface_readme.md#mar-file-creation) and [Text Classifier Readme file](https://github.com/pytorch/serve/tree/master/kubernetes/kfserving/text_classifier_readme.md#mar-file-creation). 


* Step - 2 : Create a config.properties file and place the contents like below:

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
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"<model_name>":{"1.0":{"defaultVersion":true,"marName":"<name of the mar file.>","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}
```


Please note that, the port for inference address should be set at 8085 since KFServing by default makes use of 8080 for its inference service.

When we make an Inference Request,  in Torchserve it makes use of port 8080, whereas on the KFServing side it makes use of port 8085.

Ensure that the KFServing envelope is specified in the config file as shown above. The path of the model store should be mentioned as /mnt/models/model-store because KFServing mounts the model store from that path.


The below sequence of steps need to be executed in the Kubeflow cluster.

* Step - 3 : Create PV, PVC and PV pods in KFServing

Follow the instructions in the link below for creating PV and copying the config files

[Steps for creating PVC](https://github.com/kubeflow/kfserving/blob/master/docs/samples/v1beta1/torchserve/model-archiver/README.md)


* Step - 4 : Create the Inference Service

Refer the following linn to create an inference service

[Creating inference service](https://github.com/kubeflow/kfserving/blob/master/docs/samples/v1beta1/torchserve/README.md#create-the-inferenceservice)

* Step - 5 : Hit the Curl Request to make a prediction as below :

```bash
DEPLOYMENT_NAME=torch-pred
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${DEPLOYMENT_NAME}
 -n kfserving-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v1/models/<model-name>>:predict -d @<path-to-input-file>
```


 * Step - 6 : Hit the Curl Request to make an explanation as below:


```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v1/models/<model-name>>:explain -d @<path-to-input-file>
```

Refer the individual Readmes for KFServing :

* [BERT](https://github.com/pytorch/serve/blob/master/kubernetes/kfserving/Huggingface_readme.md)
* [Text Classifier](https://github.com/pytorch/serve/blob/master/kubernetes/kfserving/text_classifier_readme.md)
* [MNIST](https://github.com/pytorch/serve/blob/master/kubernetes/kfserving/mnist_readme.md)

KFServing supports static batching for prediction - Refer the [link](mnist_readme.md#Static batching:) for an example

For v1 protocol

```json
{
  "inputs": [{
    "name": "a5c32978-fe42-4af0-a1c6-7dded82d12aa",
    "shape": [37],
    "datatype": "INT64",
    "data": [66, 108, 111, 111, 109, 98, 101, 114, 103, 32, 104, 97, 115, 32, 114, 101, 112, 111, 114, 116, 101, 100, 32, 111, 110, 32, 116, 104, 101, 32, 101, 99, 111, 110, 111, 109, 121]
  },
  {
    "name": "a5c32978-fe42-4af0-a1c6-7dded82d12ab",
    "shape": [37],
    "datatype": "INT64",
    "data": [66, 108, 111, 111, 109, 98, 101, 114, 103, 32, 104, 97, 115, 32, 114, 101, 112, 111, 114, 116, 101, 100, 32, 111, 110, 32, 116, 104, 101, 32, 101, 99, 111, 110, 111, 109, 121]
  }]
}
```

For v2 protocol

```json
{
  "inputs": [{
    "name": "a5c32978-fe42-4af0-a1c6-7dded82d12aa",
    "shape": [37],
    "datatype": "INT64",
    "data": [66, 108, 111, 111, 109, 98, 101, 114, 103, 32, 104, 97, 115, 32, 114, 101, 112, 111, 114, 116, 101, 100, 32, 111, 110, 32, 116, 104, 101, 32, 101, 99, 111, 110, 111, 109, 121]
  },
  {
    "name": "a5c32978-fe42-4af0-a1c6-7dded82d12ab",
    "shape": [37],
    "datatype": "INT64",
    "data": [66, 108, 111, 111, 109, 98, 101, 114, 103, 32, 104, 97, 115, 32, 114, 101, 112, 111, 114, 116, 101, 100, 32, 111, 110, 32, 116, 104, 101, 32, 101, 99, 111, 110, 111, 109, 121]
  }]
}
```

For the request and response of BERT and Text Classifier models, refer the "Request and Response" section of section of [BERT Readme file](https://github.com/pytorch/serve/tree/master/kubernetes/kfserving/Huggingface_readme.md#request-and-response) and [Text Classifier Readme file](https://github.com/pytorch/serve/tree/master/kubernetes/kfserving/text_classifier_readme.md#mar-file-creation).



### Troubleshooting guide for KFServing :

1. Check if the pod is up and running :

```bash
kubectl get pods -n kfserving-test
```

2. Check pod events :

```bash
kubectl describe pod <pod-name> -n kfserving-test
```

3. Getting pod logs to track errors :

```bash
kubectl log torch-pred -c kfserving-container -n kfserving-test
```

4. To get the Ingress Host and Port use the following two commands :

```bash
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
```

5. To get the service host by running the following command:

```bash
DEPLOYMENT_NAME=_HOSTNAME=$(kubectl get inferenceservice ${DEPLOYMENT_NAME}
 -n kfserving-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

## Autoscaling
One of the main serverless inference features is to automatically scale the replicas of an `InferenceService` matching the incoming workload.
KFServing by default enables [Knative Pod Autoscaler](https://knative.dev/docs/serving/autoscaling/) which watches traffic flow and scales up and down
based on the configured metrics.

[Autoscaling Example](https://github.com/kubeflow/kfserving/blob/master/docs/samples/v1beta1/torchserve/autoscaling/README.md)

## Canary Rollout
Canary rollout is a deployment strategy when you release a new version of model to a small percent of the production traffic.

[Canary Deployment](https://github.com/kubeflow/kfserving/blob/master/docs/samples/v1beta1/torchserve/canary/README.md)

## Monitoring
[Expose metrics and setup grafana dashboards](https://github.com/kubeflow/kfserving/blob/master/docs/samples/v1beta1/torchserve/metrics/README.md)