# End to End Documentation for Torchserve - KServe Model Serving

The documentation covers the steps to run Torchserve inside the KServe environment for the mnist model.

Currently, KServe supports the Inference API for all the existing models but text to speech synthesizer and it's explain API works for the eager models of MNIST,BERT and text classification only.

## Docker Image Building

- To create a CPU based image

```bash
./build_image.sh
```

- To create a CPU based image with custom tag

```bash
./build_image.sh -t <repository>/<image>:<tag>
```

- To create a GPU based image

```bash
./build_image.sh -g
```

- To create a GPU based image with custom tag

```bash
./build_image.sh -g -t <repository>/<image>:<tag>
```

### Docker Image Dev Build

```bash
DOCKER_BUILDKIT=1 docker build -f Dockerfile.dev -t pytorch/torchserve-kfs:latest-dev .
```

## Running Torchserve inference service in KServe cluster
### Create Kubernetes cluster with eksctl

- Install eksctl - https://docs.aws.amazon.com/eks/latest/userguide/eksctl.html

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: "kserve-cluster"
  region: "us-west-2"

vpc:
  id: "vpc-xxxxxxxxxxxxxxxxx"
  subnets:
    private:
      us-west-2a:
          id: "subnet-xxxxxxxxxxxxxxxxx"
      us-west-2c:
          id: "subnet-xxxxxxxxxxxxxxxxx"
    public:
      us-west-2a:
          id: "subnet-xxxxxxxxxxxxxxxxx"
      us-west-2c:
          id: "subnet-xxxxxxxxxxxxxxxxx"

nodeGroups:
  - name: ng-1
    minSize: 1
    maxSize: 4
    desiredCapacity: 2
    instancesDistribution:
      instanceTypes: ["p3.8xlarge"] # At least one instance type should be specified
      onDemandBaseCapacity: 0
      onDemandPercentageAboveBaseCapacity: 50
      spotInstancePools: 5
```

```bash
eksctl create cluster -f cluster.yaml
```

### Install KServe

Run the below command to install kserve in the cluster.

```bash
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.8/hack/quick_install.sh" | bash
```

This installs the latest kserve in the kubernetes cluster.

- create a test namespace kserve-test

```bash
kubectl create namespace kserve-test
```

### Steps for running Torchserve inference service in KServe

Here we use the mnist example in Torchserve Repository.

- Step - 1 : Create the .mar file for mnist by invoking the below command

Navigate to the cloned serve repo and run

```bash
torch-model-archiver --model-name mnist_kf --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
```

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
metrics_mode=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"mnist_kf":{"1.0":{"defaultVersion":true,"marName":"mnist_kf.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}
```

Please note that, the port for inference address should be set at 8085 since KServe by default makes use of 8080 for its inference service.

- Step - 3 : Create PV, PVC and PV pods in KServe

  For EFS backed volume refer - https://github.com/pytorch/serve/tree/master/kubernetes/EKS#setup-persistentvolume-backed-by-efs


Follow the instructions below for creating a PV and copying the config files

- Create volume

  EBS volume creation: https://docs.aws.amazon.com/cli/latest/reference/ec2/create-volume.html

  For PV and PVC refer: https://kubernetes.io/docs/concepts/storage/persistent-volumes/

- Create PV

Edit volume id in pv.yaml file

```bash
kubectl apply -f ../reference_yaml/pv-deployments/pv.yaml -n kserve-test
```

- Create PVC

```bash
kubectl apply -f ../reference_yaml/pv-deployments/pvc.yaml -n kserve-test
```

- Create pod for copying model store files to PV

```bash
kubectl apply -f ../reference_yaml/pvpod.yaml -n kserve-test
```

- Step - 4 : Copy the config.properties file and mar file to the PVC using the model-store-pod

```bash
# Create directory in PV
kubectl exec -it model-store-pod -c model-store -n kserve-test -- mkdir /pv/model-store/
kubectl exec -it model-store-pod -c model-store -n kserve-test -- mkdir /pv/config/
# Copy files the path
kubectl cp mnist.mar model-store-pod:/pv/model-store/ -c model-store -n kserve-test
kubectl cp config.properties model-store-pod:/pv/config/ -c model-store -n kserve-test
```

Refer link for other [storage options](https://github.com/kserve/kserve/tree/master/docs/samples/storage)

- Step - 5 : Create the Inference Service

```bash
# For v1 protocol
kubectl apply -f ../reference_yaml/torchserve-deployment/v1/ts_sample.yaml -n kserve-test

# For v2 protocol
kubectl apply -f ../reference_yaml/torchserve-deployment/v2/ts_sample.yaml -n kserve-test
```

Refer link for more [examples](https://github.com/kserve/kserve/tree/master/docs/samples/v1beta1/torchserve)

- Step - 6 : Generating input files

KServe supports different types of inputs (ex: tensor, bytes). Use the following instructions to generate input files based on its type.

[MNIST input generation](kf_request_json/v2/mnist/README.md##-Preparing-input)
[Bert input generation](kf_request_json/v2/bert/README.md##-Preparing-input)


- Step - 7 : Hit the Curl Request to make a prediction as below :

```bash
DEPLOYMENT_NAME=torch-pred
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${DEPLOYMENT_NAME} -n KServe-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
```

For v1 protocol

```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/mnist-kf:predict -d @./kf_request_json/v1/mnist/mnist.json
```

For v2 protocol

```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/mnist-kf/infer -d ./kf_request_json/v2/mnist/mnist_v2_bytes.json
```

- Step - 8 : Hit the Curl Request to make an explanation as below:

For v1 protocol

```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/mnist-kf:explain -d ./kf_request_json/v1/mnist/mnist.json
```

For v2 protocol

```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/mnist-kf/explain -d ./kf_request_json/v2/mnist/mnist_v2_bytes.json
```

Refer the individual readmes for KServe :

* [BERT](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/custom/torchserve/bert-sample/hugging-face-bert-sample.md)
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
