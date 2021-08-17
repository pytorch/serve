## End to End Documentation for Torchserve - KFserving Model Serving

The documentation covers the steps to run Torchserve inside the KFServing environment for the mnist model. 

Currently, KFServing supports the Inference API for all the existing models but text to speech synthesizer and it's explain API works for the eager models of MNIST,BERT and text classification only.

### Docker Image Dev Build

```
DOCKER_BUILDKIT=1 docker build -f Dockerfile.dev -t pytorch/torchserve-kfs:latest-dev .
```

### Docker Image Building

For CPU Image
```
docker build -t pytorch/torchserve-kfs:latest .
```

For GPU Image
```
docker build --build-arg BASE_IMAGE=pytorch/torchserve:latest-gpu -t pytorch/torchserve-kfs:latest-gpu .
```

Push image to repository
```
docker push pytorch/torchserve-kfs:latest
```


Individual Readmes for KFServing :

* [BERT](./examples/Huggingface_readme.md)
* [Text Classifier](./examples/text_classifier_readme.md)
* [MNIST](./examples/mnist_readme.md)

Please follow the below steps to deploy Torchserve in Kubeflow Cluster as kfpredictor:

* Step - 1 : Create the .mar file for mnist by invoking the below command :

Run the below command inside the serve folder
```bash
torch-model-archiver --model-name mnist_kf --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
```
For BERT and Text Classifier models, to generate a .mar file refer to the ".mar file creation" section of [BERT Readme file](./examples/Huggingface_readme.md#mar-file-creation) and [Text Classifier Readme file](./examples/text_classifier_readme.md#mar-file-creation). 


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

 You need to Create a volume in EC2 for EBS storage PV, PVC and PV pod. You can see below the examples of the pv.yaml, pvc.yaml and pv_pod.yaml.

pv.yaml:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-pv-volume
  labels:
    type: "amazonEBS"
spec:
  capacity:
    storage: 5Gi
  accessModes:
      - ReadWriteOnce
  awsElasticBlockStore:
    volumeID: {volume-id} #vol-074ea8934f7080df5
    fsType: ext4
```

```bash
kubectl apply -f pv.yaml -n kfserving-test
```

pvc.yaml:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pv-claim
  labels:
    type: "amazonEBS"
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

```bash
kubectl apply -f pvc.yaml -n kfserving-test
```

pv_pod.yaml:

```yaml
apiVersion: v1
kind: Pod
metadata:
 name: model-store-pod
spec:
 volumes:
   - name: model-store
     persistentVolumeClaim:
       claimName: model-pv-claim
 containers:
   - name: model-store
     image: ubuntu
     command: [ "sleep" ]
     args: [ "infinity" ]
     volumeMounts:
       - mountPath: "/pv/"
         name: model-store
```

```bash
kubectl apply -f pv_pod.yaml -n kfserving-test
```


* Step - 4 : Copy the Model Files and Config Properties.
 
First, create the model store and the config directory using the below command :
```bash
kubectl exec -t model-store-pod -c model-store -n kfserving-test -- mkdir /pv/model-store/
kubectl exec -t model-store-pod -c model-store -n kfserving-test -- mkdir /pv/config/
```

Now, copy the .mar file that we created in the previous step and the config.properties with the commands below:

```bash
kubectl cp mnist.mar model-store-pod:/pv/model-store/mnist.mar -c model-store -n kfserving-test 
kubectl cp config.properties model-store-pod:/pv/config/config.properties -c model-store -n kfserving-test 
```

* Step - 5 : Create the Inference Service

CPU Deployment : For deployment in CPU the sample yaml file is shown as below 

ts-sample.yaml 

```yaml
apiVersion: "serving.kubeflow.org/v1beta1"
kind: "InferenceService"
metadata:
  name: "torch-pred"
spec:
  predictor:
    pytorch:
      protocol: v1  # Set protocol to v2 for kfserving v2 protocol 
      storageUri: "pvc://model-pv-claim"
```

To deploy the Torchserve Inference Service in CPU use the below command :

```bash
kubectl apply -f ts-sample.yaml -n kfserving-test
```

* Step - 6 : Check if the Inference Service is up and running : 

Use the below command for the check 
```bash
kubectl get inferenceservice torch-pred -n kfserving-test
```

This shows the service is ready for inference:
```bash
NAME         URL                                            READY   AGE
torch-pred   http://torch-pred.kfserving-test.example.com   True    39m
```
* For v1 protocol

* The Curl Request to make a prediction as below :

Navigate to serve/kubernetes/kfserving/kf_request_json

The image file can be converted into string of bytes array by running  
``` 
python v1/img2bytearray.py <imagefile>
```

The JSON Input content is as below :

```json
{
  "instances": [
    {
      "data": "iVBORw0eKGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
    }
  ]
}
```

```bash
DEPLOYMENT_NAME=torch-pred
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${DEPLOYMENT_NAME}
 -n kfserving-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v1/models/mnist:predict -d @./input.json
```

The response is as below :

```json
{
  "predictions": [
    2
  ]
}
```

 * The Curl Request to make an explanation as below:


```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v1/models/mnist:explain -d @./input.json
```

The JSON Input content is as below :

```json
{
  "instances": [
    {
      "data": "iVBORw0eKGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
    }
    
  ]
}
```

The response is as below :
```json
{
  "explanations": [
    [
      [
        [
          0.004570948731989492,
          0.006216969640322402,
          0.008197565423679522,
          0.009563574612830427,
          0.008999274832810742,
          0.009673474804303854,
          0.007599905146155397,
          ,
          ,
        ]
      ]
    ]
  ]
}
```

* For v2 protocol

* The Curl Request to make a prediction as below :

Navigate to serve/kubernetes/kfserving/kf_request_json

The input image and text files can be converted to tensor using totensor.py file

``` 
python v2/totensor.py <inputfile>
```

The JSON Input content is as below :

```json
{
  "inputs": [{
    "name": "a5c32978-fe42-4af0-a1c6-7dded82d12aa",
    "shape": [37],
    "datatype": "INT64",
    "data": [66, 108, 111, 111, 109, 98, 101, 114, 103, 32, 104, 97, 115, 32, 114, 101, 112, 111, 114, 116, 101, 100, 32, 111, 110, 32, 116, 104, 101, 32, 101, 99, 111, 110, 111, 109, 121]
  }]
}
```

```bash
DEPLOYMENT_NAME=torch-pred
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${DEPLOYMENT_NAME}
 -n kfserving-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v1/models/mnist:predict -d @./input.json
```

The response is as below :

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

 * The Curl Request to make an explanation as below:


```bash
curl -v -H "Host: ${SERVICE_HOSTNAME}" http://<instance>.<region>amazonaws.com/v1/models/mnist:explain -d @./input.json
```

The JSON Input content is as below :

```json
{
  "inputs": [{
    "name": "a5c32978-fe42-4af0-a1c6-7dded82d12aa",
    "shape": [37],
    "datatype": "INT64",
    "data": [66, 108, 111, 111, 109, 98, 101, 114, 103, 32, 104, 97, 115, 32, 114, 101, 112, 111, 114, 116, 101, 100, 32, 111, 110, 32, 116, 104, 101, 32, 101, 99, 111, 110, 111, 109, 121]
  }]
}
```

The response is as below :
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

KFServing supports Static batching by adding new examples in the instances key of the request json.
But the batch size should still be set at 1, when we register the model. Explain doesn't support batching. 

For v1 protocol

```json
{
  "instances": [
    {
      "data": "iVBORw0eKGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
    },
    {
      "data": "iVBORw0eKGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
    }
  ]
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

For the request and response of BERT and Text Classifier models, refer the "Request and Response" section of section of [BERT Readme file](./examples/Huggingface_readme.md#request-and-response) and [Text Classifier Readme file](./examples/text_classifier_readme.md#mar-file-creation).


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
