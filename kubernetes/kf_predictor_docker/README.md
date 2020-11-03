## Contents of this Document

* [Prerequisites](#prerequisites)
* [End to End Documentation for Torchserve - KFserving Model Serving](#End to End Documentation for Torchserve - KFserving Model Serving)

# Prerequisites

* docker - Refer to the [official docker installation guide](https://docs.docker.com/install/)
* git    - Refer to the [official git set-up guide](https://help.github.com/en/github/getting-started-with-github/set-up-git)
* For base Ubuntu with GPU, install following nvidia container toolkit and driver- 
  * [Nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker#ubuntu-160418042004-debian-jessiestretchbuster)
  * [Nvidia driver](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)


# End to End Documentation for Torchserve - KFserving Model Serving

The documentation covers the steps to run Torchserve inside the KFServing environment for the mnist model. 

* Step - 1 : Create the .mar file for mnist by invoking the below command :
'''
torch-model-archiver --model-name mnist_kf --version 1.0 --model-file serve/examples/image_classifier/mnist/mnist.py --serialized-file serve/examples/image_classifier/mnist/mnist_cnn.pt --handler  serve/examples/image_classifier/mnist/mnist_handler.py
'''

* Step - 2 : Create a docker image for the Torchserve Repo. The dockerfile is located in serve/kubernetes/kf_predictor_docker as Dockerfile_kf.dev. Use the below command to create the docker image :

'''
DOCKER_BUILDKIT=1 docker build --no-cache --file Dockerfile_kf.dev -t <docker image name> .
'''

The KFServing wrapper will be started along with Torchserve inside the image.

* Step - 3 : Push the docker image to the docker registry that you can access from. 

* Step - 4 : Create a config.properties file and place the contents like below:

'''
 inference_address=http://0.0.0.0:8085
 management_address=http://0.0.0.0:8081
 number_of_netty_threads=4
 service_envelope=kfserving
 job_queue_size=10
 model_store=/mnt/models/model-store model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"<model_name>":{"1.0":{"defaultVersion":true,"marName":"<name of the mar file.>","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}
'''



Please note that, the port for inference address should be set at 8085 since KFServing by default makes use of 8080 for its inference service.

When we make an Inference Request,  in Torchserve it makes use of port 8080, whereas on the KFServing side it makes use of port 8085.

Ensure that the KFServing envelope is specified in the config file as shown above. The path of the model store should be mentioned as /mnt/models/model-store because KFServing.

* Step - 5 : KFServing makes use of Transformer, hence a docker image for KFServing needs to be created.  Preprocess functions in kf_handler take in list as data as opposed to the bytes array in the previous Mnist handler. The list of data is received from a kfserving image transformer. This is done as kfserving does not allow a bytes array in the request input. Due to the above reason the Mnist_kf_handler converts the list to Tensor,while the original mnist handler converts bytes array to tensor.

* Step - 6 : The dockerfile for the Image Transformer is located inside the serve/kubernetes/image_transformer folder.  Create the docker image for the Image Transformer using the below command :
'''
docker build -t <image_name>:<tag> -f transformer.Dockerfile .
'''

* Step - 7 : Push the docker image of the Image Transformer to a hub that you can access it from. 

* Step - 8 : Create PVC and PV pods in KFServing. You need to create a local storage PVC and PV pod. You can see below the examples of the pvc.yaml and pv_pod.yaml

The pvc.yaml looks like below:
'''
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
 name: model-store-claim
 labels:
   type: local
spec:
 resources:
   requests:
     storage: 700Mi
 accessModes:
   - ReadWriteOnce
 hostPath:
   path: "/mnt/data"
'''

The pv_pod.yaml looks like below:

'''
apiVersion: v1
kind: Pod
metadata:
 name: model-store-pod
spec:
 volumes:
   - name: model-store
     persistentVolumeClaim:
       claimName: model-store-claim
 containers:
   - name: model-store
     image: ubuntu
     command: [ "sleep" ]
     args: [ "infinity" ]
     volumeMounts:
       - mountPath: "/pv/"
         name: model-store
'''


* Step - 9 : Copy the Model Files and Configure properties.
 
First, create the model store and the config directory using the below command :
'''
kubectl exec -t model-store-pod -c model-store -- mkdir /pv/model-store/
kubectl exec -t model-store-pod -c model-store -- mkdir /pv/config/
'''

Now, copy the .mar file that we created in the previous step and the config.properties with the commands below:

'''
kubectl cp mnist.mar model-store-pod:/pv/model-store/mnist.mar -c model-store
kubectl cp config.properties model-store-pod:/pv/config/config.properties -c model-store
'''

* Step - 10 : Create the Inference Service

This step is for deployment of the Inference Service using the server. For the Image Classification task alone Image Transformer needs to be specified in the inference service yaml file. 

CPU Deployment : For deployment in CPU the sample yaml file is shown as below 

ts-sample.yaml 
'''
apiVersion: "serving.kubeflow.org/v1beta1"
kind: "InferenceService"
metadata:
  name: "torch-pred"
spec:
  transformer:
    containers:
      - image: dockerkc1/image-transformer:v1beta
        name: transformer-container
        env:
          - name: STORAGE_URI
            value: "pvc://model-pv-claim"
  predictor:
    pytorch:
      storageUri: "pvc://model-pv-claim"
'''

To deploy the Torchserve Inference Service in CPU use the below command :

'''
kubectl apply -f ts-sample.yaml -n kfserving-test
'''
* Step - 11 : Check if the Inference Service is up and running : 

Use the below command for the check 
'''
kubectl get inferenceservices torch-pred -n kfserving-test
'''

This shows the service is ready for inference:
'''
NAME         URL                                            READY   AGE
torch-pred   http://torch-pred.kfserving-test.example.com   True    39m
'''

* Step - 12 : Hit the Curl Request to make a prediction as below :

'''
curl -v -H "Host: torch-pred.kfserving-test.<instance>.<region>.amazonaws.com" http://<instance>.<region>amazonaws.com/v1/models/mnist_kf:predict -d @./input.json
'''

The JSON Input content is as below :


{
  "instances": [
    {
      "data": "iVBORw0eKGgoAAAANSUhEUgAAABwAAAAcCAAAAABXZoBIAAAAw0lEQVR4nGNgGFggVVj4/y8Q2GOR83n+58/fP0DwcSqmpNN7oOTJw6f+/H2pjUU2JCSEk0EWqN0cl828e/FIxvz9/9cCh1zS5z9/G9mwyzl/+PNnKQ45nyNAr9ThMHQ/UG4tDofuB4bQIhz6fIBenMWJQ+7Vn7+zeLCbKXv6z59NOPQVgsIcW4QA9YFi6wNQLrKwsBebW/68DJ388Nun5XFocrqvIFH59+XhBAxThTfeB0r+vP/QHbuDCgr2JmOXoSsAAKK7bU3vISS4AAAAAElFTkSuQmCC"
    }
  ]
}


The response is as below :

{
  "predictions": [
    2
  ]
}


### Troubleshooting guide for KFServing :

1. Check if the pod is up and running :

'''
kubectl get pods -n kfserving-test
'''

2. Check pod events :

'''
kubectl describe pod <pod-name> -n kfserving-test
'''

3. Getting pod logs to track errors :

'''
kubectl log torchserve-custom -c kfserving-container -n kfserving-test
'''

4. To get the Ingress Host and Port use the following two commands :

'''
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')

'''

5. To get the service host by running the following command:

'''
MODEL_NAME=torch-pred
SERVICE_HOSTNAME=$(kubectl get route ${MODEL_NAME}-predictor-default
 -n kfserving-test -o jsonpath='{.status.url}' | cut -d "/" -f 3)
'''
