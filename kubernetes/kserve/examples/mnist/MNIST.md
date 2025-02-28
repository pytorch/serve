# ⚠️ Notice: Limited Maintenance

This project is no longer actively maintained. While existing releases remain available, there are no planned updates, bug fixes, new features, or security patches. Users should be aware that vulnerabilities may not be addressed.

# Digit recognition model with MNIST dataset using a Kubernetes cluster

In this example, we show how to use a pre-trained custom MNIST model to perform real time Digit recognition with TorchServe.
We will be serving the model using Kserve deployed using [minikube](https://minikube.sigs.k8s.io/docs/start/).

The inference service would return the digit inferred by the model in the input image.


## Install kserve

Start minikube cluster

```
minikube start
```

For this example, we need to git clone [kserve](https://github.com/kserve/kserve)
Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/kserve, run the steps from /home/my_path/kserve

Run the following for quick install of kserve
```
./hack/quick_install.sh
```

Make sure kserve is installed on minikube cluster using

```
kubectl get pods -n kserve
```

This should result in
```
NAME                                         READY   STATUS    RESTARTS   AGE
kserve-controller-manager-57574b4878-rnsjn   2/2     Running   0          17s
```

TorchServe supports KServe V1 and V2 protocol. We show how to deploy with both for Mnist.

## KServe V1 protocol

Deploy `InferenceService` with Kserve V1 protocol

```
kubectl apply -f docs/samples/v1beta1/torchserve/v1/torchserve.yaml
```

results in

```
inferenceservice.serving.kserve.io/torchserve created
```

We  need to wait till the pod is up

```
kubectl get pods
NAME                                                  READY   STATUS    RESTARTS   AGE
torchserve-predictor-00001-deployment-8d66f9c-dkdhr   2/2     Running   0          8m19s
```

We need to set the following

```
MODEL_NAME=mnist
SERVICE_HOSTNAME=$(kubectl get inferenceservice torchserve -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

```
export INGRESS_HOST=localhost
export INGRESS_PORT=8080
```

```
INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80 &
```

Make an inference request

```
curl -H "Content-Type: application/json" -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @./docs/samples/v1beta1/torchserve/v1/mnist.json
```

Expected output is

```
{"predictions":[2]}
```

## KServe V2 protocol

Deploy `InferenceService` with Kserve V2 protocol

```
kubectl apply -f docs/samples/v1beta1/torchserve/v2/mnist.yaml
```

results in

```
inferenceservice.serving.kserve.io/torchserve-mnist-v2 created
```

We  need to check the pod is running with

```
kubectl get pods
NAME                                                              READY   STATUS    RESTARTS   AGE
torchserve-mnist-v2-predictor-00001-deployment-6c8c684dcb-4mfmr   2/2     Running   0          2m37s
```

Inspecting the logs of the pods to check the version of TorchServe

```
kubectl logs torchserve-mnist-v2-predictor-00001-deployment-6c8c684dcb-4mfmr
Defaulted container "kserve-container" out of: kserve-container, queue-proxy, storage-initializer (init)
WARNING: sun.reflect.Reflection.getCallerClass is not supported. This will impact performance.
2023-10-12T20:50:39,466 [WARN ] main org.pytorch.serve.util.ConfigManager - Your torchserve instance can access any URL to load models. When deploying to production, make sure to limit the set of allowed_urls in config.properties
2023-10-12T20:50:39,468 [INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager - Initializing plugins manager...
2023-10-12T20:50:39,659 [INFO ] main org.pytorch.serve.metrics.configuration.MetricConfiguration - Successfully loaded metrics configuration from /home/venv/lib/python3.9/site-packages/ts/configs/metrics.yaml
2023-10-12T20:50:39,779 [INFO ] main org.pytorch.serve.ModelServer -
Torchserve version: 0.8.2
TS Home: /home/venv/lib/python3.9/site-packages
Current directory: /home/model-server
Temp directory: /home/model-server/tmp
Metrics config path: /home/venv/lib/python3.9/site-packages/ts/configs/metrics.yaml

```

We need to set the following

```
MODEL_NAME=mnist
SERVICE_HOSTNAME=$(kubectl get inferenceservice torchserve-mnist-v2 -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

```
export INGRESS_HOST=localhost
export INGRESS_PORT=8080
```

```
INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80 &
```

Make an inference request with tensor input

```
curl -v -H "Content-Type: application/json" -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer -d @./docs/samples/v1beta1/torchserve/v2/tensor_conv/mnist_v2.json
```

Expected output is

```
{"model_name":"mnist","model_version":null,"id":"d3b15cad-50a2-4eaf-80ce-8b0a428bd298","parameters":null,"outputs":[{"name":"input-0","shape":[1],"datatype":"INT64","parameters":null,"data":[1]}]}
```

## Stop and Delete the cluster

```
minikube stop
minikube delete
```
