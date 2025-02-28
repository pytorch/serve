# ⚠️ Notice: Limited Maintenance

This project is no longer actively maintained. While existing releases remain available, there are no planned updates, bug fixes, new features, or security patches. Users should be aware that vulnerabilities may not be addressed.

# Text generation with GPT Fast using KServe

[GPT Fast](https://github.com/pytorch-labs/gpt-fast) is a PyTorch native solution of optimized GPT models. We are using GPT Fast version of `llama2-7b-chat-hf`.
In this example, we show how to serve GPT fast version of Llama 2 with KServe
We will be serving the model using KServe deployed using [minikube](https://minikube.sigs.k8s.io/docs/start/) on a single instance. The same solution can be extended to Kubernetes solutions of various cloud providers

The inference service returns the text generated for the given prompt.

## KServe Image

Before we setup the infrastructure, we need the correct docker image for running this example.
Currently, GPT-Fast needs PyTorch >=2.2 nightlies to run. The nightly image published by TorchServe doesn't include this. Hence, we need to build a custom image.

#### Build custom KServe image

The Dockerfile takes the nightly TorchServe KServe image and installs PyTorch nightlies on top of that.

If your username for dockerhub is `abc`, use the following command
```
docker build . -t abc/torchserve-kfs-nightly:latest-gpu
```

#### Publish KServe image

Make sure you have logged in your account using `docker login`

```
docker push abc/torchserve-kfs-nightly:latest-gpu
```

### GPT-Fast model archive

You can refer to the following [link](https://github.com/pytorch/serve/blob/master/examples/large_models/gpt_fast/README.md) to create torchserve model archive of GPT-Fast.

You would need to publish the config & the model-store to an accessible bucket.

Now we are ready to start deploying the published model.

## Install KServe

Start minikube cluster

```
minikube start --gpus all
```

Install KServe locally.
```
curl -s "https://raw.githubusercontent.com/kserve/kserve/release-0.11/hack/quick_install.sh" | bash
```

Make sure KServe is installed on minikube cluster using

```
kubectl get pods -n kserve
```

This should result in
```
NAME                                         READY   STATUS    RESTARTS   AGE
kserve-controller-manager-57574b4878-rnsjn   2/2     Running   0          17s
```

TorchServe supports KServe V1 and V2 protocol. We show how to deploy with v1 for GPT-Fast

## KServe V1 protocol

Deploy `InferenceService` with KServe V1 protocol

```
kubectl apply -f llama2_v1_gpu.yaml
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
MODEL_NAME=gpt_fast
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

#### Model loading

Once the pod is up, the model loading can take some time in case of large models. We can monitor the `ready` state to determine when the model is loaded.
You can use the following command to get the `ready` state.

```
curl -v -H "Content-Type: application/json" -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}
```

#### Inference request
Make an inference request

```
curl -H "Content-Type: application/json" -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d @.sample_text.json
```

Expected output is

```
{"predictions":["is Paris. It is located in the northern central part of the country and is known for its stunning architecture, art museums, fashion, and historical landmarks. The city is home to many famous landmarks such as the Eiffel Tower"]}
```


## Stop and Delete the cluster

```
minikube stop
minikube delete
```
