# Faster Transformer HuggingFace Bert example in Kubernetes Torchserve.

## Overview

This documnet demonstrates, running fast transformers HuggingFace BERT example with Torchserve in kubernetes setup.

Refer: [FasterTransformer_HuggingFace_Bert](../../examples/FasterTransformer_HuggingFace_Bert/README.md#faster-transformer)

## Using Azure AKS Cluster

[AKS Cluster setup](../AKS/README.md##-TorchServe-on-Azure-Kubernetes-Service-AKS)

## Using AWS EKS Cluster

[EKS Cluster setup](../EKS/README.md#-Torchserve-on-Elastic-Kubernetes-service-EKS)

## Using Google GKE Cluster

[GKE Cluster setup](../GKE/README.md##-TorchServe-on-Google-Kubernetes-Engine-GKE)

Once the cluster and the PVCs are ready, we can generate MAR file.

## Generate Mar file

[Follow steps from here to generate MAR file](../../examples/FasterTransformer_HuggingFace_Bert/README.md#how-to-get-a-torchsctipted-traced-efft-of-hf-bert-model-and-serving-it)

## Copy Mar file from container to local path

```bash
docker cp <container-id>:/workspace/serve/examples/FasterTransformer_HuggingFace_Bert/BERTSeqClassification.mar ./BERTSeqClassification.mar
```

## Create config.properties

```bash
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
NUM_WORKERS=1
number_of_gpu=1
install_py_dep_per_model=true
number_of_netty_threads=32
job_queue_size=1000
model_store=/home/model-server/shared/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"bert":{"1.0":{"defaultVersion":true,"marName":"BERTSeqClassification.mar","minWorkers":2,"maxWorkers":3,"batchSize":1,"maxBatchDelay":100,"responseTimeout":120}}}}
```

## Copy Mar file and config.properties to PVC

```bash
kubectl exec --tty pod/model-store-pod -- mkdir /pv/model-store/
kubectl cp BERTSeqClassification.mar model-store-pod:/pv/model-store/BERTSeqClassification.mar

kubectl exec --tty pod/model-store-pod -- mkdir /pv/config/
kubectl cp config.properties model-store-pod:/pv/config/config.properties
```

## Build Torchserve image

1. Clone Torchserve Repo

```bash
git clone https://github.com/pytorch/serve.git
cd serve/docker
```

2. Modify Python and Pip paths in `Dockerfile` as below

```bash
sed -i 's#/usr/bin/python3#/opt/conda/bin/python3#g' Dockerfile
sed -i 's#/usr/local/bin/pip3#/opt/conda/bin/pip3#g' Dockerfile
```

3. Change GPU check in `Dockerfile` for nvcr.io image

```bash
sed -i 's#grep -q "cuda:"#grep -q "nvidia:"#g' Dockerfile
```

4. Add `transformers==2.5.1` to `Dockerfile`

```bash
sed -i 's#pip install --no-cache-dir captum torchtext torchserve torch-model-archiver#& transformers==2.5.1#g' Dockerfile
```

5. Build image

   Refer: [NGC deep learning framework container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)

```bash
DOCKER_BUILDKIT=1 docker build -file Dockerfile -t <image-name> --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:20.12-py3 --build-arg CUDA_VERSION=cu102 .
```

6. Push image

```bash
docker push <image-name>
```

## Install Torchserve

1. Navigate to kubernetes TS Helm package folder

```bash
cd ../kubernetes/Helm
```

2. Modify values.yaml with image and memory

```yaml
torchserve_image: <image build in previous step>

namespace: torchserve

torchserve:
  management_port: 8081
  inference_port: 8080
  metrics_port: 8082
  pvd_mount: /home/model-server/shared/
  n_gpu: 1
  n_cpu: 4
  memory_limit: 32Gi
  memory_request: 32Gi

deployment:
  replicas: 1

persitant_volume:
  size: 1Gi
```

3. Install TS

```bash
helm install torchserve .
```

4. Check TS installation

```bash
Kubectl get pods -n default
Kubectl logs <pod-name> -n default
```

## Run Inference

1. Start a shell session into the TS pod

```bash
kubectl exec -it <pod-name> -- bash
```

2. Create input file

Sample_text_captum_input.txt

```json
{
  "text": "Bloomberg has decided to publish a new report on the global economy.",
  "target": 1
}
```

4. Run inference

```bash
curl -X POST http://127.0.0.1:8080/predictions/bert -T ../Huggingface_Transformers/Seq_classification_artifacts/sample_text_captum_input.txt

```
