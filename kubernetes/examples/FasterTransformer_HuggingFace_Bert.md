# Faster Transformer HuggingFace Bert example in Kubernetes Torchserve.

Batch inferencing with Transformers faces two challenges

    Large batch sizes suffer from higher latency and small or medium-sized batches this will become kernel latency launch bound.
    Padding wastes a lot of compute, (batchsize, seq_length) requires to pad the sequence to (batchsize, max_length) where difference between avg_length and max_length results in a considerable waste of computation, increasing the batch size worsen this situation.

Faster Transformers (FT) from Nvidia along with Efficient Transformers (EFFT) that is built on top of FT address the above two challenges, by fusing the CUDA kernels and dynamically removing padding during computations. The current implementation from Faster Transformers support BERT like encoder and decoder layers. In this example, we show how to get a Torchsctipted (traced) EFFT variant of Bert models from HuggingFace (HF) for sequence classification and question answering and serve it.

## Using Azure AKS Cluster

[AKS Cluster setup](../AKS/README.md##-TorchServe-on-Azure-Kubernetes-Service-AKS)

## Using AWS EKS Cluster

[EKS Cluster setup](../EKS/README.md#-Torchserve-on-Elastic-Kubernetes-service-EKS)

## Using Google GKE Cluster

[GKE Cluster setup](../GKE/README.md##-TorchServe-on-Google-Kubernetes-Engine-GKE)

Once the cluster and the PVCs are ready, we can generate MAR file.

## Generate Mar file

1. Start a pytorch docker container

```bash
nvidia-docker run --gpus all -it nvcr.io/nvidia/pytorch:21.08-py3 bash &
```

2. Start a session with the container running

```bash
nvidia-docker exec -it <container-id> bashs
```

3. Install dependencies

```bash
apt update
apt install git sudo curl -y
```

4. Build Fast Transformers

```bash
git clone https://github.com/NVIDIA/FasterTransformer.git

cd FasterTransformer

mkdir -p build

cd build

cmake -DSM=75 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON ..  # -DSM = 70 for V100 gpu ------- 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100),

make
```

5. Install transformers

```bash
pip install transformers==2.5.1
```

6. Install Torchserve

```bash
cd /workspace
git clone https://github.com/pytorch/serve.git
cd serve

python ts_scripts/install_dependencies.py --cuda=cu102
pip install torchserve torch-model-archiver torch-workflow-archiver

cp examples/FasterTransformer_HuggingFace_Bert/Bert_FT_trace.py /workspace/FasterTransformer/build/pytorch
```

7. Generate model files

```bash
cd examples/Huggingface_Transformers
python Download_Transformer_models.py

cd /workspace/FasterTransformer/build/

python pytorch/Bert_FT_trace.py --mode sequence_classification --model_name_or_path "/workspace/serve/examples/Huggingface_Transformers/Transformer_model" --tokenizer_name "bert-base-uncased" --batch_size 1 --data_type fp16 --model_type thsext

cd /workspace/serve/examples/Huggingface_Transformers

# make sure to change the ../Huggingface_Transformers/setup_config.json "save_mode":"torchscript" and "FasterTransformer":true

# change the ../Huggingface_Transformers/setup_config.json

{
"model_name":"bert-base-uncased",
"mode":"sequence_classification",
"do_lower_case":true,
"num_labels":"0",
"save_mode":"pretrained",
"max_length":"128",
"captum_explanation":false,
"embedding_name": "bert",
"FasterTransformer":true
}

# add a requirement.txt with transformers==2.5.1

echo "transformers==2.5.1" >> requirements.txt
```

8. Create Mar file

```bash
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file /workspace/FasterTransformer/build/traced_model.pt --handler ../Huggingface_Transformers/Transformer_handler_generalized.py --extra-files "../Huggingface_Transformers/setup_config.json,../Huggingface_Transformers/Seq_classification_artifacts/index_to_name.json,/workspace/FasterTransformer/build/lib/libpyt_fastertransformer.so" --requirements-file requirements.txt
```

9. Exit container

## Copy Mar file from container to local path

```bash
docker cp <container-id>:/workspace/serve/examples/Huggingface_Transformers/BERTSeqClassification.mar ./BERTSeqClassification.mar
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

4. Build image

```bash
DOCKER_BUILDKIT=1 docker build -file Dockerfile -t <image-name> --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:20.12-py3 --build-arg CUDA_VERSION=cu102 .
```

5. Push image

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

2. Create a file curl-format.txt with the content below

```bash
    time_namelookup:    %{time_namelookup}s\n
    time_connect:       %{time_connect}s\n
    time_appconnect:    %{time_appconnect}s\n
    time_pretransfer:   %{time_pretransfer}s\n
    time_redirect:      %{time_redirect}s\n

    time_starttransfer: %{time_starttransfer}s\n
                        ----------\n
    time_total:         %{time_total}s\n
```

3. Create input file

Sample_text_captum_input.txt

```json
{
  "text": "Bloomberg has decided to publish a new report on the global economy.",
  "target": 1
}
```

4. Run inference

curl -w "@curl-format.txt" -o /dev/null -s -X POST http://127.0.0.1:8080/predictions/bert -T Seq_classification_artifacts/sample_text_captum_input.txt

Output:

```bash
time_namelookup:    0.000021s
time_connect:       0.000078s
time_appconnect:    0.000000s
time_pretransfer:   0.000096s
time_redirect:      0.000000s
time_starttransfer: 0.000922s
                    ----------
time_total: 0.007053s
```
