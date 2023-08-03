# TorchServe example with Huggingface bert model

In this example we will show how to serve [Huggingface Transformers with TorchServe](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers)
model locally using kserve.

## Model archive file creation

Clone [pytorch/serve](https://github.com/pytorch/serve) repository.

Copy the [Transformer_kserve_handler.py](https://github.com/kserve/kserve/blob/master/docs/samples/v1beta1/torchserve/v2/bert/sequence_classification/Transformer_kserve_handler.py) handler file to `examples/Huggingface_Transformers` folder

Navigate to `examples/Huggingface_Transformers`

Run the following command to download the model

```
python Download_Transformer_models.py
```

### Generate mar file

```bash
torch-model-archiver --model-name BERTSeqClassification --version 1.0 \
--serialized-file Transformer_model/pytorch_model.bin \
--handler ./Transformer_kserve_handler.py \
--extra-files "Transformer_model/config.json,./setup_config.json,./Seq_classification_artifacts/index_to_name.json,./Transformer_handler_generalized.py"
```

The command will create `BERTSeqClassification.mar` file in current directory

Move the mar file to model-store

```
sudo mv BERTSeqClassification.mar /mnt/models/model-store
```

and use the following config properties (`/mnt/models/config`)

```
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8085
metrics_address=http://0.0.0.0:8082
enable_envvars_config=true
install_py_dep_per_model=true
enable_metrics_api=true
service_envelope=kservev2
metrics_mode=prometheus
NUM_WORKERS=1
number_of_netty_threads=4
job_queue_size=10
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"BERTSeqClassification":{"1.0":{"defaultVersion":true,"marName":"BERTSeqClassification.mar","minWorkers":1,"maxWorkers":5,"batchSize":1,"maxBatchDelay":5000,"responseTimeout":120}}}}
```

## Preparing input

Use [bert_bytes_v2.json](bert_bytes_v2.json) or [bert_tensor_v2](bert_tensor_v2.json).

For new sample text, follow the instructions below

For bytes input, use [tobytes](tobytes.py) utility.

```
python tobytes.py --input_text "this year business is good"
```

For tensor input, use [bert_tokenizer](bert_tokenizer.py) utility

```
python bert_tokenizer.py --input_text "this year business is good"
```


## Deploying the model in local machine

Start TorchServe

```
torchserve --start --ts-config /mnt/models/config/config.properties --ncs
```

To test locally, clone TorchServe and move to the following folder `kubernetes/kserve/kserve_wrapper`

Start Kserve

```
python __main__.py
```

## Request and response

### Sample request and response for bytes input

Navigate to `kubernetes/kserve/kf_request_json/v2/bert`

Run the following command

```bash
curl -v -H "ContentType: application/json" http://localhost:8080/v2/models/BERTSeqClassification/infer -d @./bert_bytes_v2.json
```

Expected Output

```bash
{"id": "d3b15cad-50a2-4eaf-80ce-8b0a428bd298", "model_name": "BERTSeqClassification", "model_version": "1.0", "outputs": [{"name": "predict", "shape": [], "datatype": "BYTES", "data": ["Not Accepted"]}]}
```


### Sample request and response for tensor input


Run the following command

```
curl -v -H "ContentType: application/json" http://localhost:8080/v2/models/BERTSeqClassification/infer -d @./bert_tensor_v2.json
```

Expected output
```bash
{"id": "33abc661-7265-42fc-b7d9-44e5f79a7a67", "model_name": "BERTSeqClassification", "model_version": "1.0", "outputs": [{"name": "predict", "shape": [], "datatype": "BYTES", "data": ["Not Accepted"]}]}
```
