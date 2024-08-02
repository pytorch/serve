# Deploy Llama & RAG using TorchServe

## Contents
* [Deploy Llama](#deploy-llama)
    * [Download Llama](#download-model)
    * [Generate MAR file](#generate-mar-file)
    * [Add MAR to model store](#add-the-mar-file-to-model-store)
    * [Start TorchServe](#start-torchserve)
    * [Query Llama](#query-llama)
* [Deploy RAG](#deploy-rag)
    * [Download embedding model](#download-embedding-model)
    * [Generate MAR file](#generate-mar-file-1)
    * [Add MAR to model store](#add-the-mar-file-to-model-store-1)
    * [Start TorchServe](#start-torchserve-1)
    * [Query Llama](#query-rag)
* [End-to-End](#)

### Deploy Llama

### Download Llama

Follow [this instruction](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) to get permission

Login with a Hugging Face account
```
huggingface-cli login
# or using an environment variable
huggingface-cli login --token $HUGGINGFACE_TOKEN

```bash
python ../../large_models/Huggingface_accelerate/Download_model.py --model_path model --model_name meta-llama/Meta-Llama-3-8B-Instruct
```
Model will be saved in the following path, `model/models--meta-llama--Meta-Llama-3-8B-Instruct`.

### Generate MAR file

Add the downloaded path to " model_path:" in `model-config.yaml` and run the following.

```
torch-model-archiver --model-name llama3-8b-instruct --version 1.0 --handler ../../large_models/Huggingface_accelerate/llama/custom_handler.py --config-file llama-config.yaml -r ../../large_models/Huggingface_accelerate/llama/requirements.txt --archive-format no-archive
```

### Add the mar file to model store

```bash
mkdir model_store
mv llama3-8b-instruct model_store
mv model model_store/llama3-8b-instruct
```

###  Start TorchServe

```bash
torchserve --start --ncs --ts-config ../../large_models/Huggingface_accelerate/llama/config.properties --model-store model_store --models llama3-8b-instruct --disable-token-auth --enable-model-api
```
### Query Llama

```bash
python query_llama.py
```

### Deploy RAG

### Download embedding model

```
python ../../large_models/Huggingface_accelerate/Download_model.py --model_name sentence-transformers/all-mpnet-base-v2
```
Model is download to `model/models--sentence-transformers--all-mpnet-base-v2`

### Generate MAR file

Add the downloaded path to " model_path:" in `rag-config.yaml` and run the following
```
torch-model-archiver --model-name rag --version 1.0 --handler rag_handler.py --config-file rag-config.yaml --extra-files="hf_custom_embeddings.py" -r requirements.txt --archive-format no-archive
```

### Add the mar file to model store

```bash
mkdir -p model_store
mv rag model_store
mv model model_store/rag
```

### Start TorchServe
```
torchserve --start --ncs --ts-config config.properties --model-store model_store --models rag --disable-token-auth --enable-model-api

```

### Query RAG

```bash
python query_rag.py
```

### RAG + LLM

Send the query to RAG to get the context, send the response to Llama to get more accurate results

```bash
python query_rag_llama.py
```
