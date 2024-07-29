```bash
python ../Download_model.py --model_path model --model_name meta-llama/Meta-Llama-3-8B-Instruct
```

```
torch-model-archiver --model-name llama3-8b-instruct --version 1.0 --handler ../../large_models/Huggingface_accelerate/llama/custom_handler.py --config-file llama-config.yaml -r ../../large_models/Huggingface_accelerate/llama/requirements.txt --archive-format no-archive
```


```bash
mkdir model_store
mv llama3-8b-instruct model_store
mv model model_store/llama3-8b-instruct
```

```bash
torchserve --start --ncs --ts-config ../../large_models/Huggingface_accelerate/llama/config.properties --model-store model_store --models llama3-8b-instruct
```

```bash
curl  "http://localhost:8080/predictions/llama3-8b-instruct" -T ../../large_models/Huggingface_accelerate/llama/sample_text_rag_2.txt
```

```
mkdir model_store
torch-model-archiver --model-name rag --version 1.0 --handler rag_handler.py --config-file rag-config.yaml -r requirements.txt --archive-format no-archive --export-path model_store -f
```
