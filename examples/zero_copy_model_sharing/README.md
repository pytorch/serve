This examples show how to deploy a model using a zero-copy mechanism which avoid creating copies of the model in each worker process.
The model is created in the first worker and subsequently moved to shared memory and a shared model store whre the subsequent worker can fetch the shared model.
This makes it possible to run more workers on a single machine with less memory requirements as only one copy of the model needs to be created.
Scaling up the number of worker is also faster.

Steps:

* Create MAR file
```bash
torch-model-archiver --model-name bloom --version 1.0 --handler custom_handler.py --extra-files setup_config.json -r requirements.txt
mkdir model_store && mv bloom.mar model_store/
```

* Start torchserve
```bash
torchserve --start --ncs --ts-config config.properties
```

* Run inference
```bash
curl -v "http://localhost:8080/predictions/bloom" -T sample_text.txt
```


BERT example

* Download model:
```bash
python ../Huggingface_Transformers/Download_Transformer_models.py /home/ubuntu/serve/examples/zero_copy_model_sharing/Transformer_model/setup_config.json
```

```bash
torch-model-archiver --model-name bert_sharing --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./Transformer_model/setup_config.json,../Huggingface_Transformers/Seq_classification_artifacts/index_to_name.json"
```
