# Serving Transformers Bert using TorchServe

#### Prerequisite:

```
pip install transformers
pip install torch torchtext 
pip install torchserve torch-model-archiver
```

```
git clone https://github.com/HamidShojanazeri/serve.git

```

#### Preparing Serialized file for torch-model-archiver:

```
cd serve/examples/text_classification
python bert/bert_serialization.py # outputs the jit.traced model "traced_bert.pt".
mv traced_bert.pt bert/
```

#### Archive the model:

```
torch-model-archiver --model-name BertSeqClassification --version 1.0 --serialized-file bert/traced_bert.pt --handler bert/bert_handler.py --extra-files ./index_to_name.json
```

```
mkdir model_store
mv BertSeqClassification.mar model_store/
```

#### Start TorchServe to serve the model:

```
torchserve --start --model-store model_store --models my_tc=BertSeqClassification.mar
```

#### Get predictions from a model:

```
curl http://127.0.0.1:8080/predictions/my_tc -T ./sample_text.txt
```
