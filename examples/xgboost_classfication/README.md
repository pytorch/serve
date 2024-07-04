# XGBoost Classifier for Iris dataset

This example shows how to serve an XGBoost classifier model using TorchServe.
Here we train a model to classify iris dataset

## Pre-requisites

Train an XGBoost classifier model for Iris dataset

```
pip install -r requirements.txt
python xgboost_train.py
```

results in

```
Model accuracy is 1.0
Saving trained model to iris_model.json
```

## Create model archive

```
mkdir model_store
torch-model-archiver --model-name xgb_iris --version 1.0 --serialized-file iris_model.json --handler xgboost_iris_handler.py --export-path model_store --extra-files index_to_name.json --config-file model-config.yaml -f
```

## Start TorchServe

```
torchserve --start --ncs --model-store model_store --models xgb_iris=xgb_iris.mar --disable-token-auth  --enable-model-api
```

## Inference request

We send a batch of 2 requests
```
curl -X POST http://127.0.0.1:8080/predictions/xgb_iris -T sample_input_2.txt & curl -X POST http://127.0.0.1:8080/predictions/xgb_iris -T sample_input_1.txt
```

results in

```
versicolor setosa
```
