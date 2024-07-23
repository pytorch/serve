
# TorchRec DLRM Example

This example shows how to serve a Deep Learning Recommendation Model (DLRM) with TorchRec on a single GPU (CPU is currently not yet supported by this example). It requires at least 15GB of free GPU memory to run.
The DLRM is an open source model for personalization and recommendation use cases published by Meta. More information can be found in this [paper](https://arxiv.org/abs/1906.00091) and this [blog post](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/).
TorchRec is Meta's open source library for recommender systems in Pytorch. More information on TorchRec can be found in the [official docs](https://pytorch.org/torchrec/).

## Install torchrec using the following command
This needs pytorch 1.12 to work

```bash
pip install torchrec==0.2.0
```


--- **This example requires CUDA version >= 11.3**  ---

In this example we will first create and archive the DLRM into a mar file which is subsequently registered in a TorchServe instance. Finally, we run inference using curl.

To create the model and archive it we simple need to run

```
python create_dlrm_mar.py
```

This will instantiate the model and save the state_dict into dlrm.pt which is then used by the model-archiver to create the mar file.
To register the model we need to move the mar file into the model_store folder of our choice.

```
mkdir model_store
mv dlrm.mar model_store
```

Then we can start TorchServe with:

```
torchserve --start --model-store model_store --models dlrm=dlrm.mar --disable-token-auth --enable-model-api
```

To query the model we can then run:

```
curl -H "Content-Type: application/json" --data @sample_data.json http://127.0.0.1:8080/predictions/dlrm
```

```sample_data.json``` is an example of input in the format of criteo [dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).
This has 13 integer features(dense) and 26 categorical features(sparse)

The output of the model is the log odds of a click ( based on the features) More details can be found [here](https://github.com/facebookresearch/dlrm).

The output should look like this:
```
{
    "score":  0.1051536425948143
}
```

## TorchRec DLRM with TorchServe dynamic batching


We start TorchServe with:

```
torchserve --start --model-store model_store --disable-token-auth  --enable-model-api
curl -X POST "localhost:8081/models?model_name=dlrm&url=dlrm.mar&batch_size=4&max_batch_delay=5000&initial_workers=1&synchronous=true"
```

The above commands will create the mar file and register the dlrm model with torchserve with following configuration :

 - model_name : dlrm
 - batch_size : 4
 - max_batch_delay : 5000 ms
 - workers : 1

To test batch inference execute the following commands within the specified max_batch_delay time :

```bash
curl -H "Content-Type: application/json" --data @sample_data.json http://127.0.0.1:8080/predictions/dlrm & curl -H "Content-Type: application/json" --data @sample_data.json http://127.0.0.1:8080/predictions/dlrm & curl -H "Content-Type: application/json" --data @sample_data.json http://127.0.0.1:8080/predictions/dlrm
```

The output looks like this
```
{
  "score": 0.010169986635446548
}{
  "score": 0.010169986635446548
}{
  "score": 0.010169986635446548
}
```
