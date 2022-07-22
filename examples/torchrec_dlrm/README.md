
### TorchRec DLRM Example

This example shows how to serve a Deep Learning Recommendation Model (DLRM) with TorchRec on a single GPU (CPU is currently not yet supported by this example).
The DLRM is an open source model for personalization and recommendation use cases published by Meta. More informatino can be found in this [paper](https://arxiv.org/abs/1906.00091) and this [blog post](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/).
TorchRec is Meta's open source library for recommender systems in Pytorch. More information on TorchRec can be found in the [official docs](https://pytorch.org/torchrec/).

In this example we will first create and archive the DLRM into a mar file which is subsequently registered in a TorchService instance. Finally, we run an inferent using curl.

To create the model and archive it we simple need to run

```
python create_dlrm_mar.py
```

This will instanciate the model and save the state_dict into dlrm.pt which is then used by the model-archiver to create the mar file.
To register the model we need to move the mar file into the model_store folder of our choice.

```
mkdir model_store
mv dlrm.mar model_store
```

Then we can start TorchServe with:

```
torchserve --start --model-store model_store --models dlrm=dlrm.mar
```

To query the model we can then run:

```
curl -H "Content-Type: application/json" --data @sample_data.json http://127.0.0.1:8080/predictions/dlrm
```

Out output should look like this:
```
{
    "default": [
        0.1051536425948143
    ]
}
```