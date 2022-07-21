
### TorchRec DLRM Example

This example shows how to serve a Deep Learning Recommendation Model (DLRM) model with TorchRec on a single GPU (CPU is currently not yet supported by this example).
The DLRM model is an open source model for personalization and recommender use cases published by Meta. More informatino can be found in this [paper](https://arxiv.org/abs/1906.00091) and this [blog post](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/).
TorchRec is Meta's open source library for recommender systems in Pytorch. More information on TorchRec can be found in the [official docs](https://pytorch.org/torchrec/).
Creates an DLRM model with TorchRec and runs an inference.

Create model:

```
python dlrm_single_gpu.py --num_embeddings_per_feature "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35"
```

Load model and run inference :
```
python dlrm_inference.py  --num_embeddings_per_feature "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35"
```
