# TorchServe

TorchServe is a flexible and easy to use tool for serving and scaling PyTorch models in production.

Requires python > 3.8

```bash
curl http://127.0.0.1:8080/predictions/bert -T input.txt
```
### 🚀 Quick start with TorchServe

```
# Install dependencies
# cuda is optional
python ./ts_scripts/install_dependencies.py --cuda=cu111

# Latest release
pip install torchserve torch-model-archiver torch-workflow-archiver

# Nightly build
pip install torchserve-nightly torch-model-archiver-nightly torch-workflow-archiver-nightly
```

[Getting started guide](docs/getting_started.md)

### 🐳 Quick Start with Docker

```
docker pull pytorch/torchserve
```

Refer to [torchserve docker](docker/README.md) for details.

## ⚡ Why TorchServe
* [Model Management API](docs/management_api.md): multi model management with optimized worker to model allocation
* [Inference API](docs/inference_api.md): REST and gRPC support for batched inference
* [TorchServe Workflows](examples/Workflows/README.md): deploy complex DAGs with multiple interdependent models
* Default way to serve PyTorch models in
  * [Kubeflow](https://v0-5.kubeflow.org/docs/components/pytorchserving/)
  * [MLflow](https://github.com/mlflow/mlflow-torchserve)
  * [Sagemaker](https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/)
  * [Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)
* Export your model for optimized inference
  * Torchscript out of the box
  * [ORT](https://discuss.pytorch.org/t/deploying-onnx-model-with-torchserve/97725/2)
  * [IPEX](https://github.com/pytorch/serve/tree/master/examples/intel_extension_for_pytorch)
  * [TensorRT](https://github.com/pytorch/serve/issues/1243)
  * [FasterTransformer](https://github.com/pytorch/serve/tree/master/examples/FasterTransformer_HuggingFace_Bert)
* [Performance Guide](docs/performance_guide.md): builtin support to optimize, benchmark and profile PyTorch and TorchServe performance
* [Expressive handlers](CONTRIBUTING.md): An expressive handler architecture that makes it trivial to support inferencing for your usecase with [many supported out of the box](https://github.com/pytorch/serve/tree/master/ts/torch_handler)
* [Metrics API](docs/metrics.md): out of box support for system level metrics with [Prometheus exports](https://github.com/pytorch/serve/tree/master/examples/custom_metrics), custom metrics and PyTorch profiler support


## 🤔 How does TorchServe work
* [Model Server for PyTorch Documentation](docs/README.md): Full documentation
* [TorchServe internals](docs/internals.md): How TorchServe was built
* [Contributing guide](CONTRIBUTING.md): How to contribute to TorchServe


## 🏆 Highlighted Examples
* [🤗 HuggingFace Transformers](examples/Huggingface_Transformers)
* [MultiModal models with MMF](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition) combining text, audio and video
* [Dual Neural Machine Translation](examples/Workflows/nmt_transformers_pipeline) for a complex workflow DAG

For [more examples](examples/README.md)

## 🤓 Learn More
https://pytorch.org/serve


## 🫂 Contributing

We welcome all contributions!

To learn more about how to contribute, see the contributor guide [here](https://github.com/pytorch/serve/blob/master/CONTRIBUTING.md).

To file a bug or request a feature, please file a GitHub issue. For filing pull requests, please use the template [here](https://github.com/pytorch/serve/blob/master/pull_request_template.md).

## 📰 News
* [Announcing TorchServe](https://aws.amazon.com/blogs/aws/announcing-torchserve-an-open-source-model-server-for-pytorch/)
* [How to deploy PyTorch models on Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)
* [How to Serve PyTorch Models with TorchServe](https://www.youtube.com/watch?v=XlO7iQMV3Ik)
* [Model Serving in PyTorch](https://www.youtube.com/watch?v=2A17ZtycsPw)
* [Explain Like I’m 5: TorchServe](https://www.youtube.com/watch?v=NEdZbkfHQCk)

## 💖 All Contributors

<a href="https://github.com/pytorch/serve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pytorch/serve" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
## ⚖️ Disclaimer 
This repository is jointly operated and maintained by Amazon, Meta and a number of individual contributors listed in the [CONTRIBUTORS](https://github.com/pytorch/serve/graphs/contributors) file. For questions directed at Meta, please send an email to opensource@fb.com. For questions directed at Amazon, please send an email to torchserve@amazon.com. For all other questions, please open up an issue in this repository [here](https://github.com/pytorch/serve/issues).

*TorchServe acknowledges the [Multi Model Server (MMS)](https://github.com/awslabs/multi-model-server) project from which it was derived*
