# TorchServe

TorchServe is a flexible and easy to use tool for serving PyTorch models in production.

```bash
curl http://127.0.0.1:8080/predictions/densenet161 -T kitten_small.jpg
```
## ⚡ Why TorchServe
* [Model Management API](docs/management_api.md): multi model management with optimized worker to model allocation
* [Inference API](docs/inference_api.md): REST and gRPC support for batched inference
* [TorchServe Workflows](examples/Workflows/README.md): deploy complex DAGs with multiple interdependent models
Add lots of links
* [Flexible model handlers](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py#L148-L190): use a built-in handler create your own by overriding `preprocess(), inference() and postprocess()`
* Default way to serve PyTorch models in
  * [Kubeflow](https://v0-5.kubeflow.org/docs/components/pytorchserving/)
  * [Sagemaker](https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/)
  * [Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)
* Large coverage for model optimization runtimes like
  * Torchscript
  * [ORT](https://discuss.pytorch.org/t/deploying-onnx-model-with-torchserve/97725/2)
  * [IPEX](https://github.com/pytorch/serve/tree/master/examples/intel_extension_for_pytorch)
  * [TensorRT](https://github.com/pytorch/serve/issues/1243)
  * [FasterTransformer](https://github.com/pytorch/serve/tree/master/examples/FasterTransformer_HuggingFace_Bert)
* Easy to use, extensible and expressive [model handler based design](https://github.com/pytorch/serve/tree/master/examples/) 
* [Metrics API](docs/metrics.md): out of box support for system level metrics with Prometheus exports, custom metrics and PyTorch profiler support


## 🤔 How does TorchServe work
* [Model Server for PyTorch Documentation](docs/README.md): Full documentation
* [Performance Guide](docs/performance_guide.md): For tips and tricks to optimize, benchmark and profile PyTorch and TorchServe performance
* [TorchServe internals](docs/internals.md): How TorchServe was built

### 🏁 Quick start with TorchServe
```
pip install torchserve torch-model-archiver
```

[Getting started guide](docs/getting_started.md)

### 🐳 Quick Start with Docker

```
docker pull pytorch/torchserve
```

Refer to [torchserve docker](docker/README.md) for details.

## 🏆 Highlighted Examples
* [HuggingFace Transformers](examples/Huggingface_Transformers)
* [MultiModal models with MMF](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition) combining text, audio and video
* [Dual Neural Machine Translation](examples/Workflows/nmt_tranformers_pipeline) for a complex workflow DAG

For [more examples](examples/README.md)

## 🤓 Learn More
https://pytorch.org/serve


## 🫂 Contributing

We welcome all contributions!

To learn more about how to contribute, see the contributor guide [here](https://github.com/pytorch/serve/blob/master/CONTRIBUTING.md).

To file a bug or request a feature, please file a GitHub issue. For filing pull requests, please use the template [here](https://github.com/pytorch/serve/blob/master/pull_request_template.md). Cheers!

## 💖 All Contributors

<a href="https://github.com/pytorch/serve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pytorch/serve" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
## Disclaimer 
This repository is jointly operated and maintained by Amazon, Meta and a number of individual contributors listed in the [CONTRIBUTORS](https://github.com/pytorch/serve/graphs/contributors) file. For questions directed at Meta, please send an email to opensource@fb.com. For questions directed at Amazon, please send an email to torchserve@amazon.com. For all other questions, please open up an issue in this repository [here](https://github.com/pytorch/serve/issues).

*TorchServe acknowledges the [Multi Model Server (MMS)](https://github.com/awslabs/multi-model-server) project from which it was derived*
