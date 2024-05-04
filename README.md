# TorchServe

![Nightly build](https://github.com/pytorch/serve/actions/workflows/torchserve-nightly-build.yml/badge.svg)
![Docker Nightly build](https://github.com/pytorch/serve/actions/workflows/docker-nightly-build.yml/badge.svg)
![Benchmark Nightly](https://github.com/pytorch/serve/actions/workflows/benchmark_nightly.yml/badge.svg)
![Docker Regression Nightly](https://github.com/pytorch/serve/actions/workflows/regression_tests_docker.yml/badge.svg)
![KServe Regression Nightly](https://github.com/pytorch/serve/actions/workflows/kserve_cpu_tests.yml/badge.svg)
![Kubernetes Regression Nightly](https://github.com/pytorch/serve/actions/workflows/kubernetes_tests.yml/badge.svg)

TorchServe is a flexible and easy-to-use tool for serving and scaling PyTorch models in production.

Requires python >= 3.8

```bash
curl http://127.0.0.1:8080/predictions/bert -T input.txt
```
### 🚀 Quick start with TorchServe

```bash
# Install dependencies
# cuda is optional
python ./ts_scripts/install_dependencies.py --cuda=cu121

# Latest release
pip install torchserve torch-model-archiver torch-workflow-archiver

# Nightly build
pip install torchserve-nightly torch-model-archiver-nightly torch-workflow-archiver-nightly
```

### 🚀 Quick start with TorchServe (conda)

```bash
# Install dependencies
# cuda is optional
python ./ts_scripts/install_dependencies.py --cuda=cu121

# Latest release
conda install -c pytorch torchserve torch-model-archiver torch-workflow-archiver

# Nightly build
conda install -c pytorch-nightly torchserve torch-model-archiver torch-workflow-archiver
```

[Getting started guide](docs/getting_started.md)

### 🐳 Quick Start with Docker

```bash
# Latest release
docker pull pytorch/torchserve

# Nightly build
docker pull pytorch/torchserve-nightly
```

Refer to [torchserve docker](docker/README.md) for details.

## ⚡ Why TorchServe
* Write once, run anywhere, on-prem, on-cloud, supports inference on CPUs, GPUs, AWS Inf1/Inf2/Trn1, Google Cloud TPUs, [Nvidia MPS](docs/nvidia_mps.md)
* [Model Management API](docs/management_api.md): multi model management with optimized worker to model allocation
* [Inference API](docs/inference_api.md): REST and gRPC support for batched inference
* [TorchServe Workflows](examples/Workflows/README.md): deploy complex DAGs with multiple interdependent models
* Default way to serve PyTorch models in
  * [Sagemaker](https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/)
  * [Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)
  * [Kubernetes](kubernetes) with support for [autoscaling](kubernetes#session-affinity-with-multiple-torchserve-pods), session-affinity, monitoring using Grafana works on-prem, AWS EKS, Google GKE, Azure AKS
  * [Kserve](https://kserve.github.io/website/0.8/modelserving/v1beta1/torchserve/): Supports both v1 and v2 API, [autoscaling and canary deployments](kubernetes/kserve/README.md#autoscaling) for A/B testing
  * [Kubeflow](https://v0-5.kubeflow.org/docs/components/pytorchserving/)
  * [MLflow](https://github.com/mlflow/mlflow-torchserve)
* Export your model for optimized inference. Torchscript out of the box, [PyTorch Compiler](examples/pt2/README.md) preview, [ORT and ONNX](https://github.com/pytorch/serve/blob/master/docs/performance_guide.md), [IPEX](https://github.com/pytorch/serve/tree/master/examples/intel_extension_for_pytorch), [TensorRT](https://github.com/pytorch/serve/blob/master/docs/performance_guide.md), [FasterTransformer](https://github.com/pytorch/serve/tree/master/examples/FasterTransformer_HuggingFace_Bert), FlashAttention (Better Transformers)
* [Performance Guide](docs/performance_guide.md): builtin support to optimize, benchmark, and profile PyTorch and TorchServe performance
* [Expressive handlers](CONTRIBUTING.md): An expressive handler architecture that makes it trivial to support inferencing for your use case with [many supported out of the box](https://github.com/pytorch/serve/tree/master/ts/torch_handler)
* [Metrics API](docs/metrics.md): out-of-the-box support for system-level metrics with [Prometheus exports](https://github.com/pytorch/serve/tree/master/examples/custom_metrics), custom metrics,
* [Large Model Inference Guide](docs/large_model_inference.md): With support for GenAI, LLMs including
  * [SOTA GenAI performance](https://github.com/pytorch/serve/tree/master/examples/pt2#torchcompile-genai-examples) using `torch.compile`
  * Fast Kernels with FlashAttention v2, continuous batching and streaming response
  * PyTorch [Tensor Parallel](examples/large_models/tp_llama) preview, [Pipeline Parallel](examples/large_models/Huggingface_pippy)
  * Microsoft [DeepSpeed](examples/large_models/deepspeed), [DeepSpeed-Mii](examples/large_models/deepspeed_mii)
  * Hugging Face [Accelerate](examples/large_models/Huggingface_accelerate), [Diffusers](examples/diffusers)
  * Running large models on AWS [Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-tutorials-torchserve.html) and [Inferentia2](https://pytorch.org/blog/high-performance-llama/)
  * Running [Meta Llama Chatbot locally on Mac](examples/LLM/llama)
* Monitoring using Grafana and [Datadog](https://www.datadoghq.com/blog/ai-integrations/#model-serving-and-deployment-vertex-ai-amazon-sagemaker-torchserve)


## 🤔 How does TorchServe work
* [Model Server for PyTorch Documentation](docs/README.md): Full documentation
* [TorchServe internals](docs/internals.md): How TorchServe was built
* [Contributing guide](CONTRIBUTING.md): How to contribute to TorchServe


## 🏆 Highlighted Examples
* [Serving Meta Llama with TorchServe](examples/LLM/llama/README.md)
* [Chatbot with Meta Llama on Mac 🦙💬](examples/LLM/llama/chat_app)
* [🤗 HuggingFace Transformers](examples/Huggingface_Transformers) with a [Better Transformer Integration/ Flash Attention & Xformer Memory Efficient ](examples/Huggingface_Transformers#Speed-up-inference-with-Better-Transformer)
* [Stable Diffusion](examples/diffusers)
* [Model parallel inference](examples/Huggingface_Transformers#model-parallelism)
* [MultiModal models with MMF](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition) combining text, audio and video
* [Dual Neural Machine Translation](examples/Workflows/nmt_transformers_pipeline) for a complex workflow DAG
* [TorchServe Integrations](examples/README.md#torchserve-integrations)
* [TorchServe Internals](examples/README.md#torchserve-internals)
* [TorchServe UseCases](examples/README.md#usecases)

For [more examples](examples/README.md)

## 🛡️ TorchServe Security Policy
[SECURITY.md](SECURITY.md)

## 🤓 Learn More
https://pytorch.org/serve


## 🫂 Contributing

We welcome all contributions!

To learn more about how to contribute, see the contributor guide [here](https://github.com/pytorch/serve/blob/master/CONTRIBUTING.md).

## 📰 News
* [High performance Llama 2 deployments with AWS Inferentia2 using TorchServe](https://pytorch.org/blog/high-performance-llama/)
* [Naver Case Study: Transition From High-Cost GPUs to Intel CPUs and oneAPI powered Software with performance](https://pytorch.org/blog/ml-model-server-resource-saving/)
* [Run multiple generative AI models on GPU using Amazon SageMaker multi-model endpoints with TorchServe and save up to 75% in inference costs](https://pytorch.org/blog/amazon-sagemaker-w-torchserve/)
* [Deploying your Generative AI model in only four steps with Vertex AI and PyTorch](https://cloud.google.com/blog/products/ai-machine-learning/get-your-genai-model-going-in-four-easy-steps)
* [PyTorch Model Serving on Google Cloud TPU v5](https://cloud.google.com/tpu/docs/v5e-inference#pytorch-model-inference-and-serving)
* [Monitoring using Datadog](https://www.datadoghq.com/blog/ai-integrations/#model-serving-and-deployment-vertex-ai-amazon-sagemaker-torchserve)
* [Torchserve Performance Tuning, Animated Drawings Case-Study](https://pytorch.org/blog/torchserve-performance-tuning/)
* [Walmart Search: Serving Models at a Scale on TorchServe](https://medium.com/walmartglobaltech/search-model-serving-using-pytorch-and-torchserve-6caf9d1c5f4d)
* [🎥 Scaling inference on CPU with TorchServe](https://www.youtube.com/watch?v=066_Jd6cwZg)
* [🎥 TorchServe C++ backend](https://www.youtube.com/watch?v=OSmGGDpaesc)
* [Grokking Intel CPU PyTorch performance from first principles: a TorchServe case study](https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html)
* [Grokking Intel CPU PyTorch performance from first principles( Part 2): a TorchServe case study](https://pytorch.org/tutorials/intermediate/torchserve_with_ipex_2.html)
* [Case Study: Amazon Ads Uses PyTorch and AWS Inferentia to Scale Models for Ads Processing](https://pytorch.org/blog/amazon-ads-case-study/)
* [Optimize your inference jobs using dynamic batch inference with TorchServe on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/optimize-your-inference-jobs-using-dynamic-batch-inference-with-torchserve-on-amazon-sagemaker/)
* [Using AI to bring children's drawings to life](https://ai.meta.com/blog/using-ai-to-bring-childrens-drawings-to-life/)
* [🎥 Model Serving in PyTorch](https://www.youtube.com/watch?v=2A17ZtycsPw)
* [Evolution of Cresta's machine learning architecture: Migration to AWS and PyTorch](https://aws.amazon.com/blogs/machine-learning/evolution-of-crestas-machine-learning-architecture-migration-to-aws-and-pytorch/)
* [🎥 Explain Like I’m 5: TorchServe](https://www.youtube.com/watch?v=NEdZbkfHQCk)
* [🎥 How to Serve PyTorch Models with TorchServe](https://www.youtube.com/watch?v=XlO7iQMV3Ik)
* [How to deploy PyTorch models on Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)
* [Quantitative Comparison of Serving Platforms](https://biano-ai.github.io/research/2021/08/16/quantitative-comparison-of-serving-platforms-for-neural-networks.html)
* [Efficient Serverless deployment of PyTorch models on Azure](https://medium.com/pytorch/efficient-serverless-deployment-of-pytorch-models-on-azure-dc9c2b6bfee7)
* [Deploy PyTorch models with TorchServe in Azure Machine Learning online endpoints](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/deploy-pytorch-models-with-torchserve-in-azure-machine-learning/ba-p/2466459)
* [Dynaboard moving beyond accuracy to holistic model evaluation in NLP](https://ai.facebook.com/blog/dynaboard-moving-beyond-accuracy-to-holistic-model-evaluation-in-nlp/)
* [A MLOps Tale about operationalising MLFlow and PyTorch](https://medium.com/mlops-community/engineering-lab-1-team-1-a-mlops-tale-about-operationalising-mlflow-and-pytorch-62193b55dc19)
* [Operationalize, Scale and Infuse Trust in AI Models using KFServing](https://blog.kubeflow.org/release/official/2021/03/08/kfserving-0.5.html)
* [How Wadhwani AI Uses PyTorch To Empower Cotton Farmers](https://medium.com/pytorch/how-wadhwani-ai-uses-pytorch-to-empower-cotton-farmers-14397f4c9f2b)
* [TorchServe Streamlit Integration](https://cceyda.github.io/blog/huggingface/torchserve/streamlit/ner/2020/10/09/huggingface_streamlit_serve.html)
* [Dynabench aims to make AI models more robust through distributed human workers](https://venturebeat.com/2020/09/24/facebooks-dynabench-aims-to-make-ai-models-more-robust-through-distributed-human-workers/)
* [Announcing TorchServe](https://aws.amazon.com/blogs/aws/announcing-torchserve-an-open-source-model-server-for-pytorch/)

## 💖 All Contributors

<a href="https://github.com/pytorch/serve/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=pytorch/serve" />
</a>

Made with [contrib.rocks](https://contrib.rocks).
## ⚖️ Disclaimer
This repository is jointly operated and maintained by Amazon, Meta and a number of individual contributors listed in the [CONTRIBUTORS](https://github.com/pytorch/serve/graphs/contributors) file. For questions directed at Meta, please send an email to opensource@fb.com. For questions directed at Amazon, please send an email to torchserve@amazon.com. For all other questions, please open up an issue in this repository [here](https://github.com/pytorch/serve/issues).

*TorchServe acknowledges the [Multi Model Server (MMS)](https://github.com/awslabs/multi-model-server) project from which it was derived*
