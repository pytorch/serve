# TorchServe

TorchServe is a performant, flexible and easy to use tool for serving PyTorch models in production.


## ⚡ Why TorchServe
* [Model Management API](management_api.md): multi model management with optimized worker to model allocation
* [Inference API](inference_api.md): REST and gRPC support for batched inference
* [TorchServe Workflows](https://github.com/pytorch/serve/blob/master/examples/Workflows/README.md#workflow-examples): deploy complex DAGs with multiple interdependent models
* Default way to serve PyTorch models in
  * [Kubeflow](https://v0-5.kubeflow.org/docs/components/pytorchserving/)
  * [MLflow](https://github.com/mlflow/mlflow-torchserve)
  * [Sagemaker](https://aws.amazon.com/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/)
  * [Kserve](https://kserve.github.io/website/0.8/modelserving/v1beta1/torchserve/): Supports both v1 and v2 API
  * [Vertex AI](https://cloud.google.com/blog/topics/developers-practitioners/pytorch-google-cloud-how-deploy-pytorch-models-vertex-ai)
* Export your model for optimized inference. Torchscript out of the box, [ORT and ONNX](https://github.com/pytorch/serve/blob/master/docs/performance_guide.md#performance-guide), [IPEX](https://github.com/pytorch/serve/tree/master/examples/intel_extension_for_pytorch), [TensorRT](performance_guide.md), [FasterTransformer](https://github.com/pytorch/serve/tree/master/examples/FasterTransformer_HuggingFace_Bert)
* [Performance Guide](performance_guide.md): builtin support to optimize, benchmark and profile PyTorch and TorchServe performance
* [Expressive handlers](https://github.com/pytorch/serve/blob/master/CONTRIBUTING.md#contributing-to-torchServe): An expressive handler architecture that makes it trivial to support inferencing for your usecase with [many supported out of the box](https://github.com/pytorch/serve/tree/master/ts/torch_handler)
* [Metrics API](metrics.md): out of box support for system level metrics with [Prometheus exports](https://github.com/pytorch/serve/tree/master/examples/custom_metrics), custom metrics and PyTorch profiler support
## 🤔 How does TorchServe work

* [Serving Quick Start](https://github.com/pytorch/serve/blob/master/README.md#serve-a-model) - Basic server usage tutorial
* [Model Archive Quick Start](https://github.com/pytorch/serve/tree/master/model-archiver#creating-a-model-archive) - Tutorial that shows you how to package a model archive file.
* [Installation](https://github.com/pytorch/serve/blob/master/README.md#install-torchserve) - Installation procedures
* [Serving Models](server.md) - Explains how to use TorchServe
* [REST API](rest_api.md) - Specification on the API endpoint for TorchServe
* [gRPC API](grpc_api.md) - TorchServe supports gRPC APIs for both inference and management calls
* [Packaging Model Archive](https://github.com/pytorch/serve/tree/master/model-archiver#torch-model-archiver-for-torchserve) - Explains how to package model archive file, use `model-archiver`.
* [Inference API](inference_api.md) - How to check for the health of a deployed model and get inferences
* [Management API](management_api.md) - How to manage and scale models
* [Logging](logging.md) - How to configure logging
* [Metrics](metrics.md) - How to configure metrics
* [Prometheus and Grafana metrics](metrics_api.md) - How to configure metrics API with Prometheus formatted metrics in a Grafana dashboard
* [Captum Explanations](https://github.com/pytorch/serve/blob/master/examples/captum/Captum_visualization_for_bert.ipynb) - Built in support for Captum explanations for both text and images
* [Batch inference with TorchServe](batch_inference_with_ts.md) - How to create and serve a model with batch inference in TorchServe
* [Workflows](workflows.md) - How to create workflows to compose Pytorch models and Python functions in sequential and parallel pipelines



## Default Handlers

* [Image Classifier](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_classifier.py) - This handler takes an image and returns the name of object in that image
* [Text Classifier](https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_classifier.py) - This handler takes a text (string) as input and returns the classification text based on the model vocabulary
* [Object Detector](https://github.com/pytorch/serve/blob/master/ts/torch_handler/object_detector.py) - This handler takes an image and returns list of detected classes and bounding boxes respectively
* [Image Segmenter](https://github.com/pytorch/serve/blob/master/ts/torch_handler/image_segmenter.py)- This handler takes an image and returns output shape as [CL H W], CL - number of classes, H - height and W - width

## 🏆 Highlighted Examples

* [🤗 HuggingFace Transformers](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers) with a [Better Transformer Integration](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers#Speed-up-inference-with-Better-Transformer)
* [Model parallel inference](https://github.com/pytorch/serve/blob/master/examples/Huggingface_Transformers#model-parallelism)
* [MultiModal models with MMF](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition) combining text, audio and video
* [Dual Neural Machine Translation](https://github.com/pytorch/serve/blob/master/examples/Workflows/nmt_transformers_pipeline) for a complex workflow DAG
* [TorchServe Integrations](https://github.com/pytorch/serve/blob/master/examples/README.md#torchserve-integrations)
* [TorchServe Internals](https://github.com/pytorch/serve/blob/master/examples/README.md#torchserve-internals)
* [TorchServe UseCases](https://github.com/pytorch/serve/blob/master/examples/README.md#usecases)
* [Model Zoo](https://github.com/pytorch/serve/blob/master/docs/model_zoo.md) - List of pre-trained model archives ready to be served for inference with TorchServe.

For [more examples](https://github.com/pytorch/serve/blob/master/examples/README.md#torchserve-internals)


## Advanced Features

* [Advanced configuration](configuration.md) - Describes advanced TorchServe configurations.
* [A/B test models](https://github.com/pytorch/serve/blob/master/docs/use_cases.md#serve-models-for-ab-testing) - A/B test your models for regressions before shipping them to production
* [Custom Service](custom_service.md) - Describes how to develop custom inference services.
* [Encrypted model serving](management_api.md#encrypted-model-serving) - S3 server side model encryption via KMS
* [Snapshot serialization](https://github.com/pytorch/serve/blob/master/plugins/docs/ddb_endpoint.md) - Serialize model artifacts to AWS Dynamo DB
* [Benchmarking and Profiling](https://github.com/pytorch/serve/tree/master/benchmarks#torchserve-model-server-benchmarking) - Use JMeter or Apache Bench to benchmark your models and TorchServe itself
* [TorchServe on Kubernetes](https://github.com/pytorch/serve/blob/master/kubernetes/README.md#torchserve-on-kubernetes) -  Demonstrates a Torchserve deployment in Kubernetes using Helm Chart supported in both Azure Kubernetes Service and Google Kubernetes service
* [mlflow-torchserve](https://github.com/mlflow/mlflow-torchserve) - Deploy mlflow pipeline models into TorchServe
* [Kubeflow pipelines](https://github.com/kubeflow/pipelines/tree/master/samples/contrib/pytorch-samples) - Kubeflow pipelines and Google Vertex AI Managed pipelines
* [NVIDIA MPS](mps.md) - Use NVIDIA MPS to optimize multi-worker deployment on a single GPU

## 📰 News
* [Torchserve Performance Tuning, Animated Drawings Case-Study](https://pytorch.org/blog/torchserve-performance-tuning/)
* [Walmart Search: Serving Models at a Scale on TorchServe](https://medium.com/walmartglobaltech/search-model-serving-using-pytorch-and-torchserve-6caf9d1c5f4d)
* [🎥 Scaling inference on CPU with TorchServe](https://www.youtube.com/watch?v=066_Jd6cwZg)
* [🎥 TorchServe C++ backend](https://www.youtube.com/watch?v=OSmGGDpaesc)
* [Grokking Intel CPU PyTorch performance from first principles: a TorchServe case study](https://pytorch.org/tutorials/intermediate/torchserve_with_ipex.html)
* [Grokking Intel CPU PyTorch performance from first principles( Part 2): a TorchServe case study](https://pytorch.org/tutorials/intermediate/torchserve_with_ipex_2.html)
* [Case Study: Amazon Ads Uses PyTorch and AWS Inferentia to Scale Models for Ads Processing](https://pytorch.org/blog/amazon-ads-case-study/)
* [Optimize your inference jobs using dynamic batch inference with TorchServe on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/optimize-your-inference-jobs-using-dynamic-batch-inference-with-torchserve-on-amazon-sagemaker/)
* [Using AI to bring children's drawings to life](https://ai.facebook.com/blog/using-ai-to-bring-childrens-drawings-to-life/)
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
