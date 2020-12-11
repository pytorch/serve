# Introduction

Many model serving systems provide a signature for request bodies. Examples include:

- [Seldon](https://docs.seldon.io/projects/seldon-core/en/v1.1.0/graph/protocols.html)
- [KFServing](https://github.com/kubeflow/kfserving/tree/master/docs)
- [Google Cloud AI Platform](https://cloud.google.com/ai-platform/prediction/docs/online-predict)

Data scientists use these multi-framework systems to manage deployments of many different models, possibly written in different languages and frameworks. The platforms offer additional analytics on top of model serving, including skew detection, explanations and A/B testing. These platforms need a well-structured signature in order to both standardize calls across different frameworks and to understand the input data. To simplify support for many frameworks, though, these platforms will simply pass the request body along to the underlying model server.

Torchserve currently has no fixed request body signature. Envelopes allow you to automatically translate from the fixed signature required for your model orchestrator to a flat Python list.

# Usage
1. When you write a handler, always expect a plain Python list containing data ready to go into `preprocess`. Crucially, you should assume that your handler code looks the same locally or in your model orchestrator.
1. When you deploy Torchserve behind a model orchestrator, make sure to set the corresponding `service_envelope` in your `config.properties` file. For example, if you're using Google Cloud AI Platform, which has a JSON format, you'd add `service_envelope=json` to your `config.properties` file.

# Contributing
Add new files under `ts/torch_handler/request_envelope`. Only include one class per file. The key used in `config.properties` will be the name of the .py file you write your class in.
