# TorchServe REST API

TorchServe uses a RESTful API for both inference and management calls. The API is compliant with the [OpenAPI specification 3.0](https://swagger.io/specification/).
You can easily generate client side code for Java, Scala, C#, or Javascript by using [swagger codegen](https://swagger.io/swagger-codegen/).

When TorchServe starts, it starts two web services:

* [Inference API](inference_api.md)
* [Management API](management_api.md)
* [Metrics API](metrics_api.md)
* [Workflow Inference API](workflow_inference_api.md)
* [Workflow Management API](workflow_management_api.md)

By default, TorchServe listens on port 8080 for the Inference API and 8081 for the Management API.
Both APIs are accessible only from localhost by default. To enable access from a remote host, see [TorchServe Configuration](configuration.md).
