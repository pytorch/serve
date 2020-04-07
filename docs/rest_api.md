# TorchServe REST API

TorchServe use RESTful API for both inference and management calls. The API is compliance with [OpenAPI specification 3.0](https://swagger.io/specification/).
YOu can easily generate client side code for Java, Scala, C#, or Javascript by using [swagger codegen](https://swagger.io/swagger-codegen/).

When TorchServe starts, it starts two web services:

* [Inference API](inference_api.md)
* [Management API](management_api.md)

By default, TorchServe listens on port 8080 for the Inference API and 8081 for the Management API.
Both APIs are accessible only from localhost by default. To enaboe access from a remote host, see [TorchServe Configuration](configuration.md).
