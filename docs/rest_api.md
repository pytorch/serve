# MMS REST API

MMS use RESTful API for both inference and management calls. The API is compliance with [OpenAPI specification 3.0](https://swagger.io/specification/). User can easily generate client side code for Java, Scala, C#, Javascript use [swagger codegen](https://swagger.io/swagger-codegen/).

When MMS startup, it start two web services:
* [Inference API](inference_api.md)
* [Management API](management_api.md)

By default, MMS listening on 8080 port for Inference API and 8081 on Management API.
Both API is only accessible from localhost. Please see [MMS Configuration](configuration.md) for how to enable access from remote host. 
