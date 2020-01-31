# Batch Inference with TorchServe

## Contents of this Document
* [Introduction](#introduction)
* [Conclusion](#conclusion)   

## Introduction

Batching in the Machine-Learning/Deep-Learning is a process of aggregating inference-requests and sending this aggregated requests through the ML/DL framework for inference at once.
TorchServe was designed to natively support batching of incoming inference requests. This functionality provides customer using TorchServe to optimally utilize their host resources, because most ML/DL frameworks
are optimized for batch requests. This optimal utilization of host resources in turn reduces the operational expense of hosting an inference service using TorchServe. In this document we will go through an example of how this is done
and compare the performance of running a batched inference against running single inference.

## Prerequisites:
Before jumping into this document, please go over the following docs
1. [What is TorchServe?](../README.md)
1. [What is custom service code?](custom_service.md)

## Batch Inference with TorchServe
To support batching of inference requests, TorchServe needs the following:
1. TorchServe Model Configuration: TorchServe provides means to configure "Max Batch Size" and "Max Batch Delay" through "POST /models" API. 
   TorchServe needs to know the maximum batch size that the model can handle and the maximum delay that TorchServe should wait for, to form this request-batch. 
2. Model Handler code: TorchServe requires the Model Handler to handle the batch of inference requests. 

## TODO : Add detailed example with pytorch model.

## Conclusion
The take away from the experiments is that batching is a very useful feature. In cases where the services receive heavy load of requests or each request has high I/O, its advantageous
to batch the requests. This allows for maximally utilizing the compute resources, especially GPU compute which are also more often than not more expensive. But customers should
do their due diligence and perform enough tests to find optimal batch size depending on the number of GPUs available and number of models loaded per GPU. Customers should also
analyze their traffic patterns before enabling the batch-inference. As shown in the above experiments, services receiving TPS lesser than the batch size would lead to consistent
"batch delay" timeouts and cause the response latency per request to spike. As any cutting edge technology, batch-inference is definitely a double edged sword. 

   
