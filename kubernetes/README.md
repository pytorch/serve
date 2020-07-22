
## Torchserve on Kubernetes

### Overview

The following page demonstrate how to deploy Torchserve in Kubernetes using Helm Charts. This deployment leverages a shared file system for storing snapshot / model files which are shared between multiple pods of the deployment.

We use EKS for Kubernetes Cluster and EFS for distributed storage in this deployemnt. But this can replaces with any Kubernetes cluster / Distributed storage..

### Prerequisites

* k8s cluster setup using EKS
* kubectl / helm installation

### Cluster setup

* NVIDIA Driver
* EFS-CSI Driver

### Setup EFS 

### Deploy TorchServe using Helm Charts

### Test Torchserve installation in k8s

### Roadmap
* [] Autoscaling
* [] Log Aggregation
* [] Metrics Aggregation
* [] EFK Stack Integration
* [] Readiness / Liveness Probes
* [] Canary
* [] Cloud agnostic Distributed Storage

### Troubleshooting

### Other Resources

* https://www.eksworkshop.com/beginner/190_efs/setting-up-efs/
* 
