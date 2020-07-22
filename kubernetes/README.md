
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

We need EFS for Snapshot & Model store. The `./setup_efs.sh` script created the needed EFS resources. 

This script 
* Does A
* Does B

To run the script : 


* Ensure you have AWS CLI installed and configured.
* Update the following variables in `setup_efs.sh`

    ```
    CLUSTER_NAME=TorchserveCluster // EKS TS Cluser Name
    MOUNT_TARGET_GROUP_NAME="eks-efs-group-999" // 
    SECURITY_GROUP_NAME="ec2-instance-group-999" // Securit
    EC2_KEY_NAME="machine-learning"
    ```

Then run `./setup_efs.sh`


### Deploy TorchServe using Helm Charts


| Parameter | Description | Default |
|-----------|-------------|---------|
| `image` | Torchserve Serving image | `pytorch/torchserve:latest-gpu` |
| `management-port` | TS Inference port | `8080` |
| `inference-port` | TS Management port | `8081` |
| `config` | Torchserve config with snapshot info  | `LoadBalancer` |
| `replicas`| K8S deployment replicas | `1` |
| `model-store`| EFS mountpath | `/home/model-server/model-store` |
| `persistence.size`| Storage size to request | `5Gi` |


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
