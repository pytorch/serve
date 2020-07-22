
## Torchserve on Kubernetes

## Overview

This page demonstrates a Torchserve deployment in Kubernetes using Helm Charts. This deployment leverages a shared file system for storing snapshot / model files which are shared between multiple pods of the deployment. Its uses [DockerHub Torchserve Image](https://hub.docker.com/r/pytorch/torchserve) for the deployment.

![EKS Overview](overview.png)

In this example we use EKS for Kubernetes Cluster and EFS for distributed storage. But this can replaced with any kubernetes cluster / distributed storage for PVC.

In the following sections we would 
* Create a EKS Cluster for deploying Torchserve.
* Craete a EFS backed model store which would have the models & snapshot info to be shared by multiple hosts.
* Use Helm charts to deploy Torchserve

## Prerequisites

We would need the following tools to be installed to setup the K8S Torchserve cluster.

* AWS CLI - [Installation](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html)
* eksctl - [Installation](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html)
* kubectl - [Installation](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
* helm - [Installation](https://helm.sh/docs/intro/install/)

## EKS Cluster setup

Is you have an existing EKS / Kubernetes cluster you may skip this section and skip ahead to driver installation. Ensure you have your AWS CLI configured with the credentials of an account with appropriate permissions. 

The following steps would create a EKS cluster, install all the required driver for NVIDIA GPU, EFS.


### Creating a EKS cluster

First subscribe to EKS-optimized AMI with GPU Support in the AWS Marketplace. Subscribe [here](https://aws.amazon.com/marketplace/pp/B07GRHFXGM). These hosts would be used for the EKS Node Group. 


To create a cluster run the following command. This would create a EKS cluster named **TorchserveCluster**

```eksctl create cluster -f templates/eks_cluster.yaml```

The NVIDIA device plugin for Kubernetes is a Daemonset that allows you to run GPU enabled containers. The instauctions for installing the plugin can be found [here](https://github.com/NVIDIA/k8s-device-plugin#installing-via-helm-installfrom-the-nvidia-device-plugin-helm-repository)

```
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm install \
    --version=0.6.0 \
    --generate-name \
    nvdp/nvidia-device-plugin
```

## EFS Backed Model Store Setup

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


## Deploy TorchServe using Helm Charts


| Parameter | Description | Default |
|-----------|-------------|---------|
| `image` | Torchserve Serving image | `pytorch/torchserve:latest-gpu` |
| `management-port` | TS Inference port | `8080` |
| `inference-port` | TS Management port | `8081` |
| `config` | Torchserve config with snapshot info  | `LoadBalancer` |
| `replicas`| K8S deployment replicas | `1` |
| `model-store`| EFS mountpath | `/home/model-server/model-store` |
| `persistence.size`| Storage size to request | `5Gi` |


#### Test Torchserve installation in k8s

## Roadmap
* [] Autoscaling
* [] Log Aggregation
* [] Metrics Aggregation
* [] EFK Stack Integration
* [] Readiness / Liveness Probes
* [] Canary
* [] Cloud agnostic Distributed Storage

## Troubleshooting

### Other Resources

* https://www.eksworkshop.com/beginner/190_efs/setting-up-efs/ 
