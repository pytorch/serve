* ## Torchserve on Kubernetes

  ## Overview

  This page demonstrates a Torchserve deployment in Kubernetes using Helm Charts. It uses the DockerHub Torchserve Image for the pods and a PersistentVolume for storing config / model files.

  ![EKS Overview](images/overview.png)

  In the following sections we would 
  * Create a EKS Cluster for deploying Torchserve
  * Create a PersistentVolume backed by EFS to store models and config
  * Use Helm charts to deploy Torchserve

  All these steps scripts are written for AWS EKS with Ubuntu 18.04 for deployment, but could be easily adopted for Kubernetes offering from other vendors.

  ## Prerequisites

  We would need the following tools to be installed to setup the K8S Torchserve cluster.

  * AWS CLI - [Installation](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html)
  * eksctl - [Installation](https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html)
  * kubectl - [Installation](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
  * helm - [Installation](https://helm.sh/docs/intro/install/)
  * jq  - For JSON parsing in CLI

  

  ```bash
  sudo apt-get update
  
  # Install AWS CLI & Set Credentials
  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
  unzip awscliv2.zip
  sudo ./aws/install
  
  # Verify your aws cli installation
  aws --version
  
  # Setup your AWS credentials / region
  export AWS_ACCESS_KEY_ID=
  export AWS_SECRET_ACCESS_KEY=
  export AWS_DEFAULT_REGION=
  
  
  # Install eksctl
  curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
  sudo mv /tmp/eksctl /usr/local/bin
  
  # Verify your eksctl installation
  eksctl version
  
  # Install kubectl
  curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x ./kubectl
  sudo mv ./kubectl /usr/local/bin/kubectl
  
  # Verify your kubectl installation
  kubectl version --client
  
  # Install helm
  curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
  chmod 700 get_helm.sh
  ./get_helm.sh
  
  
  # Install jq
  sudo apt-get install jq
  
  # Clone TS
  git clone https://github.com/pytorch/serve/
  cd kubernetes/
  ```

  

  ## EKS Cluster setup

  In this section we decribe creating a EKS Kubernetes cluster with GPU nodes. If you have an existing EKS / Kubernetes cluster you may skip this section and skip ahead to PersistentVolume preparation. 

  Ensure you have your installed all required dependices & configured AWS CLI from the previous steps  appropriate permissions. The following steps would,

  * Create a EKS cluster
  * Install all the required driver for NVIDIA GPU.


  ### Creating a EKS cluster

  **EKS Optimized AMI Subscription**

  First subscribe to EKS-optimized AMI with GPU Support in the AWS Marketplace. Subscribe [here](https://aws.amazon.com/marketplace/pp/B07GRHFXGM). These hosts would be used for the EKS Node Group. 

  More details about these AMIs and configuring can be found [here](https://github.com/awslabs/amazon-eks-ami) and [here](https://eksctl.io/usage/custom-ami-support/)

  **Create a EKS Cluster**


  To create a cluster run the following command. 

  First update the `templates/eks_cluster.yaml` with 

  ```yaml
  apiVersion: eksctl.io/v1alpha5
  kind: ClusterConfig
  
  metadata:
    name: "TorchserveCluster"
    region: "us-west-2" # Update AWS Region
  
  nodeGroups:
    - name: ng-1
      instanceType: g4dn.xlarge # Update Node Type
      desiredCapacity: 3 # Update Node count
  ```

  

  Then run the following command

  ```eksctl create cluster -f templates/eks_cluster.yaml```

  

  Your output should look similar to 

  ```bash
  ubuntu@ip-172-31-50-36:~/serve/kubernetes$ eksctl create cluster -f templates/eks_cluster.yaml
  [ℹ]  eksctl version 0.24.0
  [ℹ]  using region us-west-2
  [ℹ]  setting availability zones to [us-west-2c us-west-2b us-west-2a]
  [ℹ]  subnets for us-west-2c - public:192.168.0.0/19 private:192.168.96.0/19
  [ℹ]  subnets for us-west-2b - public:192.168.32.0/19 private:192.168.128.0/19
  [ℹ]  subnets for us-west-2a - public:192.168.64.0/19 private:192.168.160.0/19
  [ℹ]  nodegroup "ng-1" will use "ami-0b6e3586ae536bd40" [AmazonLinux2/1.16]
  [ℹ]  using Kubernetes version 1.16
  [ℹ]  creating EKS cluster "TorchserveCluster" in "us-west-2" region with un-managed nodes
  [ℹ]  1 nodegroup (ng-1) was included (based on the include/exclude rules)
  [ℹ]  will create a CloudFormation stack for cluster itself and 1 nodegroup stack(s)
  [ℹ]  will create a CloudFormation stack for cluster itself and 0 managed nodegroup stack(s)
  [ℹ]  if you encounter any issues, check CloudFormation console or try 'eksctl utils describe-stacks --region=us-west-2 --cluster=TorchserveCluster'
  [ℹ]  Kubernetes API endpoint access will use default of {publicAccess=true, privateAccess=false} for cluster "TorchserveCluster" in "us-west-2"
  [ℹ]  2 sequential tasks: { create cluster control plane "TorchserveCluster", 2 sequential sub-tasks: { update CloudWatch logging configuration, create nodegroup "ng-1" } }
  [ℹ]  building cluster stack "eksctl-TorchserveCluster-cluster"
  [ℹ]  deploying stack "eksctl-TorchserveCluster-cluster"
  [✔]  configured CloudWatch logging for cluster "TorchserveCluster" in "us-west-2" (enabled types: api, audit, authenticator, controllerManager, scheduler & no types disabled)
  [ℹ]  building nodegroup stack "eksctl-TorchserveCluster-nodegroup-ng-1"
  [ℹ]  --nodes-min=1 was set automatically for nodegroup ng-1
  [ℹ]  --nodes-max=1 was set automatically for nodegroup ng-1
  [ℹ]  deploying stack "eksctl-TorchserveCluster-nodegroup-ng-1"
  [ℹ]  waiting for the control plane availability...
  [✔]  saved kubeconfig as "/home/ubuntu/.kube/config"
  [ℹ]  no tasks
  [✔]  all EKS cluster resources for "TorchserveCluster" have been created
  [ℹ]  adding identity "arn:aws:iam::ACCOUNT_ID:role/eksctl-TorchserveCluster-nodegrou-NodeInstanceRole" to auth ConfigMap
  [ℹ]  nodegroup "ng-1" has 0 node(s)
  [ℹ]  waiting for at least 1 node(s) to become ready in "ng-1"
  [ℹ]  nodegroup "ng-1" has 1 node(s)
  [ℹ]  node "ip-instance_id.us-west-2.compute.internal" is ready
  [ℹ]  as you are using a GPU optimized instance type you will need to install NVIDIA Kubernetes device plugin.
  [ℹ]  	 see the following page for instructions: https://github.com/NVIDIA/k8s-device-plugin
  [ℹ]  kubectl command should work with "/home/ubuntu/.kube/config", try 'kubectl get nodes'
  [✔]  EKS cluster "TorchserveCluster" in "us-west-2" region is ready
  ```

  

  This would create a EKS cluster named **TorchserveCluster**. This step would takes a considetable amount time to create EKS clusters. You would be able to track the progress in your cloudformation console. If you run in to any error inspect the events tab of the Cloud Formation UI.

  

  ![EKS Overview](images/eks_cfn.png)

  

  Verify that the cluster has been created with the following commands 

  ```bash
  eksctl get  clusters
  kubectl get service,po,daemonset,pv,pvc --all-namespaces
  ```

  Your output should look similar to,

  ```bash
  ubuntu@ip-172-31-55-101:~/serve/kubernetes$ eksctl get  clusters
  NAME			REGION
  TorchserveCluster	us-west-2
  
  ubuntu@ip-172-31-55-101:~/serve/kubernetes$ kubectl get service,po,daemonset,pv,pvc --all-namespaces
  NAMESPACE     NAME                 TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)         AGE
  default       service/kubernetes   ClusterIP   10.100.0.1    <none>        443/TCP         27m
  kube-system   service/kube-dns     ClusterIP   10.100.0.10   <none>        53/UDP,53/TCP   27m
  
  NAMESPACE     NAME                           READY   STATUS    RESTARTS   AGE
  kube-system   pod/aws-node-2flf5             1/1     Running   0          19m
  kube-system   pod/coredns-55c5fcd78f-2h7s4   1/1     Running   0          27m
  kube-system   pod/coredns-55c5fcd78f-pm6n5   1/1     Running   0          27m
  kube-system   pod/kube-proxy-pp8t2           1/1     Running   0          19m
  
  NAMESPACE     NAME                        DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
  kube-system   daemonset.apps/aws-node     1         1         1       1            1           <none>          27m
  kube-system   daemonset.apps/kube-proxy   1         1         1       1            1           <none>          27m
  
  ```

  

  **NVIDIA Plugin**

  The NVIDIA device plugin for Kubernetes is a Daemonset that allows you to run GPU enabled containers. The instructions for installing the plugin can be found [here](https://github.com/NVIDIA/k8s-device-plugin#installing-via-helm-installfrom-the-nvidia-device-plugin-helm-repository)

  ```bash
  helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
  helm repo update
  helm install \
      --version=0.6.0 \
      --generate-name \
      nvdp/nvidia-device-plugin
  ```

  To verify that the plugin has been installed execute the following command 

  ```bash
  helm list
  ```

  Your output should look similar to

  ```bash
  ubuntu@ip-172-31-55-101:~/serve/kubernetes$ helm list
  NAME                           	NAMESPACE	REVISION	UPDATED                                	STATUS  	CHART                     	APP VERSION
  nvidia-device-plugin-1595917413	default  	1       	2020-07-28 06:23:34.522975795 +0000 UTC	deployed	nvidia-device-plugin-0.6.0	0.6.0
  ```

  

  ## Setup PersistentVolume backed by EFS

  Torchserve Helm Chart needs a PersistentVolume with a PVC label `model-store-claim` prepared with a specific folder structure shown below. This PersistentVolume contains the snapshot & model files which are shared between multiple pods of the torchserve deployment.

      model-server/
      ├── config
      │   └── config.properties
      └── model-store
          ├── mnist.mar
          └── squeezenet1_1.mar


  **Create EFS Volume for the EKS Cluster**

  This section describes steps to prepare a EFS backed PersistentVolume that would be used by the TS Helm Chart. To prepare a EFS volume as a shareifjccgiced model / config store we have to create a EFS file system, Security Group, Ingress rule, Mount Targets to enable EFS communicate across NAT of the EKS cluster. 

  The heavy lifting for these steps is performed by ``setup_efs.sh`` script. To run the script, Update the following variables in `setup_efs.sh`

  ```bash
  CLUSTER_NAME=TorchserveCluster # EKS TS Cluser Name
  MOUNT_TARGET_GROUP_NAME="eks-efs-group"
  ```

  Then run `source ./setup_efs.sh`. This would also set all the env variables which might be used for deletion at a later time

  The output of the script should look similar to,

  

  ```bash
  Configuring TorchserveCluster
  Obtaining VPC ID for TorchserveCluster
  Obtained VPC ID - vpc-fff
  Obtaining CIDR BLOCK for vpc-fff
  Obtained CIDR BLOCK - 192.168.0.0/16
  Creating Security Group
  Created Security Group - sg-fff
  Configuring Security Group Ingress
  Creating EFS Fils System
  Created EFS - fs-ff
  {
      "FileSystems": [
          {
              "OwnerId": "XXXX",
              "CreationToken": "4ae307b6-62aa-44dd-909e-eebe0d0b19f3",
              "FileSystemId": "fs-88983c8d",
              "FileSystemArn": "arn:aws:elasticfilesystem:us-west-2:ff:file-system/fs-ff",
              "CreationTime": "2020-07-29T08:03:33+00:00",
              "LifeCycleState": "creating",
              "NumberOfMountTargets": 0,
              "SizeInBytes": {
                  "Value": 0,
                  "ValueInIA": 0,
                  "ValueInStandard": 0
              },
              "PerformanceMode": "generalPurpose",
              "Encrypted": false,
              "ThroughputMode": "bursting",
              "Tags": []
          }
      ]
  }
  Waiting 30s for before procedding
  Obtaining Subnets
  Obtained Subnets - subnet-ff
  Creating EFS Mount Target in subnet-ff
  {
      "OwnerId": "XXXX",
      "MountTargetId": "fsmt-ff",
      "FileSystemId": "fs-ff",
      "SubnetId": "subnet-ff",
      "LifeCycleState": "creating",
      "IpAddress": "192.168.58.19",
      "NetworkInterfaceId": "eni-01ce1fd11df545226",
      "AvailabilityZoneId": "usw2-az1",
      "AvailabilityZoneName": "us-west-2b",
      "VpcId": "vpc-ff"
  }
  Creating EFS Mount Target in subnet-ff
  {
      "OwnerId": "XXXX",
      "MountTargetId": "fsmt-ff",
      "FileSystemId": "fs-ff",
      "SubnetId": "subnet-ff",
      "LifeCycleState": "creating",
      "IpAddress": "192.168.5.7",
      "NetworkInterfaceId": "eni-03db930b204de6ab2",
      "AvailabilityZoneId": "usw2-az3",
      "AvailabilityZoneName": "us-west-2c",
      "VpcId": "vpc-ff"
  }
  Creating EFS Mount Target in subnet-ff
  {
      "OwnerId": "XXXX",
      "MountTargetId": "fsmt-ff",
      "FileSystemId": "fs-ff",
      "SubnetId": "subnet-ff",
      "LifeCycleState": "creating",
      "IpAddress": "192.168.73.152",
      "NetworkInterfaceId": "eni-0a31830833bf6b030",
      "AvailabilityZoneId": "usw2-az2",
      "AvailabilityZoneName": "us-west-2a",
      "VpcId": "vpc-ff"
  }
  EFS File System ID - YOUR-EFS-ID
  EFS File System DNS Name - YOUR-EFS-ID.efs..amazonaws.com
  Succesfully created EFS & Mountpoints
  ```

  

  Upon completion of the script it would emit a EFS volume DNS Name similar to `fs-ab1cd.efs.us-west-2.amazonaws.com` where `fs-ab1cd` is the EFS filesystem id.

  

  You should be able to a Security Group in your AWS Console with Inbound Rules to a NFS (Port 2049)

  

  ![security_group](images/security_group.png)

  

  You should also find Mount Points in your EFS console for every region where there is a Node in the Node Group.

  

  ![](images/efs_mount.png)

  

  

  **Prepare PersistentVolume for Deployment**

  We use the [ELF Provisioner Helm Chart](https://github.com/helm/charts/tree/master/stable/efs-provisioner) to create a PersistentVolume backed by EFS. Run the following command to set this up.

  ```bash
  helm repo add stable https://kubernetes-charts.storage.googleapis.com
  helm install stable/efs-provisioner --set efsProvisioner.efsFileSystemId=YOUR-EFS-FS-ID --set efsProvisioner.awsRegion=us-west-2 --set efsProvisioner.reclaimPolicy=Retain --generate-name
  ```

  

  you should get an output similar to 

  

  ```bash
  NAME: efs-provisioner-1596010253
  LAST DEPLOYED: Wed Jul 29 08:10:56 2020
  NAMESPACE: default
  STATUS: deployed
  REVISION: 1
  TEST SUITE: None
  NOTES:
  You can provision an EFS-backed persistent volume with a persistent volume claim like below:
  
  kind: PersistentVolumeClaim
  apiVersion: v1
  metadata:
    name: my-efs-vol-1
    annotations:
      volume.beta.kubernetes.io/storage-class: aws-efs
  spec:
    storageClassName: aws-efs
    accessModes:
      - ReadWriteMany
    resources:
      requests:
        storage: 1Mi
  
  ```

  

  Verify that your EFS Provisioner installation is succesfull by invoking ```kubectl get pods```. Your output should look similar to,

  

  ```bash
  ubuntu@ip-172-31-50-36:~/serve/kubernetes$ kubectl get pods
  NAME                                          READY   STATUS    RESTARTS   AGE
  efs-provisioner-1596010253-6c459f95bb-v68bm   1/1     Running   0          109s
  ```

 Update `templates/efs_pv_claim.yaml` - `resources.request.storage` with the size of your PVC claim based on the size of the models you plan to use.  Now run, ```kubectl apply -f templates/efs_pv_claim.yaml```. This would also create a pod named `pod/model-store-pod` with PersistentVolume mounted so that we can copy the MAR / config files in the same folder structure described above. 

  

  Your output should look similar to,

  ```bash
  ubuntu@ip-172-31-50-36:~/serve/kubernetes$ kubectl apply -f templates/efs_pv_claim.yaml
  persistentvolumeclaim/model-store-claim created
  pod/model-store-pod created
  ```

  

  Verify that the PVC / Pod is created  by excuting.   ```kubectl get service,po,daemonset,pv,pvc --all-namespaces``` 

  You should see

  * ```Running``` status for ```pod/model-store-pod```  
  * ```Bound``` status for ```default/model-store-claim``` and ```persistentvolumeclaim/model-store-claim```

  

  ```bash
  ubuntu@ip-172-31-50-36:~/serve/kubernetes$ kubectl get service,po,daemonset,pv,pvc --all-namespaces
  NAMESPACE     NAME                 TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)         AGE
  default       service/kubernetes   ClusterIP   10.100.0.1    <none>        443/TCP         107m
  kube-system   service/kube-dns     ClusterIP   10.100.0.10   <none>        53/UDP,53/TCP   107m
  
  NAMESPACE     NAME                                              READY   STATUS    RESTARTS   AGE
  default       pod/efs-provisioner-1596010253-6c459f95bb-v68bm   1/1     Running   0          4m49s
  default       pod/model-store-pod                               1/1     Running   0          8s
  kube-system   pod/aws-node-xx8kp                                1/1     Running   0          99m
  kube-system   pod/coredns-5c97f79574-tchfg                      1/1     Running   0          107m
  kube-system   pod/coredns-5c97f79574-thzqw                      1/1     Running   0          106m
  kube-system   pod/kube-proxy-4l8mw                              1/1     Running   0          99m
  kube-system   pod/nvidia-device-plugin-daemonset-dbhgq          1/1     Running   0          94m
  
  NAMESPACE     NAME                                            DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
  kube-system   daemonset.apps/aws-node                         1         1         1       1            1           <none>          107m
  kube-system   daemonset.apps/kube-proxy                       1         1         1       1            1           <none>          107m
  kube-system   daemonset.apps/nvidia-device-plugin-daemonset   1         1         1       1            1           <none>          94m
  
  NAMESPACE   NAME                                                        CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                       STORAGECLASS   REASON   AGE
              persistentvolume/pvc-baf0bd37-2084-4a08-8a3c-4f77843b4736   1Gi        RWX            Delete           Bound    default/model-store-claim   aws-efs                 8s
  
  NAMESPACE   NAME                                      STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS   AGE
  default     persistentvolumeclaim/model-store-claim   Bound    pvc-baf0bd37-2084-4a08-8a3c-4f77843b4736   1Gi        RWX            aws-efs        8s
  ```

  

  Now edit the TS config file `config.properties` that would be used for the deployment. Any changes to this config should also have corresponding changes in Torchserve Helm Chart that we install in the next section. This config would be used by every torchserve instance in worker pods.

  

  The default config starts **squeezenet1_1** and **mnist** from the model zoo with 3, 5 workers.

  

  ```yaml
  inference_address=http://0.0.0.0:8080
  management_address=http://0.0.0.0:8081
  metrics_address=http://0.0.0.0:8082
  NUM_WORKERS=1
  number_of_gpu=1
  number_of_netty_threads=32
  job_queue_size=1000
  model_store=/home/model-server/shared/model-store
  model_snapshot={"name":"startup.cfg","modelCount":2,"models":{"squeezenet1_1":{"1.0":{"defaultVersion":true,"marName":"squeezenet1_1.mar","minWorkers":3,"maxWorkers":3,"batchSize":1,"maxBatchDelay":100,"responseTimeout":120}},"mnist":{"1.0":{"defaultVersion":true,"marName":"mnist.mar","minWorkers":5,"maxWorkers":5,"batchSize":1,"maxBatchDelay":200,"responseTimeout":60}}}}
  ```

  

  Now copy the files over to PersistentVolume using the following commands.

  

  ```bash
  wget https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar
  wget https://torchserve.pytorch.org/mar_files/mnist.mar
  
  kubectl exec --tty pod/model-store-pod -- mkdir /pv/model-store/
  kubectl cp squeezenet1_1.mar model-store-pod:/pv/model-store/squeezenet1_1.mar
  kubectl cp mnist.mar model-store-pod:/pv/model-store/mnist.mar
  
  
  kubectl exec --tty pod/model-store-pod -- mkdir /pv/config/
  kubectl cp config.properties model-store-pod:/pv/config/config.properties
  ```

  

  Verify that the files have been copied by executing ```kubectl exec --tty pod/model-store-pod -- find /pv/``` . You should get an output similar to,

  

  ```bash
  ubuntu@ip-172-31-50-36:~/serve/kubernetes$ kubectl exec --tty pod/model-store-pod -- find /pv/
  /pv/
  /pv/config
  /pv/config/config.properties
  /pv/model-store
  /pv/model-store/squeezenet1_1.mar
  /pv/model-store/mnist.mar
  ```

  

  Finally terminate the pod - `kubectl delete pod/model-store-pod`.

  

  ## Deploy TorchServe using Helm Charts

  
  The following table describes all the parameters for the Helm Chart.

  | Parameter          | Description              | Default                         |
  | ------------------ | ------------------------ | ------------------------------- |
  | `image`            | Torchserve Serving image | `pytorch/torchserve:latest-gpu` |
  | `inference_port`   | TS Inference port        | `8080`                          |
  | `management_port`  | TS Management port       | `8081`                          |
  | `metrics_port`     | TS Mertics port          | `8082`                          |
  | `replicas`         | K8S deployment replicas  | `1`                             |
  | `model-store`      | EFS mountpath            | `/home/model-server/shared/`    |
  | `persistence.size` | Storage size to request  | `1Gi`                           |
  | `n_gpu`            | Number of GPU in a TS Pod| `1`                             |
  | `n_cpu`            | Number of CPU in a TS Pod| `1`                             |
  | `memory_limit`     | TS Pod memory limit      | `4Gi`                           |
  | `memory_request`   | TS Pod memory request    | `1Gi`                           |


  Edit the values in `values.yaml` with the right parameters.  Somethings to consider,
  
  * Set torchserve_image to the `pytorch/torchserve:latest` if your nodes are CPU.
  * Set `persistence.size` based on the size of your models.
  * The value of `replicas` should be less than number of Nodes in the Node group.
  * `n_gpu` would be exposed to TS container by docker. This should be set to `number_of_gpu` in `config.properties` above.
  * `n_gpu` & `n_cpu` values are used on a per pod level and not in the entire cluster level

  ```yaml
  # Default values for torchserve helm chart.
  
  torchserve_image: pytorch/torchserve:latest-gpu
  
  namespace: torchserve
  
  torchserve:
    management_port: 8081
    inference_port: 8080
    metrics_port: 8082
    pvd_mount: /home/model-server/shared/
    n_gpu: 1
    n_cpu: 1
    memory_limit: 4Gi
    memory_request: 1Gi
  
  deployment:
    replicas: 1 # Changes this to number of node in Node Group
  
  persitant_volume:
    size: 1Gi
  ```


  To install Torchserve run ```helm install ts .```  
  

  ```bash
  ubuntu@ip-172-31-50-36:~/serve/kubernetes$ helm install ts .
  NAME: ts
  LAST DEPLOYED: Wed Jul 29 08:29:04 2020
  NAMESPACE: default
  STATUS: deployed
  REVISION: 1
  TEST SUITE: None
  ```
  

  Verify that torchserve has succesfully started by executing ```kubectl exec pod/torchserve-fff -- cat logs/ts_log.log``` on your torchserve pod. You can get this id by lookingup `kubectl get po --all-namespaces`

  

  Your output should should look similar to 

  ```bash
  ubuntu@ip-172-31-50-36:~/serve/kubernetes$ kubectl exec pod/torchserve-fff -- cat logs/ts_log.log
  2020-07-29 08:29:08,295 [INFO ] main org.pytorch.serve.ModelServer -
  Torchserve version: 0.1.1
  TS Home: /home/venv/lib/python3.6/site-packages
  Current directory: /home/model-server
  ......
  ```


  ## Test Torchserve Installation

  Fetch the Load Balancer Extenal IP by executing 

  ```bash
  kubectl get svc
  ```

  You should see an entry similar to 

  ```bash
  ubuntu@ip-172-31-65-0:~/ts/rel/serve$ kubectl get svc
  NAME         TYPE           CLUSTER-IP      EXTERNAL-IP                            PORT(S)                                        AGE
  torchserve   LoadBalancer   10.100.142.22   your_elb.us-west-2.elb.amazonaws.com   8080:30428/TCP,8081:31415/TCP,8082:30453/TCP   14m
  ```

  Now execute the following commands to test Management / Prediction / Metrics APIs
  ```bash
  curl http://your_elb.us-west-2.elb.amazonaws.com:8081/models
  
  # You should something similar to the following
  {
    "models": [
      {
        "modelName": "mnist",
        "modelUrl": "mnist.mar"
      },
      {
        "modelName": "squeezenet1_1",
        "modelUrl": "squeezenet1_1.mar"
      }
    ]
  }
  
  
  curl http://your_elb.us-west-2.elb.amazonaws.com:8081/models/squeezenet1_1
  
  # You should see something similar to the following
  [
    {
      "modelName": "squeezenet1_1",
      "modelVersion": "1.0",
      "modelUrl": "squeezenet1_1.mar",
      "runtime": "python",
      "minWorkers": 3,
      "maxWorkers": 3,
      "batchSize": 1,
      "maxBatchDelay": 100,
      "loadedAtStartup": false,
      "workers": [
        {
          "id": "9000",
          "startTime": "2020-07-23T18:34:33.201Z",
          "status": "READY",
          "gpu": true,
          "memoryUsage": 177491968
        },
        {
          "id": "9001",
          "startTime": "2020-07-23T18:34:33.204Z",
          "status": "READY",
          "gpu": true,
          "memoryUsage": 177569792
        },
        {
          "id": "9002",
          "startTime": "2020-07-23T18:34:33.204Z",
          "status": "READY",
          "gpu": true,
          "memoryUsage": 177872896
        }
      ]
    }
  ]
  
  
  wget https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg
  curl -X POST  http://your_elb.us-west-2.elb.amazonaws.com.us-west-2.elb.amazonaws.com:8080/predictions/squeezenet1_1 -T kitten_small.jpg
  
  # You should something similar to the following
  [
    {
      "lynx": 0.5370921492576599
    },
    {
      "tabby": 0.28355881571769714
    },
    {
      "Egyptian_cat": 0.10669822245836258
    },
    {
      "tiger_cat": 0.06301568448543549
    },
    {
      "leopard": 0.006023923866450787
    }
  ]

  curl http://your_elb.us-west-2.elb.amazonaws.com.us-west-2.elb.amazonaws.com:8082/metrcis

  # You should something similar to the following
  [
    # HELP ts_inference_requests_total Total number of inference requests.
    # TYPE ts_inference_requests_total counter
    ts_inference_requests_total{uuid="837e6ea7-d5c1-4e9d-8f2f-0a67178f22e6",model_name="squeezenet1_1",model_version="default",} 1.0
    # HELP ts_queue_latency_microseconds Cumulative queue duration in microseconds
    # TYPE ts_queue_latency_microseconds counter
    ts_queue_latency_microseconds{uuid="837e6ea7-d5c1-4e9d-8f2f-0a67178f22e6",model_name="squeezenet1_1",model_version="default",} 247.487
    # HELP ts_inference_latency_microseconds Cumulative inference duration in microseconds
    # TYPE ts_inference_latency_microseconds counter
    ts_inference_latency_microseconds{uuid="837e6ea7-d5c1-4e9d-8f2f-0a67178f22e6",model_name="squeezenet1_1",model_version="default",} 73541.565
  ]
  ```
  ## Metrics

  ## Install prometheus
  ```
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
  helm install prometheus prometheus prometheus-community/prometheus
  ```
 
  ## Install grafana
  ```
  helm repo add grafana https://grafana.github.io/helm-charts
  helm install grafana grafana/grafana
  ```
  Get admin user password by running:
  
  ```
  kubectl get secret --namespace default grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
  ```

  ## Add prometheus as data source in grafana
  ```
  kubectl get pods

  NAME                                             READY   STATUS    RESTARTS   AGE
  efs-provisioner-1603257008-b6b54d986-gng9g       1/1     Running   0          5h15m
  grafana-cbd8775fd-6f8l5                          1/1     Running   0          4h12m
  model-store-pod                                  1/1     Running   0          4h35m
  prometheus-alertmanager-776df7bfb5-hpsp4         2/2     Running   0          4h42m
  prometheus-kube-state-metrics-6df5d44568-zkcm2   1/1     Running   0          4h42m
  prometheus-node-exporter-fvsd6                   1/1     Running   0          4h42m
  prometheus-node-exporter-tmfh8                   1/1     Running   0          4h42m
  prometheus-pushgateway-85948997f7-4s4bj          1/1     Running   0          4h42m
  prometheus-server-f8677599b-xmjbt                2/2     Running   0          4h42m
  torchserve-7d468f9894-fvmpj                      1/1     Running   0          4h33m

  kubectl get pod prometheus-server-f8677599b-xmjbt -o jsonpath='{.status.podIPs[0].ip}'
  192.168.52.141
  ```
  ![Add data source](images/grafana_datasource.png)


  ## Expose grafana with loadbalancer
  ```
  kubectl patch service grafana -p '{"spec": {"type": "LoadBalancer"}}'

  kubectl get svc grafana -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
  ```

  ## Login to grafana
  http://your.grafana.elb.us-west-2.elb.amazonaws.com:3000

  ## Troubleshooting
  
  ### Troubleshooting EKCTL Cluster Creation

  #### EKS Cluster creation fails 

  * AWS Account limits
    ```
    [✖]  unexpected status "CREATE_IN_PROGRESS" while waiting for CloudFormation stack "eksctl-TorchserveCluster-nodegroup-ng-1"
    [ℹ]  fetching stack events in attempt to troubleshoot the root cause of the failure
    [!]  1 error(s) occurred and cluster hasn't been created properly, you may wish to check CloudFormation console
    [ℹ]  to cleanup resources, run 'eksctl delete cluster --region=us-west-2 --name=TorchserveCluster'
    [✖]  waiting for CloudFormation stack "eksctl-TorchserveCluster-nodegroup-ng-1": RequestCanceled: waiter context canceled
    caused by: context deadline exceeded
    Error: failed to create cluster "TorchserveCluster”
    ```
    * Check that your AWS vCPU limits can support your cluster configuration (https://console.aws.amazon.com/ec2/home#Limits). e.g. "Running G4DN Dedicated Hosts	" may be set to 0 by default causing the node group creation to fail.
  * IAM Policy of the role used during cluster creation - [Minimum IAM Policy](https://eksctl.io/usage/minimum-iam-policies/)

  * Inspect your Cloudformation console's events tab to diagonize any possible issues. You should be able to find the following resources at the end of this step in the respective AWS consoles
    * EKS Cluser in the EKS UI
    * AutoScaling Group of the Node Groups
    * EC2 Node correponding to the node groups.

  * Check the eksctl documentation at [eksctl](https://eksctl.io/introduction/) website / [github repo](https://github.com/weaveworks/eksctl/issues). 

  ### Troubleshooting EFS Persitent Volume Creation

  #### "Error: failed to download "stable/efs-provisioner"
  * Run the command `helm repo add stable https://kubernetes-charts.storage.googleapis.com`
  
  #### "exec user process caused “exec format error”
  * Check whether your nodes are x86 based. The current setup instruction does not support ARM based instances.

  #### My pod is stuck in the *Init / Creating* status. Persitent volume claim is stuck in the *Pending* status
  * Incorrect CLUSTER_NAME or Duplicate MOUNT_TARGET_GROUP_NAME in `setup_efs.sh`
    * Rerun the script with a different `MOUNT_TARGET_GROUP_NAME` to avoid conflict with a previous run

  * Faulty execution of ``setup_efs.sh`` 
    * Look up the screenshots above for the expected AWS Console UI for Security group & EFS. If you don't see the Ingress Permissions / Mount Points created, execute steps from ```setup_efs.sh``` to make sure that they complete as expected.  We need 1 Mount Point for every region where Nodes would be deployed. *This step is critical to the setup* . If you have any errors, the `aws-efs-csi` driver might throw errors which might be hard to diagonize.

  * EFS CSI Driver installation
    * Incorrect efs filesystem ID passed when invoking `helm install stable/efs-provisioner ...`
      * Run `kubectl get pods` then use the pod name to run `kubectl describe pod <efs-provisioner-xxxx>`.
      * If the output says `mount.nfs:Failed to resolve server xxxxx.efs.us-west-2.amazonaws.com: Name or service not known`, the efs filesystem ID was likely incorrect. To resolve the issue:
        * Run 'kubectl get pods` and `kubectl delete pod <efs-provisioner-xxxx>` to delete the existing pod.
        * Run `helm list` and `helm uninstall <efs-provisioner-xxxx>`
        * Re-run command `helm install stable/efs-provisioner --set efsProvisioner.efsFileSystemId=<Your EFS ID> --set efsProvisioner.awsRegion=us-west-2 --set efsProvisioner.reclaimPolicy=Retain --generate-name` and make sure to use the correct EFS ID in the input. The EFS ID can be found in the output of the `setup_efs` script or by visiting your [AWS EFS console](https://console.aws.amazon.com/efs/home#file-systems)
    * You may inspect the values by running ``helm list`` and ```helm get all YOUR_RELEASE_ID``` to verify if the values used for the installation
    * You can execute the following commands to inspect the pods / events to debug EFS / CSI Issues

      ```bash
      kubectl get events --sort-by='.metadata.creationTimestamp'

      kubectl get pod --all-namespaces # Get the Pod ID

      kubectl logs pod/efs-provisioner-YOUR_POD
      kubectl logs pod/efs-provisioner-YOUR_POD
      kubectl describe pod/efs-provisioner-YOUR_POD
      ```

    * A more involved debugging step would involve installing a simple example app to verify EFS / EKS setup as described [here](https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html) (Section : *To deploy a sample application and verify that the CSI driver is working*)

    * More info about the driver can be found at 

      * [Github Page](https://github.com/kubernetes-sigs/aws-efs-csi-driver/) / [Helm Chart](https://github.com/kubernetes-incubator/external-storage/tree/master/aws/efs) / [EKS Workshop](https://www.eksworkshop.com/beginner/190_efs/efs-provisioner/) / [AWS Docs](https://aws.amazon.com/premiumsupport/knowledge-center/eks-persistent-storage/)

### Troubleshooting Torchserve Helm Chart
  #### Check configuration
  * Incorrect values in ``values.yaml``
    * If you changed values in `torchserve.pvd_mount`, make sure `config.properties` was also updated to match the values.
  * Invalid `config.properties`
    * You can verify these values by running this for local TS installation.
  #### TS Pods hanging in *Pending* state
    * Ensure you have available Nodes in Node Group.

  #### Helm Installation Issues
  * You may inspect the values by running ``helm list`` and `helm get all ts` to verify if the values used for the installation.
  * You can uninstall / reinstall the helm chart by executing  `helm uninstall ts` and `helm install ts .`
  * `helm install ts .` fails with `Error: create: failed to create: Request entity too large: limit is 3145728` or `invalid: data: Too long: must have at most 1048576 characters`.
    * Ensure that you dont have any stale files in your kubernetes directory where you are executing the command. If so, move them out of the directory or add them to .helmignore file.
  * `kubectl get svc` does't show my torchserve service
    * Try reinstalling the helm chart by executing `helm uninstall ts` and `helm install ts .`
  * "Error: unable to build kubernetes objects from release manifest: unable to recognize “”: no matches for kind “ClusterConfig” in version “eksctl.io/v1alpha5”"
    * Helm is picking up other .yaml files. Make sure you’ve added other files correctly to .helmignore. It should only run with values.yaml.
  * `kubectl describe pod` shows error message "0/1 nodes are available: 1 Insufficient cpu."
    * Ensure that the `n_cpu` value in `values.yaml` is set to a number that can be supported by the nodes in the cluster.
    
  ## Deleting Resources

  * Delete EFS `aws efs delete-file-system --file-system-id $FILE_SYSTEM_ID`
  * Delete Security Groups ``aws ec2 delete-security-group --group-id $MOUNT_TARGET_GROUP_ID` 
  * Delete EKS cluster `eksctl delete cluster --name $CLUSTER_NAME`

If you run to any issue. Delete these manually from the UI. Note that, EKS cluster & node are deployed as CFN templates. 
  

  ## Roadmap
  * [] Autoscaling
  * [] Log / Metrics Aggregation using [AWS Container Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ContainerInsights.html)
  * [] EFK Stack Integration
  * [] Readiness / Liveness Probes
  * [] Canary
  * [] Cloud agnostic Distributed Storage example
