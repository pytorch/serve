## TorchServe on Google Kubernetes Engine (AKS)

### 1 Create an GKE cluster

This quickstart requires that you are running the gcloud version 319.0.0 or later. Run `gcloud --version` to find the version. If you need to install or upgrade, see [Install gcloud SDK](https://cloud.google.com/sdk/docs/install).

#### 1.1 Set Gcloud account information

```bash
gcloud init
```

#### 1.2 Create GKE cluster

Use the [gcloud container clusters create](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-zonal-cluster) command to create an GKE cluster. The following example creates a cluster named *torchserve* with one node with a *nvidia-tesla-t4* GPU. This will take several minutes to complete.

```bash
gcloud container clusters create torchserve --machine-type n1-standard-4 --accelerator type=nvidia-tesla-t4,count=1 --num-nodes 1 --region us-west1 --node-locations us-west1-a
```

#### 1.3 Connect to the cluster

To manage a Kubernetes cluster, you use [kubectl](https://kubernetes.io/docs/user-guide/kubectl/), the Kubernetes command-line client. If you use GKE Cloud Shell, `kubectl` is already installed. To install `kubectl` locally, use the [gcloud components install](https://kubernetes.io/docs/tasks/tools/install-kubectl/) command:

```bash
gcloud components install kubectl
```

To configure `kubectl` to connect to your Kubernetes cluster, use the [gcloud container clusters get-credentials](https://cloud.google.com/sdk/gcloud/reference/container/clusters/get-credentials) command. This command downloads credentials and configures the Kubernetes CLI to use them.

```bash
gcloud container clusters get-credentials torchserve --region us-west1 --project pytorch-tests-261423
```

#### 1.4 Install helm

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

### 2 Deploy TorchServe on GKE

#### 2.1 Download the github repository and enter the kubernetes directory

```git clone https://github.com/pytorch/serve.git```

```cd serve/kubernetes/GKE```

#### 2.2 Install NVIDIA device plugin

Before the GPUs in the nodes can be used, you must deploy a DaemonSet for the NVIDIA device plugin. This DaemonSet runs a pod on each node to provide the required drivers for the GPUs.

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

`kubectl get nodes "-o=custom-columns=NAME:.metadata.name,MEMORY:.status.allocatable.memory,CPU:.status.allocatable.cpu,GPU:.status.allocatable.nvidia\.com/gpu"` should show something similar to:

```bash
NAME                                        MEMORY       CPU     GPU
gke-torchserve-default-pool-aa9f7d99-ggc9   12698376Ki   3920m   1
```

#### 2.3 Create a storage disk

A standard storage class is created with google compute disk. If multiple pods need concurrent access to the same storage volume, you need Google NFS. Create the storage disk named *nfs-disk* with the following command:

```gcloud compute disks create --size=100GB --zone=us-west1-a nfs-disk```

#### 2.4 Create NFS Server

Modify the values.yaml in the nfs-provisioner with persistent volume name, disk name and node affinity zones and install nfs-provisioner using helm.

```bash
cd GKE

helm instlal nfs-server  ./nfs-provisioner
```

```kubetl get pods``` should show something similiar to:

```bash
NAME                                             READY   STATUS    RESTARTS   AGE
pod/nfs-server-nfs-provisioner-bcc7c96cc-5xr2k   1/1     Running   0          19h
```

#### 2.5 Create PV and PVC

Run the below command and get NFS server IP:

```bash
kubectl get svc -n default nfs-server-nfs-provisioner -o jsonpath='{.spec.clusterIP}'
```

Replace storage size and server IP in pv_pvc.yaml with the server IP got from above command. Run the below kubectl command and create PV and PVC

```bash
kubectl apply -f templates/pv_pvc.yaml -n default
```

Verify that the PVC / PV is created by excuting.

```kubectl get pvc,pv -n default```

Your output should look similar to

```bash
NAME                   CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM         STORAGECLASS   REASON   AGE
persistentvolume/nfs   10Gi       RWX            Retain           Bound    default/nfs                           20h

NAME                        STATUS   VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS   AGE
persistentvolumeclaim/nfs   Bound    nfs      10Gi       RWX                           20h
```

#### 2.6 Create a pod and copy MAR / config files

Create a pod named `pod/model-store-pod` with PersistentVolume mounted so that we can copy the MAR / config files.

```kubectl apply -f templates/pod.yaml```

Your output should look similar to

```pod/model-store-pod created```

Verify that the pod is created by excuting.

```kubectl get po```

Your output should look similar to

```bash
NAME                                              READY   STATUS    RESTARTS   AGE
model-store-pod                                   1/1     Running   0          143m
```

#### 2.6 Down and copy MAR / config files

```bash
wget https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar
wget https://torchserve.pytorch.org/mar_files/mnist.mar

kubectl exec --tty pod/model-store-pod -- mkdir /pv/model-store/
kubectl cp squeezenet1_1.mar model-store-pod:/pv/model-store/squeezenet1_1.mar
kubectl cp mnist.mar model-store-pod:/pv/model-store/mnist.mar

kubectl exec --tty pod/model-store-pod -- mkdir /pv/config/
kubectl cp config.properties model-store-pod:/pv/config/config.properties
```

Verify that the MAR / config files have been copied to the directory.

```kubectl exec --tty pod/model-store-pod -- ls -lR /pv/```

Your output should look similar to

```bash
/pv/:
total 28
drwxr-xr-x 2 root root  4096 Nov 21 16:30 config
-rw-r--r-- 1 root root    16 Nov 21 15:42 index.html
drwx------ 2 root root 16384 Nov 21 15:42 lost+found
drwxr-xr-x 2 root root  4096 Nov 21 16:12 model-store

/pv/config:
total 4
-rw-rw-r-- 1 1000 1000 579 Nov 21 16:30 config.properties

/pv/lost+found:
total 0

/pv/model-store:
total 8864
-rw-rw-r-- 1 1000 1000 4463882 Nov 21 16:12 mnist.mar
-rw-rw-r-- 1 1000 1000 4609382 Nov 21 16:11 squeezenet1_1.mar
```

#### 2.7 Install Torchserve using Helm Charts

Enter the Helm directory and install TorchServe using Helm Charts.
```cd ../Helm```

```helm install ts .```

Your output should look similar to

```bash
NAME: ts
LAST DEPLOYED: Thu Aug 20 02:07:38 2020
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
```

#### 2.8 Check the status of TorchServe

```kubectl get po```

The installation will take a few minutes. Output like this means the installation is not completed yet.

```bash
NAME                               READY   STATUS              RESTARTS   AGE
torchserve-75f5b67469-5hnsn        0/1     ContainerCreating   0          6s

Output like this means the installation is completed.

NAME                               READY   STATUS    RESTARTS   AGE
torchserve-75f5b67469-5hnsn        1/1     Running   0          2m36s
```

### 3 Test Torchserve Installation

#### 3.1 Fetch the Load Balancer Extenal IP

Fetch the Load Balancer Extenal IP by executing.

```kubectl get svc```

Your output should look similar to

```bash
NAME               TYPE           CLUSTER-IP     EXTERNAL-IP    PORT(S)                         AGE
kubernetes         ClusterIP      10.0.0.1       <none>         443/TCP                         5d19h
torchserve         LoadBalancer   10.0.39.88     your-external-IP   8080:30306/TCP,8081:30442/TCP   48s
```

#### 3.2 Test Management / Prediction APIs

```bash
curl http://your-external-IP:8081/models
```

Your output should look similar to

```json
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
```

```bash
curl http://your-external-IP:8081/models/mnist
```

Your output should look similar to

```json
[
  {
    "modelName": "mnist",
    "modelVersion": "1.0",
    "modelUrl": "mnist.mar",
    "runtime": "python",
    "minWorkers": 5,
    "maxWorkers": 5,
    "batchSize": 1,
    "maxBatchDelay": 200,
    "loadedAtStartup": false,
    "workers": [
      {
        "id": "9003",
        "startTime": "2020-08-20T03:06:38.435Z",
        "status": "READY",
        "gpu": false,
        "memoryUsage": 32194560
      },
      {
        "id": "9004",
        "startTime": "2020-08-20T03:06:38.436Z",
        "status": "READY",
        "gpu": false,
        "memoryUsage": 31842304
      },
      {
        "id": "9005",
        "startTime": "2020-08-20T03:06:38.436Z",
        "status": "READY",
        "gpu": false,
        "memoryUsage": 44621824
      },
      {
        "id": "9006",
        "startTime": "2020-08-20T03:06:38.436Z",
        "status": "READY",
        "gpu": false,
        "memoryUsage": 42045440
      },
      {
        "id": "9007",
        "startTime": "2020-08-20T03:06:38.436Z",
        "status": "READY",
        "gpu": false,
        "memoryUsage": 31584256
      }
    ]
  }
]
```

### 4 Delete the cluster

To avoid google charges, you should clean up unneeded resources. When the GKE cluster is no longer needed, use the gcloud container clusters delete command to remove GKE cluster.

```bash
gcloud container clusters delete torchserve --region us-west1
```
