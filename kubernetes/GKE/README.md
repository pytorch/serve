## TorchServe on Google Kubernetes Engine (GKE)

### 1 Create an GKE cluster

This quickstart requires that you are running the gcloud version 319.0.0 or later. Run `gcloud --version` to find the version. If you need to install or upgrade, see [Install gcloud SDK](https://cloud.google.com/sdk/docs/install).

#### 1.1 Set Gcloud account information

Create a gcloud configuration with account and project info.

```bash
gcloud init
```

Set compute/region and compute/zone in config

```bash
gcloud config set compute/region us-west1
gcloud config set compute/zone us-west1-a
```

If you have multiple configurations, activate required config.

```bash
gcloud config configurations activate <config_name>
```

#### 1.2 Create GKE cluster

Use the [gcloud container clusters create](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-zonal-cluster) command to create an GKE cluster. The following example creates a cluster named *torchserve* with one node. This will take several minutes to complete.

```bash
gcloud container clusters create torchserve --machine-type n1-standard-4 --num-nodes 1

WARNING: Warning: basic authentication is deprecated, and will be removed in GKE control plane versions 1.19 and newer. For a list of recommended authentication methods, see: https://cloud.google.com/kubernetes-engine/docs/how-to/api-server-authentication
WARNING: Currently VPC-native is not the default mode during cluster creation. In the future, this will become the default mode and can be disabled using `--no-enable-ip-alias` flag. Use `--[no-]enable-ip-alias` flag to suppress this warning.
WARNING: Newly created clusters and node-pools will have node auto-upgrade enabled by default. This can be disabled using the `--no-enable-autoupgrade` flag.
WARNING: Starting with version 1.18, clusters will have shielded GKE nodes by default.
WARNING: Your Pod address range (`--cluster-ipv4-cidr`) can accommodate at most 1008 node(s). 
WARNING: Starting with version 1.19, newly created clusters and node-pools will have COS_CONTAINERD as the default node image when no image type is specified.
Machines with GPUs have certain limitations which may affect your workflow. Learn more at https://cloud.google.com/kubernetes-engine/docs/how-to/gpus
Creating cluster ts in us-west1... Cluster is being health-checked (master is healthy)...done.                                                                    
Created [https://container.googleapis.com/v1/projects/xxxxx-xxxx-35xx55/zones/us-west1/clusters/ts].
To inspect the contents of your cluster, go to: https://console.cloud.google.com/kubernetes/workload_/gcloud/us-west1/ts?project=xxxxx-xxxx-35xx55
kubeconfig entry generated for ts.
NAME  LOCATION  MASTER_VERSION   MASTER_IP      MACHINE_TYPE   NODE_VERSION     NUM_NODES  STATUS
ts    us-west1  1.16.13-gke.401  34.83.140.167  n1-standard-4  1.16.13-gke.401  1          RUNNING
```

##### 1.2.1 Add GPU Accelerator

For running a cluster with GPU accelerator use `--accelerator type=,count=`

The below command creates a cluster with a single n1-standard-4 node with and nvidia-testla-t4 GPU accelerator.

```bash
gcloud container clusters create torchserve --machine-type n1-standard-4 --accelerator type=nvidia-tesla-t4,count=1 --num-nodes 1
```

#### 1.3 Connect to the cluster

To manage a Kubernetes cluster, you use [kubectl](https://kubernetes.io/docs/user-guide/kubectl/), the Kubernetes command-line client. If you use GKE Cloud Shell, `kubectl` is already installed. To install `kubectl` locally, use the [gcloud components install](https://kubernetes.io/docs/tasks/tools/install-kubectl/) command:

Below command require Cloud SDK component manager enabled.

```bash
gcloud components install kubectl
```

To configure `kubectl` to connect to your Kubernetes cluster, use the [gcloud container clusters get-credentials](https://cloud.google.com/sdk/gcloud/reference/container/clusters/get-credentials) command. This command downloads credentials and configures the Kubernetes CLI to use them.

```bash
gcloud container clusters get-credentials torchserve
Fetching cluster endpoint and auth data.
kubeconfig entry generated for torchserve.
```

#### 1.4 Install helm

```bash
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

### 2 Deploy TorchServe on GKE

#### 2.1 Download the github repository and enter the kubernetes directory

```bash
git clone https://github.com/pytorch/serve.git

cd serve/kubernetes/GKE
```

**_NOTE:_** By default the helm chart installs GPU version of torchserve. Follow steps in section [2.2](####-2.2-For-CPU-setup) for running in a CPU only cluster. For GPU setup section [2.2](####-2.2-For-CPU-setup) can be skipped.

#### 2.2 For CPU setup

* Change torchserve image in Helm/values.yaml to the CPU version
* Set `n_gpu` to `0` in Helm/values.yaml
* Skip NVIDIA plugin installation in section [2.3](#####-2.3-Install-NVIDIA-device-plugin)
  
#### 2.3 Install NVIDIA device plugin

Before the GPUs in the nodes can be used, you must deploy a DaemonSet for the NVIDIA device plugin. This DaemonSet runs a pod on each node to provide the required drivers for the GPUs.

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
daemonset.apps/nvidia-driver-installer created
```

```bash
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,MEMORY:.status.allocatable.memory,CPU:.status.allocatable.cpu,GPU:.status.allocatable.nvidia\.com/gpu"
```

should show something similar to:

```bash
NAME                                        MEMORY       CPU     GPU
gke-torchserve-default-pool-aa9f7d99-ggc9   12698376Ki   3920m   1
```

#### 2.4 Create a storage disk

A standard storage class is created with google compute disk. If multiple pods need concurrent access to the same storage volume, you need Google NFS. Create the storage disk named *nfs-disk* with the following command:

```bash
gcloud compute disks create --size=200GB --zone=us-west1-a nfs-disk

NAME     ZONE        SIZE_GB  TYPE         STATUS
nfs-disk  us-west1-a  200      pd-standard  READY

New disks are unformatted. You must format and mount a disk before it
can be used. You can find instructions on how to do this at:

https://cloud.google.com/compute/docs/disks/add-persistent-disk#formatting

```

#### 2.5 Create NFS Server

Modify the values.yaml in the nfs-provisioner with persistent volume name, disk name and node affinity zones and install nfs-provisioner using helm.

```bash
cd GKE

helm install mynfs ./nfs-provisioner/
```

```kubectl get pods``` should show something similiar to:

```bash
NAME                                             READY   STATUS    RESTARTS   AGE
pod/mynfs-nfs-provisioner-bcc7c96cc-5xr2k   1/1     Running   0          19h
```

#### 2.6 Create PV and PVC

Run the below command and get NFS server IP:

```bash
kubectl get svc -n default mynfs-nfs-provisioner -o jsonpath='{.spec.clusterIP}'
```

Replace storage size and server IP in pv_pvc.yaml with the server IP got from above command. Run the below kubectl command and create PV and PVC

```bash
kubectl apply -f templates/pv_pvc.yaml -n default
```

Verify that the PVC / PV is created by excuting.

```bash
kubectl get pvc,pv -n default
```

Your output should look similar to

```bash
NAME                   CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM         STORAGECLASS   REASON   AGE
persistentvolume/nfs   10Gi       RWX            Retain           Bound    default/nfs                           20h

NAME                        STATUS   VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS   AGE
persistentvolumeclaim/nfs   Bound    nfs      10Gi       RWX                           20h
```

#### 2.7 Create a pod and copy MAR / config files

Create a pod named `pod/model-store-pod` with PersistentVolume mounted so that we can copy the MAR / config files.

```bash
kubectl apply -f templates/pod.yaml
```

Your output should look similar to

```bash
pod/model-store-pod created
```

Verify that the pod is created by excuting.

```bash
kubectl get po
```

Your output should look similar to

```bash
NAME                                              READY   STATUS    RESTARTS   AGE
model-store-pod                                   1/1     Running   0          143m
```

#### 2.8 Down and copy MAR / config files

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

```bash
kubectl exec --tty pod/model-store-pod -- ls -lR /pv/
```

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

Delete model store pod.

```bash
kubectl delete pod/model-store-pod
pod "model-store-pod" deleted
```

### 3 Install Torchserve

[Install Torchserve using Helm Chart](../README.md#-Torchserve-on-Kubernetes)

### 4 Troubleshooting

#### 4.1 Troubleshooting GKE Cluster Creation**

Possible errors in this step may be a result of

* IAM limits.
* Quota restrictions during cluster creation - [GKE Quotas](https://cloud.google.com/compute/quotas)

You should able be able to find the following resources at the end of this step in the respective Gcoud consoles

* GKE -> Cluser in the Gcloud Console

#### 4.2 Troubleshooting NFS Persitant Volume Creation

Possible error in this step may be a result of one of the following. Your pod my be struck in *Init / Creating* forever / persitant volume claim may be in *Pending* forever.

* Storage disk not created / wrong storage disk name.

  * Check if the storage is created and the correct name is given in the values.yaml

* Node affinity does not match

  * Check affinity zone in values.yaml to match the storage disk.

* Wrong Server IP
  
  * Check the server IP mentioned in the pv_pvc.yaml

* You can execute the following commands to inspect the pods / events to debug NFS Issues

    ```bash
    kubectl get events --sort-by='.metadata.creationTimestamp'
    kubectl get pod --all-namespaces # Get the Pod ID
    kubectl logs pod/mynfs-nfs-provisioner-YOUR_POD
    kubectl logs pod/mynfs-nfs-provisioner-YOUR_POD
    kubectl describe pod/mynfs-nfs-provisioner-YOUR_POD
    ```

#### 4.5 Delete the Resources

To avoid google charges, you should clean up unneeded resources. When the GKE cluster is no longer needed, use the gcloud container clusters delete command to remove GKE cluster.

```bash
gcloud compute disks delete nfs-disk

The following disks will be deleted:
 - [nfs-disk] in [us-west1-a]

Do you want to continue (Y/n)?  y

Deleted [https://www.googleapis.com/compute/v1/projects/xxxxx-xxxx-356xx55/zones/us-west1-a/disks/nfs-disk].
```

```bash
gcloud container node-pools delete default-pool --cluster torchserve

The following node pool will be deleted.
[default-pool] in cluster [torchserve] in [us-west1]

Do you want to continue (Y/n)?  y

done.

Deleted [https://container.googleapis.com/v1/projects/xxxxx-xxxx-35xx55/zones/us-west1/clusters/torchserve/nodePools/default-pool].
```

```bash
gcloud container clusters delete torchserve

The following clusters will be deleted.
 - [torchserve] in [us-west1]

Do you want to continue (Y/n)?  y

Deleting cluster torchserve...

done.

Deleted [https://container.googleapis.com/v1/projects/xxxxx-xxxx-35xx55/zones/us-west1/clusters/torchserve].
```
