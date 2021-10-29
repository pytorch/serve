## TorchServe on Azure Kubernetes Service (AKS)

### 1 Create an AKS cluster

This quickstart requires that you are running the Azure CLI version 2.0.64 or later. Run `az --version` to find the version. If you need to install or upgrade, see [Install Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli).

#### 1.1 Set Azure account information

```az login```

```az account set -s your-subscription-ID```

#### 1.2 Create a resource group

An Azure resource group is a logical group in which Azure resources are deployed and managed. When you create a resource group, you are asked to specify a location. This location is where resource group metadata is stored, it is also where your resources run in Azure if you don't specify another region during resource creation. Create a resource group using the [az group create](https://docs.microsoft.com/en-us/cli/azure/group#az-group-create) command.

The following example creates a resource group named *myResourceGroup* in the *eastus* location.

```az group create --name myResourceGroup --location eastus```

#### 1.3 Create AKS cluster

Use the [az aks create](https://docs.microsoft.com/en-us/cli/azure/aks?view=azure-cli-latest#az-aks-create) command to create an AKS cluster. The following example creates a cluster named *myAKSCluster* with one node. This will take several minutes to complete.

```az aks create --resource-group myResourceGroup --name myAKSCluster --node-vm-size Standard_NC6 --node-count 1 --generate-ssh-keys```

#### 1.4 Connect to the cluster

To manage a Kubernetes cluster, you use [kubectl](https://kubernetes.io/docs/user-guide/kubectl/), the Kubernetes command-line client. If you use Azure Cloud Shell, `kubectl` is already installed. To install `kubectl` locally, use the [az aks install-cli](https://docs.microsoft.com/en-us/cli/azure/aks?view=azure-cli-latest#az-aks-install-cli) command:

```az aks install-cli```

To configure `kubectl` to connect to your Kubernetes cluster, use the [az aks get-credentials](https://docs.microsoft.com/en-us/cli/azure/aks?view=azure-cli-latest#az-aks-get-credentials) command. This command downloads credentials and configures the Kubernetes CLI to use them.

```az aks get-credentials --resource-group myResourceGroup --name myAKSCluster```

#### 1.5 Install helm

```
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3
chmod 700 get_helm.sh
./get_helm.sh
```

### 2 Deploy TorchServe on AKS

#### 2.1 Download the github repository and enter the kubernetes directory

```git clone https://github.com/pytorch/serve.git```

```cd serve/kubernetes/AKS```

#### 2.2 Install NVIDIA device plugin

Before the GPUs in the nodes can be used, you must deploy a DaemonSet for the NVIDIA device plugin. This DaemonSet runs a pod on each node to provide the required drivers for the GPUs.

```kubectl apply -f templates/nvidia-device-plugin-ds.yaml```
`kubectl get pods` should show something similar to:

```bash
NAME                  READY  STATUS  RESTARTS  AGE

nvidia-device-plugin-daemonset-7lvxd  1/1   Running  0     42s
```


#### 2.3 Create a storage class

A storage class is used to define how an Azure file share is created. If multiple pods need concurrent access to the same storage volume, you need Azure Files. Create the storage class with the following kubectl apply command:

```kubectl apply -f templates/Azure_file_sc.yaml```

#### 2.4 Create PersistentVolume

```kubectl apply -f templates/AKS_pv_claim.yaml```

Your output should look similar to

```persistentvolumeclaim/model-store-claim created```

Verify that the PVC / PV is created by excuting.

```kubectl get pvc,pv```

Your output should look similar to

```bash
NAME                                      STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS      AGE
persistentvolumeclaim/model-store-claim   Bound    pvc-c9e235a8-ca2b-4d04-8f25-8262de1bb915   1Gi        RWO            managed-premium   29s

NAME                                                        CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                       STORAGECLASS      REASON   AGE
persistentvolume/pvc-c9e235a8-ca2b-4d04-8f25-8262de1bb915   1Gi        RWO            Delete           Bound    default/model-store-claim   managed-premium            28s
```

#### 2.5 Create a pod and copy MAR / config files

Create a pod named `pod/model-store-pod` with PersistentVolume mounted so that we can copy the MAR / config files.

```kubectl apply -f templates/model_store_pod.yaml```

Your output should look similar to

```pod/model-store-pod created```

Verify that the pod is created by excuting.

```kubectl get po```

Your output should look similar to

```bash
NAME                                   READY   STATUS    RESTARTS   AGE
model-store-pod                        1/1     Running   0          143m
nvidia-device-plugin-daemonset-qccgn   1/1     Running   0          144m
```

#### 2.6 Down and copy MAR / config files

```bash
wget https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar
wget https://torchserve.pytorch.org/mar_files/mnist.mar

kubectl exec --tty pod/model-store-pod -- mkdir /mnt/azure/model-store/
kubectl cp squeezenet1_1.mar model-store-pod:/mnt/azure/model-store/squeezenet1_1.mar
kubectl cp mnist.mar model-store-pod:/mnt/azure/model-store/mnist.mar

kubectl exec --tty pod/model-store-pod -- mkdir /mnt/azure/config/
kubectl cp config.properties model-store-pod:/mnt/azure/config/config.properties
```

Verify that the MAR / config files have been copied to the directory.

```kubectl exec --tty pod/model-store-pod -- find /mnt/azure/```

Your output should look similar to

```bash
/mnt/azure/
/mnt/azure/config
/mnt/azure/config/config.properties
/mnt/azure/lost+found
/mnt/azure/model-store
/mnt/azure/model-store/mnist.mar
/mnt/azure/model-store/squeezenet1_1.mar
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

```
curl http://your-external-IP:8081/models
```

Your output should look similar to

```
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

```
curl http://your-external-IP:8081/models/mnist
```

Your output should look similar to

```
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

To avoid Azure charges, you should clean up unneeded resources. When the aks cluster is no longer needed, use the az aks delete command to remove aks cluster.

```
az aks delete --name myAKSCluster --resource-group myResourceGroup --yes --no-wait
```

Or if resource group is no longer needed, use the az group delete command to remove the resource group and all related resources.

```
az group delete --name myResourceGroup --yes --no-wait
```

## Troubleshooting
  

  **Troubleshooting Azure Cli login**

  Az login command will open your default browser, it will do so and load an Azure sign-in page.
  Otherwise, open a browser page at https://aka.ms/devicelogin and enter the authorization code displayed in your terminal.
  If no web browser is available or the web browser fails to open, use device code flow with az login --use-device-code.
  Or you can login with your credential in command line, more details, see https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli.
  
  **Troubleshooting Azure resource for AKS cluster creation**
  
  * Check AKS availble region, https://azure.microsoft.com/en-us/global-infrastructure/services/?products=kubernetes-service
  * Check AKS quota and VM size limitation, https://docs.microsoft.com/en-us/azure/aks/quotas-skus-regions
  * Check whether your subscription has enough quota to create AKS cluster, https://docs.microsoft.com/en-us/azure/networking/check-usage-against-limits
  
  **For more AKS troubleshooting, please visit https://docs.microsoft.com/en-us/azure/aks/troubleshooting**
