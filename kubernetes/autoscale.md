# Autoscaler 

Setup Kubernetes HPA(Horizontal Pod Autoscaler) for Torchserve, tuned for torchserve metrics. This uses Prometheus as metrics collector and Prometheus Adapter as mertrics server, serving Torchserve metrics for HPA.

## Steps

### 1. Install Torchserve with metrics enabled for prometheus format

[Install TorchServe using Helm Charts](README.md##-Deploy-TorchServe-using-Helm-Charts)
### 2. Install Prometheus

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install prometheus prometheus-community/prometheus
```

The above command outputs prometheus server url:

```bash
NAME: prometheus
LAST DEPLOYED: Wed Sep  8 19:10:49 2021
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
The Prometheus server can be accessed via port 80 on the following DNS name from within your cluster:
prometheus-server.default.svc.cluster.local
...
...
```

### 3. Install Prometheus Adapater

- Update Prometheus url and port in adapter.yaml. Use the url given in prometheus installation output.

```yaml
# Url to access prometheus
prometheus:
  # Value is templated
  url: http://prometheus-server.default.svc.cluster.local
  port: 80
  path: ""
```

- Update external metrics rules in adapter.yaml. Here we enabling external metrics in prometheus adapter and serving `ts_queue_latency_microseconds` metric.

```yaml
external:
- seriesQuery: '{__name__=~"^ts_queue_latency_microseconds"}'
  resources:
    overrides:
      namespace:
        resource: namespace
      service:
        resource: service
      pod:
        resource: pod
  name:
    matches: "^(.*)_microseconds"
    as: "ts_queue_latency_microseconds"
  metricsQuery: ts_queue_latency_microseconds
```

Refer: [Prometheus Adapter External Metrics](https://github.com/kubernetes-sigs/prometheus-adapter/blob/master/docs/externalmetrics.md)

- Install Prometheus adapter

```bash
helm install -f adapter.yaml prometheus-adapter prometheus-community/prometheus-adapter
```

The output of above command is

```
NAME: adapter
LAST DEPLOYED: Wed Sep  8 19:49:28 2021
NAMESPACE: default
STATUS: deployed
REVISION: 1
TEST SUITE: None
NOTES:
adapter-prometheus-adapter has been deployed.
In a few minutes you should be able to list metrics using the following command(s):

  kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1

  kubectl get --raw /apis/external.metrics.k8s.io/v1beta1
```

#### Check External metrics list

```bash
kubectl get --raw /apis/external.metrics.k8s.io/v1beta1 | jq
```

```json
{
  "kind": "APIResourceList",
  "apiVersion": "v1",
  "groupVersion": "external.metrics.k8s.io/v1beta1",
  "resources": [
    {
      "name": "ts_queue_latency_microseconds",
      "singularName": "",
      "namespaced": true,
      "kind": "ExternalMetricValueList",
      "verbs": [
        "get"
      ]
    }
  ]
}
```

### 4. Deploy Horizontal Pod Autoscaler for Torchserve

- Change `targetValue` as per requirement.

```yaml
kind: HorizontalPodAutoscaler
apiVersion: autoscaling/v2beta1
metadata:
  name: torchserve
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: torchserve
  # autoscale between 1 and 5 replicas
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: External
    external:
      metricName: ts_queue_latency_microseconds
      targetValue: "7000000m"
```

```bash
kubectl apply -f hpa.yaml
```

### 5. Check status of HPG

```bash
kubectl describe hpa torchserve
```

```bash
Name:                                              torchserve
Namespace:                                         default
Labels:                                            <none>
Annotations:                                       <none>
CreationTimestamp:                                 Wed, 08 Sep 2021 20:09:48 +0530
Reference:                                         Deployment/torchserve
Metrics:                                           ( current / target )
  "ts_queue_latency_microseconds" (target value):  5257630m / 7k
Min replicas:                                      1
Max replicas:                                      5
Deployment pods:                                   3 current / 3 desired
Conditions:
  Type            Status  Reason              Message
  ----            ------  ------              -------
  AbleToScale     True    ReadyForNewScale    recommended size matches current size
  ScalingActive   True    ValidMetricFound    the HPA was able to successfully calculate a replica count from external metric ts_queue_latency_microseconds(nil)
  ScalingLimited  False   DesiredWithinRange  the desired count is within the acceptable range
Events:           <none>
```