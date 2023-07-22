# Metrics API

Metrics API is listening on port 8082 and only accessible from localhost by default. To change the default setting, see [TorchServe Configuration](configuration.md). The default metrics endpoint returns Prometheus formatted metrics when [metrics_mode](https://github.com/pytorch/serve/blob/master/docs/metrics.md) configuration is set to `prometheus`. You can query metrics using curl requests or point a [Prometheus Server](#prometheus-server) to the endpoint and use [Grafana](#grafana) for dashboards.

By default these APIs are enabled however same can be disabled by setting `enable_metrics_api=false` in torchserve config.properties file.
For details refer [Torchserve config](configuration.md) docs.

```console
curl http://127.0.0.1:8082/metrics

# HELP Requests5XX Torchserve prometheus counter metric with unit: Count
# TYPE Requests5XX counter
# HELP DiskUsage Torchserve prometheus gauge metric with unit: Gigabytes
# TYPE DiskUsage gauge
DiskUsage{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 20.054508209228516
# HELP GPUUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE GPUUtilization gauge
# HELP PredictionTime Torchserve prometheus gauge metric with unit: ms
# TYPE PredictionTime gauge
PredictionTime{ModelName="resnet18",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 83.13
# HELP WorkerLoadTime Torchserve prometheus gauge metric with unit: Milliseconds
# TYPE WorkerLoadTime gauge
WorkerLoadTime{WorkerName="W-9000-resnet18_1.0",Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 4593.0
WorkerLoadTime{WorkerName="W-9001-resnet18_1.0",Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 4592.0
# HELP MemoryAvailable Torchserve prometheus gauge metric with unit: Megabytes
# TYPE MemoryAvailable gauge
MemoryAvailable{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 5829.7421875
# HELP GPUMemoryUsed Torchserve prometheus gauge metric with unit: Megabytes
# TYPE GPUMemoryUsed gauge
# HELP ts_inference_requests_total Torchserve prometheus counter metric with unit: Count
# TYPE ts_inference_requests_total counter
ts_inference_requests_total{model_name="resnet18",model_version="default",hostname="88665a372f4b.ant.amazon.com",} 3.0
# HELP GPUMemoryUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE GPUMemoryUtilization gauge
# HELP HandlerTime Torchserve prometheus gauge metric with unit: ms
# TYPE HandlerTime gauge
HandlerTime{ModelName="resnet18",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 82.93
# HELP ts_inference_latency_microseconds Torchserve prometheus counter metric with unit: Microseconds
# TYPE ts_inference_latency_microseconds counter
ts_inference_latency_microseconds{model_name="resnet18",model_version="default",hostname="88665a372f4b.ant.amazon.com",} 290371.129
# HELP CPUUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE CPUUtilization gauge
CPUUtilization{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 0.0
# HELP MemoryUsed Torchserve prometheus gauge metric with unit: Megabytes
# TYPE MemoryUsed gauge
MemoryUsed{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 8245.62109375
# HELP QueueTime Torchserve prometheus gauge metric with unit: Milliseconds
# TYPE QueueTime gauge
QueueTime{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 0.0
# HELP ts_queue_latency_microseconds Torchserve prometheus counter metric with unit: Microseconds
# TYPE ts_queue_latency_microseconds counter
ts_queue_latency_microseconds{model_name="resnet18",model_version="default",hostname="88665a372f4b.ant.amazon.com",} 365.21
# HELP DiskUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE DiskUtilization gauge
DiskUtilization{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 5.8
# HELP Requests2XX Torchserve prometheus counter metric with unit: Count
# TYPE Requests2XX counter
Requests2XX{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 8.0
# HELP Requests4XX Torchserve prometheus counter metric with unit: Count
# TYPE Requests4XX counter
# HELP WorkerThreadTime Torchserve prometheus gauge metric with unit: Milliseconds
# TYPE WorkerThreadTime gauge
WorkerThreadTime{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 1.0
# HELP DiskAvailable Torchserve prometheus gauge metric with unit: Gigabytes
# TYPE DiskAvailable gauge
DiskAvailable{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 325.05113983154297
# HELP MemoryUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE MemoryUtilization gauge
MemoryUtilization{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 64.4
```

```console
curl "http://127.0.0.1:8082/metrics?name[]=ts_inference_latency_microseconds&name[]=ts_queue_latency_microseconds" --globoff

# HELP ts_queue_latency_microseconds Torchserve prometheus counter metric with unit: Microseconds
# TYPE ts_queue_latency_microseconds counter
ts_queue_latency_microseconds{model_name="resnet18",model_version="default",hostname="88665a372f4b.ant.amazon.com",} 365.21
# HELP ts_inference_latency_microseconds Torchserve prometheus counter metric with unit: Microseconds
# TYPE ts_inference_latency_microseconds counter
ts_inference_latency_microseconds{model_name="resnet18",model_version="default",hostname="88665a372f4b.ant.amazon.com",} 290371.129
```

#### Prometheus server

To view these metrics on a Prometheus server, download and install using the instructions [here](https://prometheus.io/download/#prometheus). Create a minimal `prometheus.yml` config file as below and run `./prometheus --config.file=prometheus.yml`.

```yaml
global:
  scrape_interval:     15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
    - targets: ['localhost:9090']
  - job_name: 'torchserve'
    static_configs:
    - targets: ['localhost:8082'] #TorchServe metrics endpoint
```
Navigate to `http://localhost:9090/` on a browser to execute queries and create graphs

<img width="1231" alt="Prometheus Server" src="https://user-images.githubusercontent.com/5276346/234722761-007e168a-ebc0-4644-be60-23b2f33fa4f2.png">

#### Grafana

Once you have the Torchserve and Prometheus servers running, you can further [setup](https://prometheus.io/docs/visualization/grafana/) Grafana, point it to Prometheus server and navigate to `http://localhost:3000/` to create dashboards and graphs.

You can use command given below to start Grafana -
`sudo systemctl daemon-reload && sudo systemctl enable grafana-server && sudo systemctl start grafana-server`

<img width="1220" alt="Grafana Dashboard" src="https://user-images.githubusercontent.com/5276346/234725829-7f60e0d8-c76d-4019-ac8f-7d60069c4e58.png">
