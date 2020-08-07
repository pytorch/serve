# Metrics API

Metrics API is listening on port 8082 and only accessible from localhost by default. To change the default setting, see [TorchServe Configuration](configuration.md). The default metrics endpoint returns Prometheus formatted metrics. You can query metrics using curl requests or point a [Prometheus Server](#prometheus-server) to the endpoint and use [Grafana](#grafana) for dashboards.

By default these APIs are enable however same can be disabled by setting `enable_metrics_api=false` in torchserve config.properties file.
For details refer [Torchserve config](configuration.md) docs.

```console
curl http://127.0.0.1:8082/metrics

# HELP ts_inference_latency_microseconds Cumulative inference duration in microseconds
# TYPE ts_inference_latency_microseconds counter
ts_inference_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noopversioned",model_version="1.11",} 1990.348
ts_inference_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noop",model_version="default",} 2032.411
# HELP ts_inference_requests_total Total number of inference requests.
# TYPE ts_inference_requests_total counter
ts_inference_requests_total{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noopversioned",model_version="1.11",} 1.0
ts_inference_requests_total{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noop",model_version="default",} 1.0
# HELP ts_queue_latency_microseconds Cumulative queue duration in microseconds
# TYPE ts_queue_latency_microseconds counter
ts_queue_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noopversioned",model_version="1.11",} 364.884
ts_queue_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noop",model_version="default",} 82.349
```

```console
curl "http://127.0.0.1:8082/metrics?name[]=ts_inference_latency_microseconds&name[]=ts_queue_latency_microseconds" --globoff

# HELP ts_inference_latency_microseconds Cumulative inference duration in microseconds
# TYPE ts_inference_latency_microseconds counter
ts_inference_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noopversioned",model_version="1.11",} 1990.348
ts_inference_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noop",model_version="default",} 2032.411
# HELP ts_queue_latency_microseconds Cumulative queue duration in microseconds
# TYPE ts_queue_latency_microseconds counter
ts_queue_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noopversioned",model_version="1.11",} 364.884
ts_queue_latency_microseconds{uuid="d5f84dfb-fae8-4f92-b217-2f385ca7470b",model_name="noop",model_version="default",} 82.349
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
Navigate to http://localhost:9090/ on a browser to execute queries and create graphs 

<img width="1231" alt="PrometheusServer" src="https://user-images.githubusercontent.com/880376/86984450-806fc680-c143-11ea-9ae2-f2ef42f24f4c.png">

#### Grafana

Once you have the Torchserve and Prometheus servers running, you can further [setup](https://prometheus.io/docs/visualization/grafana/) Grafana, point it to Prometheus server and navigate to http://localhost:3000/ to create dashboards and graphs.

You can use command given below to start Grafana - 
`sudo systemctl daemon-reload && sudo systemctl enable grafana-server && sudo systemctl start grafana-server`

<img width="1220" alt="Screen Shot 2020-07-08 at 5 51 57 PM" src="https://user-images.githubusercontent.com/880376/86984550-c4fb6200-c143-11ea-9434-09d4d43dd6d4.png">
