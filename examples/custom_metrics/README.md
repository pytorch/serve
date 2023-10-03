# Torchserve custom metrics with prometheus support

In this example, we show how to use a pre-trained custom MNIST model and export custom metrics using prometheus.

We use the following pytorch example of MNIST model for digit recognition : https://github.com/pytorch/examples/tree/master/mnist

Run the commands given in following steps from the root directory of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

## Steps

- Step 1: In this example we add the following custom metrics and access them in prometheus format via the [metrics API endpoint](https://github.com/pytorch/serve/blob/master/docs/metrics_api.md):
  - InferenceRequestCount
  - InitializeCallCount
  - PreprocessCallCount
  - PostprocessCallCount
  - RequestBatchSize
  - SizeOfImage
  - HandlerMethodTime
  - ExamplePercentMetric

  The custom metrics configuration file `metrics.yaml` in this example builds on top of the [default metrics configuration file](https://github.com/pytorch/serve/blob/master/ts/configs/metrics.yaml) to include the custom metrics listed above.
  The `config.properties` file in this example configures torchserve to use the custom metrics configuration file and sets the `metrics_mode` to `prometheus`. The custom handler
  `mnist_handler.py` updates the metrics listed above.

  Refer: [Custom Metrics](https://github.com/pytorch/serve/blob/master/docs/metrics.md#custom-metrics-api)\
  Refer: [Custom Handler](https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handlers)

- Step 2: Create a torch model archive using the torch-model-archiver utility.

  ```bash
  torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler examples/custom_metrics/mnist_handler.py
  ```

- Step 3: Register the model to torchserve using the above model archive file.

  ```bash
  mkdir model_store
  mv mnist.mar model_store/
  torchserve --ncs --start --model-store model_store --ts-config examples/custom_metrics/config.properties --models mnist=mnist.mar
  ```

- Step 4: Make Inference request

  ```bash
  curl http://127.0.0.1:8080/predictions/mnist -T examples/image_classifier/mnist/test_data/0.png
  ```

- Step 5: Install prometheus using the instructions [here](https://prometheus.io/download/#prometheus).

- Step 6: Create a minimal `prometheus.yaml` config file as below and run `./prometheus --config.file=prometheus.yaml`.

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

- Step 7: Test metrics API endpoint
```console
curl http://127.0.0.1:8082/metrics

# HELP Requests2XX Torchserve prometheus counter metric with unit: Count
# TYPE Requests2XX counter
Requests2XX{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 1.0
# HELP PredictionTime Torchserve prometheus gauge metric with unit: ms
# TYPE PredictionTime gauge
PredictionTime{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 62.78
# HELP DiskUsage Torchserve prometheus gauge metric with unit: Gigabytes
# TYPE DiskUsage gauge
DiskUsage{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 8.438858032226562
# HELP WorkerLoadTime Torchserve prometheus gauge metric with unit: Milliseconds
# TYPE WorkerLoadTime gauge
WorkerLoadTime{WorkerName="W-9000-mnist_1.0",Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 7425.0
# HELP Requests5XX Torchserve prometheus counter metric with unit: Count
# TYPE Requests5XX counter
# HELP CPUUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE CPUUtilization gauge
CPUUtilization{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 100.0
# HELP WorkerThreadTime Torchserve prometheus gauge metric with unit: Milliseconds
# TYPE WorkerThreadTime gauge
WorkerThreadTime{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 3.0
# HELP DiskAvailable Torchserve prometheus gauge metric with unit: Gigabytes
# TYPE DiskAvailable gauge
DiskAvailable{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 308.94310760498047
# HELP ts_inference_requests_total Torchserve prometheus counter metric with unit: Count
# TYPE ts_inference_requests_total counter
ts_inference_requests_total{model_name="mnist",model_version="default",hostname="88665a372f4b.ant.amazon.com",} 1.0
# HELP GPUMemoryUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE GPUMemoryUtilization gauge
# HELP HandlerTime Torchserve prometheus gauge metric with unit: ms
# TYPE HandlerTime gauge
HandlerTime{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 62.64
# HELP ts_inference_latency_microseconds Torchserve prometheus counter metric with unit: Microseconds
# TYPE ts_inference_latency_microseconds counter
ts_inference_latency_microseconds{model_name="mnist",model_version="default",hostname="88665a372f4b.ant.amazon.com",} 64694.367
# HELP MemoryUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE MemoryUtilization gauge
MemoryUtilization{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 53.1
# HELP MemoryAvailable Torchserve prometheus gauge metric with unit: Megabytes
# TYPE MemoryAvailable gauge
MemoryAvailable{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 7677.29296875
# HELP PostprocessCallCount Torchserve prometheus counter metric with unit: count
# TYPE PostprocessCallCount counter
PostprocessCallCount{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 1.0
# HELP ExamplePercentMetric Torchserve prometheus histogram metric with unit: percent
# TYPE ExamplePercentMetric histogram
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="0.005",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="0.01",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="0.025",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="0.05",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="0.075",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="0.1",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="0.25",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="0.5",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="0.75",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="1.0",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="2.5",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="5.0",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="7.5",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="10.0",} 0.0
ExamplePercentMetric_bucket{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",le="+Inf",} 1.0
ExamplePercentMetric_count{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 1.0
ExamplePercentMetric_sum{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 50.0
# HELP GPUUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE GPUUtilization gauge
# HELP MemoryUsed Torchserve prometheus gauge metric with unit: Megabytes
# TYPE MemoryUsed gauge
MemoryUsed{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 7903.734375
# HELP QueueTime Torchserve prometheus gauge metric with unit: Milliseconds
# TYPE QueueTime gauge
QueueTime{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 0.0
# HELP ts_queue_latency_microseconds Torchserve prometheus counter metric with unit: Microseconds
# TYPE ts_queue_latency_microseconds counter
ts_queue_latency_microseconds{model_name="mnist",model_version="default",hostname="88665a372f4b.ant.amazon.com",} 115.79
# HELP PreprocessCallCount Torchserve prometheus counter metric with unit: count
# TYPE PreprocessCallCount counter
PreprocessCallCount{ModelName="mnist",Hostname="88665a372f4b.ant.amazon.com",} 1.0
# HELP RequestBatchSize Torchserve prometheus gauge metric with unit: count
# TYPE RequestBatchSize gauge
RequestBatchSize{ModelName="mnist",Hostname="88665a372f4b.ant.amazon.com",} 1.0
# HELP SizeOfImage Torchserve prometheus gauge metric with unit: kB
# TYPE SizeOfImage gauge
SizeOfImage{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 0.265625
# HELP Requests4XX Torchserve prometheus counter metric with unit: Count
# TYPE Requests4XX counter
# HELP HandlerMethodTime Torchserve prometheus gauge metric with unit: ms
# TYPE HandlerMethodTime gauge
HandlerMethodTime{MethodName="preprocess",ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 25.554895401000977
# HELP InitializeCallCount Torchserve prometheus counter metric with unit: count
# TYPE InitializeCallCount counter
InitializeCallCount{ModelName="mnist",Level="Model",Hostname="88665a372f4b.ant.amazon.com",} 1.0
# HELP DiskUtilization Torchserve prometheus gauge metric with unit: Percent
# TYPE DiskUtilization gauge
DiskUtilization{Level="Host",Hostname="88665a372f4b.ant.amazon.com",} 2.7
# HELP GPUMemoryUsed Torchserve prometheus gauge metric with unit: Megabytes
# TYPE GPUMemoryUsed gauge
# HELP InferenceRequestCount Torchserve prometheus counter metric with unit: count
# TYPE InferenceRequestCount counter
InferenceRequestCount{Hostname="88665a372f4b.ant.amazon.com",} 1.0
```

- Step 8: Navigate to `http://localhost:9090/` on a browser to execute queries and create graphs

<img width="1777" alt="Screenshot 2023-08-03 at 6 46 47 PM" src="https://github.com/pytorch/serve/assets/5276346/a87d6ee4-a760-4da8-b0f6-d461df7e500d">
